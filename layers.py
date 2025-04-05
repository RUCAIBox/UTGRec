import numpy as np
import torch
import time
from sklearn.cluster import KMeans
from torch import nn
import torch.nn.functional as F
import torch.distributed as distributed

import torch
from transformers.activations import ACT2FN


class AllGatherFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        tensor = tensor.contiguous()
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(distributed.get_world_size())]

        distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone().contiguous()
        distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = distributed.get_rank() * ctx.batch_size
        idx_to = (distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to].contiguous()

def get_quantizer(model_args):

    if model_args.code_type == "multi":
        quantizer = MultiVQLayer(
            n_codebooks=model_args.code_num,
            codebook_size=model_args.codebook_size,
            codebook_dim=model_args.codebook_dim,
            vq_type=model_args.vq_type,
            beta=model_args.beta,
            decay=model_args.ema_decay,
            sk_epsilon=model_args.sk_epsilon,
            fix_codebook=model_args.fix_codebook,
            fix_code_embs=model_args.sim_vq_fix_code_embs,
        )
    elif model_args.code_type == "tree":
        quantizer = TreeVQLayer(
            n_codebooks=model_args.code_num,
            codebook_size=model_args.codebook_size,
            codebook_dim=model_args.codebook_dim,
            vq_type=model_args.vq_type,
            beta=model_args.beta,
            decay=model_args.ema_decay,
            sk_epsilon=model_args.sk_epsilon,
            fix_codebook=model_args.fix_codebook,
            fix_code_embs=model_args.sim_vq_fix_code_embs,
        )
    else:
        raise NotImplementedError("Only support multi or tree codebook.")

    return quantizer



class MLPLayers(nn.Module):


    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(ACT2FN[activation])

        mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)

    def forward(self, x):
        return self.mlp_layers(x)








class MultiVQLayer(nn.Module):

    def __init__(self,  n_codebooks, codebook_size, codebook_dim,
                 beta=0.25, vq_type="ema", sk_epsilon=-1, fix_codebook=False, decay=0.99, fix_code_embs=False):
        super(MultiVQLayer, self).__init__()

        self.n_codebooks = n_codebooks

        if isinstance(codebook_size, int):
            self.codebook_sizes = [codebook_size] * n_codebooks
        elif isinstance(codebook_size, list):
            if len(codebook_size) == n_codebooks:
                self.codebook_sizes = codebook_size
            else:
                raise ValueError("codebook_size must be an int or a list of int with length equal to n_codebooks")



        if vq_type=="vq": # codebook_size, codebook_dim, beta = 0.25, sk_epsilon = -1, fix_codebook = False
            self.vq_layers = nn.ModuleList([
                VQLayer(codebook_size=codebook_size, codebook_dim=codebook_dim, beta=beta, sk_epsilon=sk_epsilon, fix_codebook=fix_codebook)
                for codebook_size in self.codebook_sizes
            ])
        elif vq_type=="ema": # codebook_size, codebook_dim, beta=0.25, sk_epsilon=-1, fix_codebook=False, decay=0.99
            self.vq_layers = nn.ModuleList([
                EMAVQLayer(codebook_size=codebook_size, codebook_dim=codebook_dim, beta=beta, sk_epsilon=sk_epsilon, fix_codebook=fix_codebook, decay=decay)
                for codebook_size in self.codebook_sizes
            ])
        elif vq_type == "simvq": # codebook_size, codebook_dim, beta=0.25, sk_epsilon=-1, fix_codebook=False, fix_code_embs=False
            self.vq_layers = nn.ModuleList([
                SimVQLayer(codebook_size=codebook_size, codebook_dim=codebook_dim, beta=beta, sk_epsilon=sk_epsilon, fix_codebook=fix_codebook, fix_code_embs=fix_code_embs)
                for codebook_size in self.codebook_sizes
            ])
        else:
            raise NotImplementedError("Only support vq, ema or simvq quantization.")

    def forward(self, x: torch.Tensor):

        batch_size, code_num, _ = x.shape
        assert code_num <= self.n_codebooks

        quantized_x = torch.zeros_like(x)
        code_x = torch.zeros(batch_size, code_num, device=x.device)
        quant_losses = []
        num_unused_codes = 0

        for vq_layer, level in zip(self.vq_layers, range(self.n_codebooks)):
            quant, quant_loss, unused_codes, codes = vq_layer(x[:,level])
            quantized_x[:, level] = quant
            code_x[:, level] = codes

            quant_losses.append(quant_loss)
            num_unused_codes += unused_codes


        mean_quant_loss = sum(quant_losses) / len(quant_losses)

        return quantized_x.contiguous(), mean_quant_loss, num_unused_codes, code_x.contiguous()


    def get_topk_tail_token(self, x, topk=1, used=False):

        return self.vq_layers[-1].get_topk_tail_token(x, topk, used)



class TreeVQLayer(nn.Module):

    def __init__(self,  n_codebooks, codebook_size, codebook_dim,
                 beta=0.25, vq_type="ema", sk_epsilon=-1, fix_codebook=False, decay=0.99, fix_code_embs=False):
        super(TreeVQLayer, self).__init__()

        self.n_codebooks = n_codebooks
        if isinstance(codebook_size, int):
            self.root_codebook_size = codebook_size
            self.shared_codebook_size = codebook_size * (n_codebooks - 1)
        elif isinstance(codebook_size, list):
            if len(codebook_size) == 2:
                self.root_codebook_size = codebook_size[0]
                self.shared_codebook_size = codebook_size[1]
            else:
                raise ValueError("codebook_size must be an int or a list of two int")

        if vq_type=="vq":
            self.root_vq_layer = VQLayer(codebook_size=self.root_codebook_size, codebook_dim=codebook_dim, beta=beta, sk_epsilon=sk_epsilon, fix_codebook=fix_codebook)
            self.shared_vq_layer = VQLayer(codebook_size=self.shared_codebook_size, codebook_dim=codebook_dim, beta=beta, sk_epsilon=sk_epsilon, fix_codebook=fix_codebook)
        elif vq_type=="ema":
            self.root_vq_layer = EMAVQLayer(codebook_size=self.root_codebook_size, codebook_dim=codebook_dim, beta=beta, sk_epsilon=sk_epsilon, fix_codebook=fix_codebook, decay=decay)
            self.shared_vq_layer = EMAVQLayer(codebook_size=self.shared_codebook_size, codebook_dim=codebook_dim, beta=beta, sk_epsilon=sk_epsilon, fix_codebook=fix_codebook, decay=decay)
        elif vq_type == "simvq":
            self.root_vq_layer = SimVQLayer(codebook_size=self.root_codebook_size, codebook_dim=codebook_dim, beta=beta, sk_epsilon=sk_epsilon, fix_codebook=fix_codebook, fix_code_embs=fix_code_embs)
            self.shared_vq_layer = SimVQLayer(codebook_size=self.shared_codebook_size, codebook_dim=codebook_dim, beta=beta, sk_epsilon=sk_epsilon, fix_codebook=fix_codebook, fix_code_embs=fix_code_embs)
        else:
            raise NotImplementedError("Only support vq, ema or simvq quantization.")


    def forward(self, x: torch.Tensor):

        batch_size, code_num, _ = x.shape
        assert code_num <= self.n_codebooks

        quantized_x = torch.zeros_like(x)
        code_x = torch.zeros(batch_size, code_num, device=x.device)
        quant_losses = []
        num_unused_codes = 0

        quant, quant_loss, unused_codes, codes = self.root_vq_layer(x[:, 0])
        quantized_x[:, 0] = quant
        code_x[:, 0] = codes
        quant_losses.append(quant_loss)
        num_unused_codes += unused_codes


        if code_num > 1:
            quant, quant_loss, unused_codes, codes = self.shared_vq_layer(x[:, 1:].contiguous())
            quantized_x[:, 1:] = quant
            code_x[:, 1:] = codes
            quant_losses.append(quant_loss)
            num_unused_codes += unused_codes

        mean_quant_loss = sum(quant_losses) / len(quant_losses)
        # sum_quant_loss = sum(quant_losses)

        return quantized_x.contiguous(), mean_quant_loss, num_unused_codes, code_x.contiguous()


    def get_topk_tail_token(self, x, topk=1, used=False):

        return self.shared_vq_layer.get_topk_tail_token(x, topk, used)









class VQLayer(nn.Module):


    def __init__(self, codebook_size, codebook_dim, beta=0.25, sk_epsilon=-1, fix_codebook=False):
        super(VQLayer, self).__init__()

        assert isinstance(codebook_size, int)
        self.n_embed = codebook_size
        self.dim = codebook_dim
        self.beta = beta
        self.sk_epsilon = sk_epsilon
        self.fix_codebook = fix_codebook

        self.embed = nn.Embedding(self.n_embed, self.dim)
        if self.fix_codebook:
            self.initted = True
            for param in self.embed.parameters():
                param.requires_grad = False
        else:
            self.initted = False

    def _init_emb(self, x):

        if not distributed.is_initialized():
            print("Init CodeBook")
        elif distributed.is_initialized() and distributed.get_rank() == 0:
            print("Init CodeBook")
        else:
            pass


        device = x.device
        x = x.contiguous()

        if distributed.is_initialized():
            x_list = [torch.empty_like(x) for _ in range(distributed.get_world_size())]
            distributed.all_gather(x_list, x)
            x = torch.cat(x_list, dim=0)

        if x.size(0) < self.n_embed:
            vectors = self._tile_with_noise(x, self.n_embed)
            sampled_values = vectors[torch.randperm(vectors.size(0), device=vectors.device)][:self.n_embed].clone().detach().contiguous()
            if distributed.is_initialized():
                distributed.broadcast(sampled_values, 0)
            init_embed = sampled_values
        else:
            kmeans = KMeans(n_clusters=self.n_embed, n_init='auto').fit(x.to(torch.float).detach().cpu().numpy())
            centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device).view(self.n_embed, self.dim).contiguous()
            if distributed.is_initialized():
                distributed.broadcast(centers, 0)
                # distributed.broadcast(cluster_size, 0)
            init_embed = centers


        self._copy_init_embed(init_embed)
        self.initted = True

    def _copy_init_embed(self, init_embed):
        self.embed.weight.data.copy_(init_embed)


    def get_code_embs(self):
        return self.embed.weight


    @staticmethod
    def center_distance(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    @torch.no_grad()
    def sinkhorn(self, distances, epsilon=0.003, iterations=5):
        Q = torch.exp(- distances / epsilon)

        B = Q.shape[0]  # number of samples to assign
        K = Q.shape[1]  # how many centroids per block (usually set to 256)

        # make the matrix sums to 1
        sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)

        if distributed.is_initialized():
            B *= distributed.get_world_size()
            distributed.all_reduce(sum_Q, op=distributed.ReduceOp.SUM)

        Q /= sum_Q
        # print(Q.sum())
        for _ in range(iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_0 = torch.sum(Q, dim=0, keepdim=True)
            if distributed.is_initialized():
                distributed.all_reduce(sum_0, op=distributed.ReduceOp.SUM)
            Q /= sum_0
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q

    @torch.no_grad()
    def _tile_with_noise(self, x, target_n):
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.1 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x




    def forward(self, x: torch.Tensor):

        latent = x.view(-1, self.dim)

        if not self.initted and self.training and not self.fix_codebook:
            self._init_emb(latent)

        code_embs = self.get_code_embs()

        dist = (
            latent.pow(2).sum(1, keepdim=True)
            - 2 * latent @ code_embs.t()
            + code_embs.pow(2).sum(1, keepdim=True).t()
        )


        if self.sk_epsilon != -1 and self.training:
            dist = self.center_distance(dist)
            dist = dist.double()
            Q = self.sinkhorn(dist, self.sk_epsilon)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise RuntimeError(f"Sinkhorn Algorithm returns nan/inf values.")
            embed_ind = torch.argmax(Q, dim=-1)
        else:
            embed_ind = torch.argmin(dist, dim=-1)
            # _, embed_ind = (-dist).max(1)



        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(latent.dtype)
        embed_onehot_sum = embed_onehot.sum(0)
        if distributed.is_initialized():
            distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
        unused_codes = (embed_onehot_sum == 0).sum().item()




        x_q = F.embedding(embed_ind, code_embs).view(x.shape)

        quant_loss = F.mse_loss(x_q, x.detach()) + self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()

        embed_ind = embed_ind.view(*x.shape[:-1])

        return x_q, quant_loss, unused_codes, embed_ind

    def get_topk_tail_token(self, x, topk=1, used=False):

        latent = x.view(-1, self.dim)

        code_embs = self.get_code_embs()

        dist = (
                latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ code_embs.t()
                + code_embs.pow(2).sum(1, keepdim=True).t()
        )
        dist = -dist
        topk_prob, topk_idx = dist.topk(topk, dim=-1)

        if used:
            codes = topk_idx[:, -1]
            fix = torch.zeros_like(codes, dtype=torch.bool)
        else:
            if not (topk_idx[:, -2] == topk_idx[0, -2]).all():
                codes = topk_idx[:, -1]
                fix = torch.zeros_like(codes, dtype=torch.bool)
            else:
                # If the current code is not occupied, the closest item to the center remains unchanged
                fix = (topk_prob[:, -2] == topk_prob[:, -2].max())
                # If there are multiple items with the same distance, randomly maintain one constant
                if fix.sum() > 1:
                    ind = fix.cpu().numpy().tolist().index(True)
                    fix[ind+1:] = False

                codes = torch.where(fix, topk_idx[:, -2], topk_idx[:, -1])

        codes = codes.view(x.shape[:-1])

        return codes, fix



class EMAVQLayer(VQLayer):

    def __init__(self, codebook_size, codebook_dim, beta=0.25, sk_epsilon=-1, fix_codebook=False, decay=0.99):
        super(EMAVQLayer, self).__init__(codebook_size, codebook_dim, beta, sk_epsilon, fix_codebook)

        self.decay = decay
        self.eps = 1e-5

        embed = torch.zeros(self.n_embed, self.dim)
        self.embed = nn.Parameter(embed, requires_grad=False)
        nn.init.xavier_normal_(self.embed)
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("cluster_size", torch.ones(self.n_embed))
        if fix_codebook:
            self.initted=True


    def _copy_init_embed(self, init_embed):
        self.embed.data.copy_(init_embed)
        self.embed_avg.data.copy_(init_embed)
        self.cluster_size.data.copy_(torch.ones(self.n_embed, device=init_embed.device))

    def get_code_embs(self):
        return self.embed


    def forward(self, x: torch.Tensor):

        latent = x.view(-1, self.dim)

        if not self.initted and self.training and not self.fix_codebook:
            self._init_emb(latent)

        code_embs = self.get_code_embs()

        dist = (
                latent.pow(2).sum(1, keepdim=True)
                - 2 * latent @ code_embs.t()
                + code_embs.pow(2).sum(1, keepdim=True).t()
        )

        if self.sk_epsilon != -1 and self.training:
            dist = self.center_distance(dist)
            dist = dist.double()
            Q = self.sinkhorn(dist, self.sk_epsilon)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                raise RuntimeError(f"Sinkhorn Algorithm returns nan/inf values.")
            embed_ind = torch.argmax(Q, dim=-1)
        else:
            embed_ind = torch.argmin(dist, dim=-1)
            # _, embed_ind = (-dist).max(1)


        x_q = F.embedding(embed_ind, code_embs).view(x.shape)

        if self.training and not self.fix_codebook:

            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(latent.dtype)
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = embed_onehot.t() @ latent
            if distributed.is_initialized():
                distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
                distributed.all_reduce(embed_sum, op=distributed.ReduceOp.SUM)

            unused_codes = (embed_onehot_sum == 0).sum().item()

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(
                embed_sum, alpha=1 - self.decay
            )

            n = self.cluster_size.sum()
            norm_w = (
                n * (self.cluster_size + self.eps) / (n + self.n_embed * self.eps)
            )
            embed_normalized = self.embed_avg / norm_w.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
        else:
            embed_onehot = F.one_hot(embed_ind, self.n_embed).type(latent.dtype)
            embed_onehot_sum = embed_onehot.sum(0)
            if distributed.is_initialized():
                distributed.all_reduce(embed_onehot_sum, op=distributed.ReduceOp.SUM)
            unused_codes = (embed_onehot_sum == 0).sum().item()


        quant_loss = self.beta * F.mse_loss(x, x_q.detach())
        x_q = x + (x_q - x).detach()

        embed_ind = embed_ind.view(*x.shape[:-1])

        return x_q, quant_loss, unused_codes, embed_ind


class SimVQLayer(VQLayer):

    def __init__(self, codebook_size, codebook_dim, beta=0.25, sk_epsilon=-1, fix_codebook=False, fix_code_embs=False):
        super(SimVQLayer, self).__init__(codebook_size, codebook_dim, beta, sk_epsilon, fix_codebook)

        nn.init.normal_(self.embed.weight, mean=0, std=self.dim ** -0.5)
        self.initted=True

        self.embed_proj = nn.Linear(self.dim, self.dim, bias=False)
        # self.embed_proj = nn.Linear(self.dim, self.dim)


        if fix_code_embs:
            for param in self.embed.parameters():
                param.requires_grad = False

        if self.fix_codebook:
            for param in self.embed.parameters():
                param.requires_grad = False
            for param in self.embed_proj.parameters():
                param.requires_grad = True

    def get_code_embs(self):
        return self.embed_proj(self.embed.weight)

    def _init_emb(self, x):
        pass

    def _copy_init_embed(self, init_embed):
        pass




