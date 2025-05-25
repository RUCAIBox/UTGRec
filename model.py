import copy
import os
import time
import warnings
from typing import Union, Optional, Callable, Tuple, List

import torch
from torch import Tensor
from peft import LoraConfig, get_peft_model
from torch import nn
import torch.distributed as distributed
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, Qwen2VLModel, Qwen2VLForConditionalGeneration, GPT2LMHeadModel
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast

from diffloss import DiffLoss
from layers import *
import torch.nn.functional as F

from vae import AutoencoderKL


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["visual"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            if "lm_head" in name:
                continue
            lora_module_names.add(name)

    return list(lora_module_names)


class MMTokenizerModel(PreTrainedModel):
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
            self,
            llm,
            model_args,
    ):
        super(MMTokenizerModel, self).__init__(llm.config)
        self.llm = llm
        self.model_args = model_args
        self.lora = model_args.lora

        self.vocab_size = self.llm.vocab_size


        if model_args.lora:

            peft_config = LoraConfig(
                inference_mode=False,
                r=model_args.lora_r,
                target_modules=find_all_linear_names(self.llm),
                modules_to_save=model_args.lora_modules_to_save.split(","),
                lora_alpha=model_args.lora_alpha,
                bias="none",
                lora_dropout=model_args.lora_dropout,
            )
            self.llm = get_peft_model(self.llm, peft_config)
        else:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False
            for name, param in self.llm.get_input_embeddings().named_parameters():
                param.requires_grad = True
            for name, param in self.llm.get_output_embeddings().named_parameters():
                param.requires_grad = True

        if model_args.fix_encoder:
            for name, param in self.llm.named_parameters():
                param.requires_grad = False



        # VAE for latent diffusion
        self.vae = AutoencoderKL(embed_dim=model_args.vae_embed_dim, ch_mult=(1, 1, 2, 2, 4),
                                 ckpt_path=model_args.vae_path).eval()
        self.vae_stride = 16
        for param in self.vae.parameters():
            param.requires_grad = False



        # codebook
        self.code_num = model_args.code_num
        codebook_size = model_args.codebook_size.strip().split(",")
        codebook_size = [int(_) for _ in codebook_size]
        if len(codebook_size) == 1:
            codebook_size = codebook_size[0]
        model_args.codebook_size = codebook_size
        self.codebook_size = model_args.codebook_size
        self.codebook_dim = model_args.codebook_dim
        self.quantizer = get_quantizer(model_args)
        self.vq_loss_weight = model_args.vq_loss_weight


        # decoder
        if model_args.lora:
            self.decoder_embeddings = self.llm.base_model.model.model.embed_tokens
            self.lm_head = self.llm.base_model.model.lm_head
        else:
            self.decoder_embeddings = self.llm.model.embed_tokens
            self.lm_head = self.llm.lm_head

        self.decoder_config = copy.deepcopy(self.llm.config)
        self.decoder_config.num_hidden_layers = model_args.decoder_layers
        self.decoder_config._attn_implementation = "eager"
        self.decoder_config.is_decoder = False
        self.decoder_config.hidden_size = model_args.decoder_hidden_size
        self.decoder_config.intermediate_size = 4 * model_args.decoder_hidden_size
        self.decoder_config.num_attention_heads = model_args.decoder_attention_heads

        if model_args.adapter_hidden_sizes is None or model_args.adapter_hidden_sizes.strip()=="None":
            self.adapter_l2c = nn.Linear(self.llm.config.hidden_size, self.codebook_dim)
        else:
            adapter_hidden_sizes = model_args.adapter_hidden_sizes.strip().split(",")
            adapter_hidden_sizes = [int(_) for _ in adapter_hidden_sizes]

            adapter_hidden_sizes = [self.llm.config.hidden_size] + adapter_hidden_sizes + [self.codebook_dim]
            self.adapter_l2c = MLPLayers(adapter_hidden_sizes, 0.1, self.llm.config.hidden_act)


        self.lm_decoder = Qwen2VLModel(self.decoder_config)
        self.lm_decoder.embed_tokens = None
        self.vision_decoder = Qwen2VLModel(self.decoder_config)
        self.vision_decoder.embed_tokens = None


        self.adapter_c2l = nn.Linear(self.codebook_dim, self.decoder_config.hidden_size)
        if self.llm.config.hidden_size != self.decoder_config.hidden_size:
            self.text_down = nn.Linear(self.llm.config.hidden_size, self.decoder_config.hidden_size)
            self.text_up = nn.Linear(self.decoder_config.hidden_size, self.llm.config.hidden_size)
        else:
            self.text_down = nn.Identity()
            self.text_up = nn.Identity()

        self.vision_mask_token = nn.Parameter(torch.randn(1, 1, self.decoder_config.hidden_size), requires_grad=True)
        self.vision_proj = nn.Linear(model_args.vae_embed_dim, self.decoder_config.hidden_size)



        if model_args.fix_decoder:
            for name, param in self.lm_decoder.named_parameters():
                param.requires_grad = False
            for name, param in self.vision_decoder.named_parameters():
                param.requires_grad = False
            for name, param in self.lm_head:
                param.requires_grad = False



        self.cross_entropy = nn.CrossEntropyLoss()
        self.diffloss = DiffLoss(
            target_channels=model_args.vae_embed_dim,
            z_channels=self.decoder_config.hidden_size,
            width=model_args.diffloss_w,
            depth=model_args.diffloss_d,
            num_sampling_steps=model_args.diff_sampling_steps,
        )
        self.diffusion_batch_mul = model_args.diffusion_batch_mul



        self.diffloss_weight = model_args.diffloss_weight
        self.cl_loss_weight = model_args.cl_loss_weight
        self.pos_recon_loss_weight = model_args.pos_recon_loss_weight



        self.step = 0
        self.vq_warmup_steps = model_args.vq_warmup_steps


        self.adapter_l2c.apply(self.llm._init_weights)
        self.adapter_c2l.apply(self.llm._init_weights)
        self.lm_decoder.apply(self.llm._init_weights)
        self.vision_decoder.apply(self.llm._init_weights)
        self.text_down.apply(self.llm._init_weights)
        self.text_up.apply(self.llm._init_weights)
        self.vision_proj.apply(self.llm._init_weights)
        self.vision_mask_token.data.normal_(mean=0.0, std=0.02)



    def gradient_checkpointing_enable(self, **kwargs):
        self.llm.gradient_checkpointing_enable(**kwargs)
        self.llm.enable_input_require_grads()

    def shift_and_sub(self, x):

        a = x[:,:-1]
        b = x[:,1:]
        _x = torch.cat([x[:,0].unsqueeze(1), b - a ], dim=1).contiguous()
        return _x


    def reverse_shift_and_sub(self, _x):
        _, s, _ = _x.shape

        x = []
        x.append(_x[:, 0])
        for i in range(1, s):
            x.append(x[-1] + _x[:, i])
        x = torch.stack(x, dim=1).contiguous()

        return x


    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = 1
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def info_nce(self, a, b, tau=0.1):

        a = F.normalize(a, dim=-1, p=2)
        b = F.normalize(b, dim=-1, p=2)

        if distributed.is_initialized():
            a = AllGatherFunction.apply(a).contiguous()
            b = AllGatherFunction.apply(b).contiguous()

        sim = torch.mm(a, b.t()) / tau
        label = torch.arange(a.size(0), dtype=torch.long).to(a.device)
        loss = F.cross_entropy(sim, label)

        return loss


    def calculate_diffloss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss


    def calculate_cl_loss(self, x, y):

        assert x.shape[1] == y.shape[1]


        loss = 0
        num = x.shape[1]
        for i in range(num):

            a = x[:, i, :]
            b = y[:, i, :]
            loss += self.info_nce(a,b)

        loss = loss / num


        return loss


    def get_extended_attention_mask(
            self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None,
            dtype: torch.float = None
    ) -> Tensor:

        if dtype is None:
            dtype = self.dtype

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def encode_item(self, inputs):

        code_token_indices = inputs.pop("code_token_indices")

        llm_outputs = self.llm(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )
        hidden_states = llm_outputs.hidden_states[-1]

        B, seq_len, d = hidden_states.shape
        code_num = code_token_indices.shape[1]

        code_token_states = torch.gather(hidden_states, 1,
                                         code_token_indices.view(B, code_num, 1).expand(B, code_num, d))

        return code_token_states

    def decode_item(self, decoder_inputs, code_states):

        B, code_num, _ = code_states.shape

        decoder_input_embeddings = self.decoder_embeddings(decoder_inputs["input_ids"])
        decoder_input_embeddings = self.text_down(decoder_input_embeddings)
        decoder_input_states = torch.cat([code_states, decoder_input_embeddings], dim=1)

        decoder_code_token_attention_mask = torch.ones((B, code_num), device=decoder_inputs["attention_mask"].device)
        decoder_attention_mask = torch.cat([decoder_code_token_attention_mask, decoder_inputs["attention_mask"]], dim=1)
        # decoder_attention_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(2).expand((-1, -1, decoder_input_states.shape[1], -1))
        decoder_attention_mask = self.get_extended_attention_mask(decoder_attention_mask, decoder_input_states.shape[:-1])

        lm_decoder_outputs = self.lm_decoder(
            input_ids=None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_input_states,
            return_dict=True,
        )
        lm_decoder_hidden_states = lm_decoder_outputs.last_hidden_state
        lm_decoder_hidden_states = self.text_up(lm_decoder_hidden_states)

        logits = self.lm_head(lm_decoder_hidden_states)

        shift_logits = logits[..., code_num:, :].contiguous()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        # ingore the cls token
        shift_labels = decoder_inputs["labels"].contiguous().view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        lm_loss = self.cross_entropy(shift_logits, shift_labels)



        with torch.no_grad():
            latent_images = self.vae.encode(decoder_inputs["input_images"]).sample().mul_(0.2325).detach()

        vision_decoder_input_states = self.patchify(latent_images).to(self.dtype)
        gt_latents = vision_decoder_input_states.clone().detach()

        vision_decoder_input_states = self.vision_proj(vision_decoder_input_states)
        visionmask_tokens = self.vision_mask_token.repeat(vision_decoder_input_states.shape[0],
                                                          vision_decoder_input_states.shape[1], 1).to(vision_decoder_input_states.dtype)
        vision_decoder_input_states = torch.where(decoder_inputs["input_image_masks"].unsqueeze(-1).bool(), visionmask_tokens,
                                                  vision_decoder_input_states)
        vision_decoder_input_states = torch.cat([code_states, vision_decoder_input_states], dim=1)

        vision_seq_len = vision_decoder_input_states.shape[1]
        # vision_decoder_attention_mask = torch.ones((B, 1, vision_seq_len, vision_seq_len), device=vision_decoder_input_states.device)
        vision_decoder_attention_mask = torch.ones((B, vision_seq_len), device=vision_decoder_input_states.device)
        vision_decoder_attention_mask = self.get_extended_attention_mask(vision_decoder_attention_mask,
                                                                         vision_decoder_input_states.shape[:-1])

        vision_decoder_outputs = self.vision_decoder(
            input_ids=None,
            attention_mask=vision_decoder_attention_mask,
            inputs_embeds=vision_decoder_input_states,
            return_dict=True,
        )
        vision_decoder_hidden_states = vision_decoder_outputs.last_hidden_state
        vision_decoder_hidden_states = vision_decoder_hidden_states[..., code_num:, :].contiguous()

        diffloss = self.calculate_diffloss(z=vision_decoder_hidden_states, target=gt_latents,
                                           mask=decoder_inputs["input_image_masks"])


        return lm_loss, diffloss


    def forward(self,
                inputs,
                decoder_inputs,
                pos_inputs,
                pos_decoder_inputs,
                ):


        code_token_states = self.encode_item(inputs)

        cl_loss = torch.tensor(0, device=code_token_states.device)

        code_token_states = self.adapter_l2c(code_token_states)
        # B, code_num, _ = code_token_states.shape

        if self.cl_loss_weight > 0:
            pos_item_states = self.encode_item(pos_inputs)
            pos_item_states = self.adapter_l2c(pos_item_states)
            cl_loss = self.calculate_cl_loss(code_token_states, pos_item_states)


        if self.step <= self.vq_warmup_steps:
            quant_loss = torch.tensor(0, device=code_token_states.device)
            quantized_code_states = code_token_states



            # ensure gradient flow
            code_token_states = self.shift_and_sub(code_token_states)




            temp, _, _, _ = self.quantizer(code_token_states)
            quantized_code_states = quantized_code_states + 0*temp

            if self.step == self.vq_warmup_steps:
                if distributed.get_rank() == 0:
                    print("End the embedding warmup")
                self.quantizer.root_vq_layer.initted = False
                self.quantizer.shared_vq_layer.initted = False


            self.step += 1
        else:
            code_token_states = self.shift_and_sub(code_token_states)


            quantized_code_states, quant_loss, num_unused_codes, code_ids = self.quantizer(code_token_states)

            quantized_code_states = self.reverse_shift_and_sub(quantized_code_states)





            if distributed.get_rank() == 0 and self.training:
                print(f"\nnum_unused_codes: {num_unused_codes}")


        quantized_code_states = self.adapter_c2l(quantized_code_states)


        lm_loss, diffloss = self.decode_item(decoder_inputs, quantized_code_states)

        recon_loss = lm_loss + self.diffloss_weight * diffloss

        if self.pos_recon_loss_weight > 0:
            pos_lm_loss, pos_diffloss = self.decode_item(pos_decoder_inputs, quantized_code_states)
            pos_recon_loss = pos_lm_loss + self.diffloss_weight * pos_diffloss
        else:
            pos_lm_loss = torch.tensor(0, device=code_token_states.device)
            pos_diffloss = torch.tensor(0, device=code_token_states.device)
            pos_recon_loss = torch.tensor(0, device=code_token_states.device)


        loss = (recon_loss + self.pos_recon_loss_weight * pos_recon_loss +
                self.vq_loss_weight * quant_loss + self.cl_loss_weight * cl_loss)

        return (loss, )


    @torch.no_grad()
    def encode(self,
               inputs,
               **kwargs
        ):

        code_token_states = self.encode_item(inputs)
        code_token_states = self.adapter_l2c(code_token_states)
        B, code_num, _ = code_token_states.shape

        code_token_states = self.shift_and_sub(code_token_states)
        quantized_code_states, _, _, code_ids = self.quantizer(code_token_states)

        return quantized_code_states, code_ids, code_token_states

    @torch.no_grad()
    def get_topk_tail_token(self, code_token_states, topk=1, used=False):


        res = self.quantizer.get_topk_tail_token(code_token_states, topk, used)

        return res

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            max_shard_size: Union[int, str] = "5GB",
            safe_serialization: bool = True,
            variant: Optional[str] = None,
            token: Optional[Union[str, bool]] = None,
            save_peft_format: bool = True,
            **kwargs,
    ):
        super().save_pretrained(
            save_directory=save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=False,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs,
        )



def fix_qwen2vl_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    >>> messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.get_dtype())
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_mask = (
                (input_ids == self.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # The code is copied from https://github.com/huggingface/transformers/pull/33487
    if position_ids is None and input_ids is not None:
        position_ids, _ = self.get_rope_index(
            input_ids, image_grid_thw, video_grid_thw, attention_mask
        )

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=rope_deltas,
    )