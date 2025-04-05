import os
from dataclasses import dataclass, field
from types import NoneType
from typing import Optional, List, Union

import transformers



@dataclass
class ModelArguments:


    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )

    lora: bool = field(
        default=True,
        metadata={"help": "Whether to use lora training."},
    )
    lora_r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_target_modules: Optional[str] = field(default="q_proj,k_proj,v_proj,o_proj")
    lora_modules_to_save: Optional[str] = field(default="embed_tokens,lm_head")

    vae_embed_dim: Optional[int] = field(
        default=16,
        metadata={"help": "The dimension of VAE embeddings."},
    )
    vae_path: Optional[str] = field(
        default="vae/kl16.ckpt",
        metadata={"help": "Path to pretrained VAE model."},
    )
    diffloss_w: Optional[int] = field(
        default=1024,
        metadata={"help": "The weight of diffusion loss."},
    )
    diffloss_d: Optional[int] = field(
        default=3,
        metadata={"help": "The depth of diffusion loss."},
    )
    diff_sampling_steps: Optional[str] = field(
        default="100",
        metadata={"help": "The sampling steps of diffusion loss."},
    )
    diffusion_batch_mul: Optional[int] = field(
        default=4,
        metadata={"help": "The repeat num for diffusion loss."},
    )
    diffloss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "The weight of diffusion loss."},
    )

    cl_loss_weight: Optional[float] = field(
        default=0.0,
        metadata={"help": "The weight of contrastive loss."},
    )

    pos_recon_loss_weight: Optional[float] = field(
        default=0.0,
        metadata={"help": "The weight of positive item reconstruction loss."},
    )

    vq_loss_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "The weight of vq loss."},
    )

    code_type: Optional[str] = field(
        default="tree",
        metadata={"help": "The type of codebook. multi or tree."},
    )

    vq_type: Optional[str] = field(
        default="simvq",
        metadata={"help": "The type of quantization. vq, ema or simvq."},
    )

    vq_warmup_steps: Optional[int] = field(
        default=-1,
        metadata={"help": "The warmup steps of embeddings."},
    )

    code_num: Optional[int] = field(
        default=3,
        metadata={"help": "The number of code tokens."},
    )
    codebook_size: Optional[str] = field(
        default="256",
        metadata={"help": "The size of codebook."},
    )
    codebook_dim: Optional[int] = field(
        default=128,
        metadata={"help": "The dimension of codebook vector."},
    )
    sk_epsilon: Optional[float] = field(
        default=-1,
        metadata={"help": "The epsilon of Sinkhorn Algorithm."},
    )
    beta: Optional[float] = field(
        default=0.25,
        metadata={"help": "The weight of quantization loss."},
    )
    ema_decay: Optional[float] = field(
        default=0.99,
        metadata={"help": "The decay for ema update."}
    )
    sim_vq_fix_code_embs: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to fix code embeddings for SimVQ."},
    )

    adapter_hidden_sizes: Optional[str] = field(
        default="768",
        metadata={"help": "The hidden sizes of adapter layers."},
    )

    decoder_layers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of decoder layers."},
    )
    decoder_hidden_size: Optional[int] = field(
        default=1536,
        metadata={"help": "The hidden size of decoder."},
    )
    decoder_attention_heads: Optional[int] = field(
        default=12,
        metadata={"help": "The number of decoder attention heads."}
    )

    model_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained Tokenizer for final token generation."}
    )

    fix_encoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to fix encoder weights."},
    )

    fix_decoder: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to fix decoder weights."},
    )

    fix_codebook: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to fix codebook weights."},
    )








@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_path: Optional[str] = field(
        default="data/AmazonReviews2023/",
        metadata={"help": "The data directory."}
    )

    datasets: Optional[str] = field(
        default="Musical_Instruments",
        metadata={"help": "The dataset name."}
    )
    valid_datasets: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset name."}
    )

    dec_mlm_probability: Optional[float] = field(
        default=1.0,
        metadata={"help": "The probability of masking a word."}
    )

    max_feature_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum item text feature length after tokenization."},
    )

    tokens_file: Optional[str] = field(
        default="test.sem_ids",
        metadata={"help": "The file to save item tokens."},
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    seed: Optional[int] = field(default=2024)
    cache_dir: Optional[str] = field(default=None)
    optim: Optional[str] = field(default="adamw_torch")
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."},
    )
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    dataloader_num_workers: Optional[int] = field(default=8)
    warmup_ratio: Optional[float] = field(default=0.03)
    weight_decay: Optional[float] = field(default=0.01)
    load_best_model_at_end: Optional[bool] = field(default=False)
    save_total_limit: Optional[int] = field(default=10)
    logging_steps: Optional[float] = field(default=10)
    report_to: Optional[List[str]] = field(default="none")
    output_dir: Optional[str] = field(
        default="ckpt/test/",
        metadata={"help": "The output directory where the model checkpoints will be written."}
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use bfloat16."},
    )
    llm_learning_rate: Optional[float] = field(
        default=None,
        metadata={"help": "The initial learning rate for llm."}
    )



