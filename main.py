

import warnings
warnings.filterwarnings("ignore")

import logging
from trainer import CustomizedTrainer
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, Trainer, Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor, Qwen2Tokenizer


from transformers.trainer_utils import set_seed
from arguments import ModelArguments, DataArguments, TrainingArguments
from model import MMTokenizerModel, fix_qwen2vl_forward
from utils import *
from data import Collator, MASK_TOKEN



logger = logging.getLogger(__name__)

Qwen2VLForConditionalGeneration.forward = fix_qwen2vl_forward


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.code_num = model_args.code_num
    if training_args.llm_learning_rate is None:
        training_args.llm_learning_rate = training_args.learning_rate

    set_seed(training_args.seed)
    ensure_dir(training_args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if local_rank==0 else logging.WARN,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if local_rank==0:

        logger.info("Training/evaluation args %s", training_args)
        logger.info("Model args %s", model_args)
        logger.info("Data args %s", data_args)

    if ddp:
        device_map = {"": local_rank}
        device = torch.device("cuda", local_rank)
        training_args.ddp_find_unused_parameters = False
    else:
        device = torch.device("cuda")

    train_dataset = get_datasets(data_args, mode="train")
    valid_dataset = get_datasets(data_args, mode="valid")
    logger.info(f"Train item number: {len(train_dataset)}")
    if valid_dataset is not None:
        logger.info(f"Valid item number: {len(valid_dataset)}")
    else:
        training_args.eval_strategy = "no"
        training_args.eval_steps = None


    logger.info(f"Load base model from {model_args.model_name_or_path}")
    llm = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    # print(llm.model.norm.weight[:10])


    min_pixels = 128 * 28 * 28
    max_pixels = 1024 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        min_pixels = min_pixels,
        max_pixels = max_pixels,
    )
    tokenizer = processor.tokenizer



    code_tokens = [f'<CODE_{i}>' for i in range(data_args.code_num)]
    added_num = tokenizer.add_tokens(code_tokens + [MASK_TOKEN], special_tokens=True)
    tokenizer._mask_token = MASK_TOKEN
    tokenizer.padding_side = "right"
    # added_num = tokenizer.add_tokens(code_tokens, special_tokens=True)
    logger.info(f"Added {added_num} tokens")


    if local_rank == 0:
        tokenizer.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)

    processor.tokenizer = tokenizer

    llm.resize_token_embeddings(len(tokenizer))

    model = MMTokenizerModel(llm, model_args)



    model = model.to(device).to(torch.bfloat16)

    logger.info(model)


    if model_args.model_ckpt is not None and model_args.model_ckpt != "":
        logger.info(f"Load pretrained model from {model_args.model_ckpt}")
        model, missing_keys, unexpected_keys = load_model_weight(model, model_args.model_ckpt)
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")
        model.quantizer.root_vq_layer.initted = True
        model.quantizer.shared_vq_layer.initted = True
        model.vq_warmup_steps = -1


    logger.info(f"Model size: {model.num_parameters()}")
    logger.info(f"Model trainable parameters: {model.num_parameters(only_trainable=True)}")


    collator = Collator(data_args, tokenizer, processor, mode="train")


    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
        model.gradient_checkpointing_enable()
        # model.enable_input_require_grads()

    trainer = CustomizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload




if __name__ == "__main__":
    main()
