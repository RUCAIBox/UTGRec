import gc
import json
import os
from functools import partial

import torch
from torch.utils.data import ConcatDataset
from transformers import WEIGHTS_NAME
from transformers.utils import WEIGHTS_INDEX_NAME

from data import ItemMMDataset



def load_model_weight(model, ckpt_path):


    weights_index_file = os.path.join(ckpt_path, WEIGHTS_INDEX_NAME)

    if not os.path.exists(weights_index_file):
        weights_file = os.path.join(ckpt_path, WEIGHTS_NAME)
        ckpt = torch.load(weights_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(ckpt)
    else:

        with open(weights_index_file, "r", encoding="utf-8") as f:
            index = json.load(f)

        shard_files = list(set(index["weight_map"].values()))

        # If strict=True, error before loading any of the state dicts.
        loaded_keys = index["weight_map"].keys()
        model_keys = model.state_dict().keys()
        missing_keys = [key for key in model_keys if key not in loaded_keys]
        unexpected_keys = [key for key in loaded_keys if key not in model_keys]


        weights_only_kwarg = {"weights_only": True}
        loader = partial(torch.load, map_location="cpu", **weights_only_kwarg)

        for shard_file in shard_files:
            state_dict = loader(os.path.join(ckpt_path, shard_file))
            model.load_state_dict(state_dict, strict=False)

            # Make sure memory is freed before we load the next state dict.
            del state_dict
            gc.collect()

        # Return the same thing as PyTorch load_state_dict function.
        missing_keys, unexpected_keys = torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)


    return model, missing_keys, unexpected_keys





def ensure_dir(dir_path):

    os.makedirs(dir_path, exist_ok=True)




def init_new_tokens(model, tokenizer, code_num):


    input_embs = model.get_input_embeddings()
    input_embs.weight.data[-code_num:] = input_embs.weight.data[tokenizer.cls_token_id]

    output_embs = model.get_output_embeddings()
    output_embs.weight.data[-code_num:] = output_embs.weight.data[tokenizer.cls_token_id]

    return model



def get_datasets(data_args, mode="train"):

    if mode == "train":
        dataset_names = data_args.datasets.strip().split(",")
        dataset_list = []
        for dataset_name in dataset_names:
            dataset = ItemMMDataset(args=data_args, dataset=dataset_name.strip(), mode="train")
            dataset_list.append(dataset)
    elif mode == "valid":
        if data_args.valid_datasets is None:
            return None

        dataset_names = data_args.valid_datasets.strip().split(",")
        dataset_list = []
        for dataset_name in dataset_names:
            dataset = ItemMMDataset(args=data_args, dataset=dataset_name.strip(), mode="all")
            dataset_list.append(dataset)

    all_dataset = ConcatDataset(dataset_list)

    return all_dataset