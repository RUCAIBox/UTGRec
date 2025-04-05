import copy
import os
from logging import getLogger
from typing import Any, Tuple, List, Mapping, Dict, Union, Optional

import numpy as np
from tqdm import tqdm
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask
from transformers.data.data_collator import tolist, _torch_collate_batch

import torchvision.transforms as transforms

from vae import center_crop_arr
from qwen_vl_utils import process_vision_info

MASK_TOKEN="<mask>"



class ItemMMDataset(Dataset):

    def __init__(self, args, dataset, mode='all'):
        super(ItemMMDataset, self).__init__()

        self.args = args
        self.dataset = dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.mode = mode

        seq_file = os.path.join(self.data_path, 'all_item_seqs.json')
        id_mapping_file = os.path.join(self.data_path, 'id_mapping.json')
        meatdata_file = os.path.join(self.data_path, 'metadata.sentence.json')

        with open(seq_file, 'r') as f:
            self.all_item_seqs = json.load(f)
        with open(id_mapping_file, 'r') as f:
            self.id_mapping = json.load(f)
        with open(meatdata_file, 'r') as f:
            self.metadata = json.load(f)


        self.user2id = self.id_mapping['user2id']
        self.item2id = self.id_mapping['item2id']
        self.id2item = self.id_mapping['id2item']
        self.id2user = self.id_mapping['id2user']

        self.all_items = self.get_item_data()
        self.co_items = self.get_co_occurring_items()


    def get_item_data(self):

        if self.mode == 'all':
            all_items = self.id2item[1:]
            return all_items

        train_item_set = set()
        for user in self.all_item_seqs:
            items = self.all_item_seqs[user]
            items = items[:-2]
            train_item_set.update(items)

        if self.mode == 'train':
            use_items = train_item_set
        elif self.mode == 'valid':
            use_items = set(self.metadata.keys()) - train_item_set

        return sorted(list(use_items))

    def get_co_occurring_items(self):
        co_items = defaultdict(set)


        for user in self.all_item_seqs:
            items = self.all_item_seqs[user]
            items = items[:-2]
            for i, item in enumerate(items):
                if i==0:
                    co_items[item].add(items[i+1])
                elif i==len(items)-1:
                    co_items[item].add(items[i-1])
                else:
                    co_items[item].add(items[i-1])
                    co_items[item].add(items[i+1])


        for item in self.all_items:
            if item not in co_items:
                co_items[item].add(item)

            if len(co_items[item]) == 0:
                co_items[item].add(item)

        return co_items


    def sample_co_occurring_items(self, item):
        co_items = self.co_items[item]
        co_items = sorted(list(co_items))
        co_item = np.random.choice(co_items)
        return co_item

    def get_item_info(self, item):

        text = self.metadata[item]

        item_image_file = os.path.join(self.data_path, 'Images', f"{item}.jpg")

        if not os.path.exists(item_image_file):
            print(f"No image {item_image_file}: {item}")
            pil_image = None
            item_image_file = None
        else:
            try:
                pil_image = Image.open(item_image_file).convert("RGB")
            except Exception as e:
                print(f"Error loading image {item_image_file}: {e}")
                pil_image = None
                item_image_file = None

        return text, pil_image, item_image_file

    def __getitem__(self, idx):

        item = self.all_items[idx]

        pos_item = self.sample_co_occurring_items(item)

        item_info = self.get_item_info(item)
        pos_item_info = self.get_item_info(pos_item)

        return item_info, pos_item_info

    def __len__(self):

        return len(self.all_items)


def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
    if isinstance(examples[0], Mapping):
        input_ids = [e["input_ids"] for e in examples]
    else:
        input_ids = examples
        examples = [{"input_ids": e} for e in examples]

    batch_input = _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
    # print(batch_input)

    mask_labels = []
    for e in examples:
        ref_tokens = []
        for id in tolist(e["input_ids"]):
            # token = self.tokenizer._convert_id_to_token(id)
            token = self.tokenizer.convert_ids_to_tokens(id)
            ref_tokens.append(token)

        # For Chinese tokens, we need extra inf to mark sub-word, e.g [喜,欢]-> [喜，##欢]
        if "chinese_ref" in e:
            ref_pos = tolist(e["chinese_ref"])
            len_seq = len(e["input_ids"])
            for i in range(len_seq):
                if i in ref_pos:
                    ref_tokens[i] = "##" + ref_tokens[i]
        mask_labels.append(self._whole_word_mask(ref_tokens))
    # print(mask_labels)
    batch_mask = _torch_collate_batch(mask_labels, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
    inputs, labels = self.torch_mask_tokens(batch_input, batch_mask)
    return {"input_ids": inputs, "labels": labels}

DataCollatorForWholeWordMask.torch_call = torch_call



class Collator:


    def __init__(self, args, tokenizer, processor, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.processor = processor
        self.mode = mode

        self.max_feature_length = args.max_feature_length
        # self.enc_mlm_probability = args.enc_mlm_probability if mode == "train" else 0
        self.dec_mlm_probability = args.dec_mlm_probability
        self.code_num = args.code_num
        self.code_tokens = [f'<CODE_{i}>' for i in range(self.code_num)]
        self.code_tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in self.code_tokens]
        self.mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)

        self.special_token_ids = ([self.tokenizer.convert_tokens_to_ids(_) for _ in self.tokenizer.additional_special_tokens] +
                                  [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, self.mask_token_id] + self.code_tokens_ids)
        # print(self.special_token_ids)
        self.random_tokens_cands = list(range(len(self.tokenizer)))
        self.random_tokens_cands = [token_id for token_id in self.random_tokens_cands if token_id not in self.special_token_ids]
        self.random_tokens_cands = torch.LongTensor(self.random_tokens_cands)


        self.dec_mask_collator = DataCollatorForWholeWordMask(
            tokenizer,
            mlm_probability=self.dec_mlm_probability,
            # mlm=False,
        )

        self.dec_mask_collator.torch_mask_tokens = self.torch_mask_tokens_bert

        if mode == "train":
            self.vae_image_transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.vae_image_transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        block = {}

        self.prompt = f"This is an image of an item. The item's information also includes: {block}. Compress the image and information into {self.code_num} tokens, representing content from coarse to fine granularity."

        self.nopic_prompt = f"An item is missing its corresponding image. The item's information also includes: {block}. Compress the information into {self.code_num} tokens, representing content from coarse to fine granularity."


    def get_pad_attention_mask(self, input_ids):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return attention_mask

    def get_code_token_indices(self, input_ids):
        indices = []
        for code_token_id in self.code_tokens_ids:
            code_token_index = [seq.tolist().index(code_token_id) for seq in input_ids]
            indices.append(torch.LongTensor(code_token_index))

        indices = torch.stack(indices, dim=1)


        return indices


    def torch_mask_tokens_bert(self, inputs: Any, mask_labels: Any) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """
        import torch

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the"
                " --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)


        for spid in self.special_token_ids:
            if spid in inputs:
                spid_mask = labels.eq(spid)
                probability_matrix.masked_fill_(spid_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        # indices_replaced = masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words_idx = torch.randint(len(self.random_tokens_cands), labels.shape, dtype=torch.long)
        random_words = self.random_tokens_cands[random_words_idx]

        # random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        return inputs, labels

    def cut_text(self, text):

        tokens = self.tokenizer.encode(text,
                                     add_special_tokens=False,
                                     max_length=self.max_feature_length,
                                     truncation=True)
        cutoff_text = self.tokenizer.decode(tokens, skip_special_tokens=True)

        return cutoff_text



    def get_batch_data(self, item_info):
        text, pil_image, item_image_file = item_info
        # print(text)
        cutoff_text = self.cut_text(text)
        # print(cutoff_text)

        if item_image_file is not None:
            input_text = self.prompt.format(cutoff_text)
            input_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item_image_file},
                        {"type": "text", "text": input_text},
                    ],
                }
            ]
        else:
            input_text = self.nopic_prompt.format(cutoff_text)
            input_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_text},
                    ],
                }
            ]

        input_text = self.processor.apply_chat_template(
            input_message, tokenize=False, add_generation_prompt=True
        )

        input_text = input_text + "".join(self.code_tokens)

        decoder_input_tokens = self.tokenizer.encode(text, max_length=self.max_feature_length, truncation=True)


        if pil_image is None:
            decoder_input_image = torch.zeros(3, 256, 256)
            decoder_input_image_mask = torch.zeros(256)
        else:
            # print(np.array(pil_image)[0])
            decoder_input_image = self.vae_image_transform(pil_image)
            decoder_input_image_mask = torch.zeros(256)
            mask_num = int(np.ceil(256 * self.dec_mlm_probability))
            mask_indices = torch.randperm(256)[:mask_num]
            decoder_input_image_mask[mask_indices] = 1


        return input_text, input_message, decoder_input_tokens, decoder_input_image, decoder_input_image_mask

    def __call__(self, batch):

        #text, image, image_size, pil_image

        input_texts = []
        input_messages = []
        decoder_input_ids = []
        decoder_input_images = []
        decoder_input_image_masks = []

        pos_input_texts = []
        pos_input_messages = []
        pos_decoder_input_ids = []
        pos_decoder_input_images = []
        pos_decoder_input_image_masks = []

        for item_info, pos_item_info in batch:

            (input_text, input_message,
             decoder_input_tokens, decoder_input_image, decoder_input_image_mask) = self.get_batch_data(item_info)
            input_texts.append(input_text)
            input_messages.append(input_message)
            decoder_input_ids.append({"input_ids": decoder_input_tokens})
            decoder_input_images.append(decoder_input_image)
            decoder_input_image_masks.append(decoder_input_image_mask)

            (pos_input_text, pos_input_message,
             pos_decoder_input_tokens, pos_decoder_input_image, pos_decoder_input_image_mask) = self.get_batch_data(pos_item_info)
            pos_input_texts.append(pos_input_text)
            pos_input_messages.append(pos_input_message)
            pos_decoder_input_ids.append({"input_ids": pos_decoder_input_tokens})
            pos_decoder_input_images.append(pos_decoder_input_image)
            pos_decoder_input_image_masks.append(pos_decoder_input_image_mask)

        image_inputs, video_inputs = process_vision_info(input_messages)
        inputs = self.processor(
            text=input_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        pos_image_inputs, pos_video_inputs = process_vision_info(pos_input_messages)
        pos_inputs = self.processor(
            text=pos_input_texts,
            images=pos_image_inputs,
            videos=pos_video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs["code_token_indices"] = self.get_code_token_indices(inputs["input_ids"])
        pos_inputs["code_token_indices"] = self.get_code_token_indices(pos_inputs["input_ids"])

        decoder_inputs = self.dec_mask_collator(decoder_input_ids)
        decoder_inputs["attention_mask"] = self.get_pad_attention_mask(decoder_inputs["input_ids"])
        decoder_inputs["input_images"] = torch.stack(decoder_input_images, dim=0)
        decoder_inputs["input_image_masks"] = torch.stack(decoder_input_image_masks, dim=0).long()


        pos_decoder_inputs = self.dec_mask_collator(pos_decoder_input_ids)
        pos_decoder_inputs["attention_mask"] = self.get_pad_attention_mask(pos_decoder_inputs["input_ids"])
        pos_decoder_inputs["input_images"] = torch.stack(pos_decoder_input_images, dim=0)
        pos_decoder_inputs["input_image_masks"] = torch.stack(pos_decoder_input_image_masks, dim=0).long()


        return {
            "inputs": inputs,
            "decoder_inputs": decoder_inputs,
            "pos_inputs": pos_inputs,
            "pos_decoder_inputs": pos_decoder_inputs,
        }






