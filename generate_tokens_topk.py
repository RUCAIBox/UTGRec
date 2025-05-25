import json
import random
import warnings
from collections import defaultdict

import numpy as np

# warnings.filterwarnings("ignore")

import logging

import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration

from transformers.trainer_utils import set_seed
from arguments import ModelArguments, DataArguments, TrainingArguments
from model import MMTokenizerModel, fix_qwen2vl_forward
from utils import *
from data import Collator, MASK_TOKEN


logger = logging.getLogger(__name__)

Qwen2VLForConditionalGeneration.forward = fix_qwen2vl_forward


def offset_tokens(item2tokens, offset):
	new_item2tokens = {}
	for item, tokens in item2tokens.items():
		new_tokens = []
		for i in range(len(tokens)):
			new_tokens.append(int(tokens[i] + offset[i]))
		new_item2tokens[item] = tuple(new_tokens)
	return new_item2tokens


def check_conflict(all_code_ids):
	all_code_ids = torch.cat(all_code_ids, dim=0).long().cpu()
	all_codes = set()

	for a in all_code_ids:
		all_codes.add(str(a.tolist()))

	rate = ( all_code_ids.shape[0] - len(all_codes) ) / all_code_ids.shape[0]
	print(rate)



def main():
	parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	data_args.code_num = model_args.code_num

	set_seed(training_args.seed)
	data_args.enc_mlm_probability = 0.0

	print("Model args %s", model_args)
	print("Data args %s", data_args)

	device = torch.device("cuda")

	print(f"Load tokenizer from {model_args.model_ckpt}")
	min_pixels = 128 * 28 * 28
	max_pixels = 1024 * 28 * 28
	processor = AutoProcessor.from_pretrained(
		model_args.model_ckpt,
		min_pixels=min_pixels,
		max_pixels=max_pixels,
	)
	tokenizer = processor.tokenizer
	tokenizer.padding_side = "right"
	processor.tokenizer = tokenizer
	print(f"The tokenizer has {len(tokenizer)} tokens")

	logger.info(f"Load base model from {model_args.model_name_or_path}")
	llm = Qwen2VLForConditionalGeneration.from_pretrained(
		model_args.model_name_or_path,
		attn_implementation="flash_attention_2",
		torch_dtype=torch.bfloat16,
		device_map="auto"
	)

	llm.resize_token_embeddings(len(tokenizer))
	model = MMTokenizerModel(llm, model_args).to(torch.bfloat16)
	print(model)

	print(f"Loading pretrained model from {model_args.model_ckpt}")
	model, missing_keys, unexpected_keys = load_model_weight(model, model_args.model_ckpt)
	print(f"Missing keys: {missing_keys}")
	print(f"Unexpected keys: {unexpected_keys}")
	model.to(device)
	model.eval()

	code_num = model.code_num
	codebook_size = model.codebook_size

	if isinstance(codebook_size, list):
		assert len(codebook_size) == 2
		root_codebook_size = codebook_size[0]
		shared_codebook_size = codebook_size[1]
	else:
		root_codebook_size = codebook_size
		shared_codebook_size = codebook_size * (code_num - 1)

	offset = [0] + [root_codebook_size] * (code_num - 1)

	print(f"Offset: {offset}")

	collator = Collator(data_args, tokenizer, processor, mode="test")

	dataset_names = data_args.datasets.split(",")
	for dataset_name in dataset_names:
		print(f"Load dataset {dataset_name}")

		dataset = ItemMMDataset(data_args, dataset=dataset_name,mode="all")
		print(f"All item number: {len(dataset)}")
		dataloader = DataLoader(
			dataset,
			batch_size=16,
			num_workers=8,
			shuffle=False,
			collate_fn=collator,
		)

		all_code_ids = []
		all_code_states = []
		for batch in tqdm(dataloader):

			_, code_ids, code_token_states = model.encode(
					inputs=batch["inputs"].to(device),
				)
			# print(code_ids[0])
			all_code_ids.append(code_ids.long())
			all_code_states.append(code_token_states[:, -1])
			check_conflict(all_code_ids)

		all_code_ids = torch.cat(all_code_ids, dim=0).long().cpu().numpy()
		all_code_states = torch.cat(all_code_states, dim=0)

		print(f'Item tokens shape: {all_code_ids.shape}')
		print(f'Item code states shape: {all_code_states.shape}')


		tokens2item = defaultdict(list)
		max_conflict = 0
		for i in range(all_code_ids.shape[0]):
			tokens = tuple(all_code_ids[i].tolist())
			tokens2item[tokens].append(i)
			max_conflict = max(max_conflict, len(tokens2item[tokens]))
		print(f'Maximum conflict: {max_conflict}')
		print(f'Conflict rate: {(all_code_ids.shape[0] - len(tokens2item)) / all_code_ids.shape[0]}')


		item2tokens = {}
		used_tokens = set()
		collision_item_groups = []
		for tokens, iids in tokens2item.items():
			if len(iids) == 1:
				item = dataset.id2item[iids[0]+1]
				item2tokens[item] = tokens
				used_tokens.add(tokens)
			else:
				# Add a flag to indicate whether the conflicting tokens has been used
				collision_item_groups.append([False] + iids)



		# print(collision_item_groups)
		total_collision = 0
		for group in collision_item_groups:
			total_collision = total_collision + len(group) - 1
		print(total_collision)


		total_item = len(dataset)
		if isinstance(codebook_size, list):
			max_code = codebook_size[-1]
		else:
			max_code = codebook_size
		for k in range(2, max_code):
			if len(list(item2tokens.keys())) == total_item:
				break


			new_tokens2item = defaultdict(list)
			for collision_info in collision_item_groups:
				used = collision_info[0]
				collision_items = collision_info[1:]
				code_token_states = all_code_states[collision_items]

				code_ids, fix = model.get_topk_tail_token(code_token_states, topk=k, used=used)
				code_ids = code_ids.long().cpu().numpy()
				fix = fix.cpu().numpy()
				for iid, tail_code, f in zip(collision_items, code_ids, fix):
					tokens = tuple( all_code_ids[iid][:-1].tolist() + [tail_code] )

					if f and tokens not in used_tokens:
						item = dataset.id2item[iid + 1]
						item2tokens[item] = tokens
						used_tokens.add(tokens)
					else:
						new_tokens2item[tokens].append(iid)

			if k == max_code - 1:
				for tokens, iids in new_tokens2item.items():
					if len(iids) == 1 and tokens not in used_tokens:
						item = dataset.id2item[iids[0] + 1]
						item2tokens[item] = tokens
						used_tokens.add(tokens)
					else:
						for iid in iids:
							tail_code = random.randint(0, max_code - 1)
							tokens = tuple(all_code_ids[iid][:-1].tolist() + [tail_code])
							tmp = 1
							while tokens in used_tokens and tmp < 1000:
								tail_code = random.randint(0, max_code - 1)
								tokens = tuple(all_code_ids[iid][:-1].tolist() + [tail_code])
								tmp += 1
							if tokens in used_tokens:
								second_token = random.randint(0, max_code - 1)
								tokens = tuple(all_code_ids[iid][:-2].tolist() + [second_token, all_code_ids[iid][-1]])
								tmp = 1
								while tokens in used_tokens and tmp < 1000:
									second_token = random.randint(0, max_code - 1)
									tokens = tuple(all_code_ids[iid][:-2].tolist() + [second_token, all_code_ids[iid][-1]])
									tmp += 1

							item = dataset.id2item[iid + 1]
							item2tokens[item] = tokens
							used_tokens.add(tokens)
			else:
				collision_item_groups = []
				for tokens, iids in new_tokens2item.items():
					if len(iids) == 1 and tokens not in used_tokens:
						item = dataset.id2item[iids[0] + 1]
						item2tokens[item] = tokens
						used_tokens.add(tokens)
					else:
						# Add a marker to indicate whether the conflicting index has been used
						if tokens in used_tokens:
							collision_item_groups.append([True] + iids)
						else:
							collision_item_groups.append([False] + iids)

				# print(collision_item_groups)
				total_collision = 0
				for group in collision_item_groups:
					total_collision = total_collision + len(group) - 1
				print(total_collision)


		# assert len(used_tokens) == total_item
		print(f'Used tokens: {len(used_tokens)}')
		print(f'Total item: {total_item}')


		item2tokens = offset_tokens(item2tokens, offset)

		tokens_path = os.path.join(
			data_args.data_path, dataset_name,
			data_args.tokens_file
		)
		print(f'Saving item tokens to {tokens_path}...')
		with open(tokens_path, 'w') as f:
			json.dump(item2tokens, f)



if __name__ == "__main__":
	main()
