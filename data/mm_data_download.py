import argparse
import collections
import concurrent
import gzip
import html
import json
import os
import random
import re
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests
import shutil
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def check_path(path):
    os.makedirs(path, exist_ok=True)

def download_image(info):
    url, file_path = info

    if os.path.exists(file_path):
        try:
            pil_image = Image.open(file_path).convert("RGB")
            return ""
        except Exception as e:
            pass

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return f"{file_path} downloaded."
    except Exception as e:
        return f"Error downloading {file_path}: {e}"





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Musical_Instruments')
    parser.add_argument('--img_size', type=str, default='large')
    parser.add_argument('--meta_path', type=str, default='./AmazonReviews2023/Metadata/')
    parser.add_argument('--data_path', type=str, default='./AmazonReviews2023/')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    check_path(os.path.join(args.data_path, args.dataset))

    print(' Dataset: ', args.dataset)

    img_size = args.img_size

    print(' Image size: ', img_size)

    # load item IDs with meta data
    meta_file_path = os.path.join(args.meta_path, f'meta_{args.dataset}.jsonl.gz')
    id_mapping_file = os.path.join(args.data_path, args.dataset, 'id_mapping.json')
    with open(id_mapping_file, 'r') as f:
        id_mapping = json.load(f)
    all_items = id_mapping['id2item'][1:]

    item_img_url = {}
    tot_imgs = 0
    with gzip.open(meta_file_path, "r") as fp:
        for line in tqdm(fp, desc="Load metas"):
            # print(line)
            data = json.loads(line)
            item = data["parent_asin"]

            if item not in all_items:
                continue

            images = data["images"]
            if len(images) > 0:
                main_images_index=0
                for k, img in enumerate(images):
                    if img["variant"].lower() == "main":
                        main_images_index = k
                        break
                main_images = images[main_images_index]
                tot_imgs += 1
            else:
                print(f'{item} has no image')
                main_images = {}

            item_img_url[item] = main_images

    print(f'Number of items: {len(item_img_url)}')
    print(f'Number of items with images: {tot_imgs}')


    path = os.path.join(args.data_path, args.dataset,  "Images")
    check_path(path)
    download_info = []
    for item in tqdm(item_img_url):
        if item_img_url[item] != {}:
            if img_size not in item_img_url[item]:
                k = list(item_img_url[item].keys())[-1]
                img_url = item_img_url[item][k]
            else:
                img_url = item_img_url[item][img_size]
            img_path = os.path.join(path, f'{item}.jpg')

            download_info.append((img_url, img_path))
        else:
            pass

    # Use ThreadPoolExecutor to download images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(download_image, info): info[1] for info in download_info}

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            img_path = futures[future]
            try:
                res = future.result()
                if res != "":
                    print(res)
            except Exception as e:
                print(f"Error downloading {img_path}: {e}")
