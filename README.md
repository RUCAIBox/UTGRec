# UTGRec

This is the official PyTorch implementation for the paper:

> Universal Item Tokenization for Transferable Generative Recommendation

## Overview

In this paper, we propose **UTGRec**, a <u>U</u>niversal item <u>T</u>okenization approach for transferable <u>G</u>enerative  <u>Rec</u>ommendation. Specifically, we design a universal item tokenizer for encoding rich item semantics by adapting a multimodal large language model (MLLM).  By devising tree-structured codebooks, we discretize content representations into corresponding codes for item tokenization. To effectively learn the universal item tokenizer on multiple domains, we introduce two key techniques in our approach. For raw content reconstruction, we employ dual lightweight decoders to reconstruct item text and images from discrete representations to capture general knowledge embedded in the content. For collaborative knowledge integration, we assume that co-occurring items are similar and integrate collaborative signals through co-occurrence alignment and reconstruction. Finally,  we present a joint learning framework to pre-train and adapt the transferable generative recommender across multiple domains. Extensive experiments on four public datasets demonstrate the superiority of UTGRec compared to both traditional and generative recommendation baselines.

![model](./asset/model.png)

## Requirements

```
torch==2.4.1+cu124
transformers==4.45.2
deepspeed==0.15.4
accelerate==1.0.1
flash-attn==2.6.3
```

VAE checkpoint for DiffLoss:  [download.py](https://github.com/LTH14/mar/blob/main/util/download.py)

## Data

You can find all the datasets we used in [Google Drive](https://drive.google.com/file/d/16dkTf-Pe0fpF3IyuBtRg8p3EF1k5E0kz/view?usp=sharing). Please download the file to the `data/` folder and unzip it. Then, run the following command to download item images:

```shell
cd data

all_data=(Musical_Instruments Industrial_and_Scientific Video_Games Office_Products Arts_Crafts_and_Sewing Baby_Products CDs_and_Vinyl Cell_Phones_and_Accessories Software)

for data in "${all_data[@]}"; do
	python mm_data_download.py --dataset ${data}
done
```

Please note that some request errors may occur during the download process. To ensure that all images are successfully downloaded, you may need to run the above command multiple times. Files that have already been downloaded will not be downloaded again.

## Script

Pre-training: `pretrain.sh`

Finetuning: `finetune.sh`

