export WANDB_MODE=disabled
export DISABLE_MLFLOW_INTEGRATION=TRUE
export TORCH_DISTRIBUTED_DEBUG=DETAIL

#export NCCL_DEBUG=INFO
export NCCL_NET=IB
export GLOO_SOCKET_IFNAME=bond1
export NCCL_IB_GID_INDEX=3



NODE_RANK=0
#NODE_RANK=0
NNODES=2
MASTER_ADDR=xxx.xxx.xxx.xxx
NUM_GPUS=8
MASTER_PORT=11337


FT_DATASETS=Musical_Instruments #Industrial_and_Scientific,Video_Games,Office_Products

CKPT=XXXXXXX
MODEL_CKPT=./ckpt/${CKPT}


CODE_SIZE=256
CODE_NUM=3
CODE_DIM=128
DEV_BATCH_SIZE=16
STRATEGY=steps
SAVE_EVAL_STEPS=0.5

EPOCHS=40
LR=1e-4
VQ_W=200
DIFF_W=5
CL_W=0.01
POS_W=0.03

DATE=`date +%Y%m%d-%H%M%S`
OUTPUT_DIR=./ckpt/${FT_DATASETS}-${DATE}

torchrun --node_rank=$NODE_RANK --nnodes=$NNODES --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT main.py \
    --output_dir $OUTPUT_DIR \
    --datasets $FT_DATASETS \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --per_device_eval_batch_size $DEV_BATCH_SIZE \
    --learning_rate $LR \
    --llm_learning_rate $LR \
    --num_train_epochs $EPOCHS \
    --save_strategy $STRATEGY \
    --eval_strategy $STRATEGY \
    --save_steps $SAVE_EVAL_STEPS \
    --eval_steps $SAVE_EVAL_STEPS \
    --vq_loss_weight $VQ_W \
    --diffloss_weight $DIFF_W \
    --cl_loss_weight $CL_W \
    --pos_recon_loss_weight $POS_W \
    --code_num $CODE_NUM \
    --codebook_size $CODE_SIZE \
    --codebook_dim $CODE_DIM \
    --deepspeed zero_cfg/zero2.json \
    --model_ckpt $MODEL_CKPT \
    --fix_codebook True



if [ "$NODE_RANK" -eq 0 ]; then
    # 将 BX 的值写入缓存文件
    echo "$OUTPUT_DIR" > output_dir_cache.txt
    echo "Writting OUTPUT_DIR"
else
    echo "NODE_RANK!=0"
fi
sleep 20
if [ -f "output_dir_cache.txt" ]; then
    OUTPUT_DIR=$(<"output_dir_cache.txt")
    echo "Cache OUTPUT_DIR: $OUTPUT_DIR"
else
    echo "output_dir_cache.txt not find"
fi


CUDA_VISIBLE_DEVICES=0 \
nohup python -u generate_tokens_topk.py \
  --datasets $FT_DATASETS \
  --code_num $CODE_NUM \
  --codebook_size $CODE_SIZE \
  --codebook_dim $CODE_DIM \
  --model_ckpt ${OUTPUT_DIR} \
  --tokens_file utgrec.sem_ids \
> gclogs/utgrec_finetune.log 2>&1 &

