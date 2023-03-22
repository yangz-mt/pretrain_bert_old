#!/usr/bin/env sh

root_path=$PWD
PY_FILE_PATH="$root_path/run_pretraining.py"

tensorboard_path="$root_path/tensorboard"
log_path="$root_path/exp_log"
ckpt_path="$root_path/ckpt"

mkdir -p $tensorboard_path
mkdir -p $log_path
mkdir -p $ckpt_path

export PYTHONPATH=$PWD

num_gpus="1"
export CUDA_VISIBLE_DEVICES="0"


# gdb -args 
python run_pretrain_ddp.py \
                --local_rank 0 \
                --lr 6e-4 \
                --train_micro_batch_size_per_gpu 5 \
                --eval_micro_batch_size_per_gpu 20 \
                --gradient_accumulation_steps 1 \
                --mlm_model_type bert \
                --epoch 15 \
                --max_grad_norm 1.0 \
                --data_path_prefix ../dataset/wudao_h5_new_eval  \
                --eval_data_path_prefix ../dataset/wudao_h5_new_eval \
                --tokenizer_path ../dataset/chinese-roberta-wwm-ext-large \
                --bert_config ./config.json \
                --tensorboard_path $tensorboard_path \
                --log_path $log_path \
                --ckpt_path $ckpt_path \
                --log_interval 100 \
                --wandb \
                --dtr
