#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_LSN_s
ckpt1=./models/checkpoints/longshortnet_s.pth
ckpt2=./models/checkpoints/longshortnet_m.pth
cur_exp_name=${today}_LSN_s_LSN_m_lora

python tools/train_dil.py -f $cur_cfg \
                          --ckpt1 $ckpt1 \
                          --ckpt2 $ckpt2 \
                          --router-mode max \
                          --lora --lora-rank 32 \
                          --experiment-name $cur_exp_name \
                          --logfile "train_log.txt" \
                          --eval-batch-size 4 \
                          -d 4 -b 4 --fp16

clear_gpu;
