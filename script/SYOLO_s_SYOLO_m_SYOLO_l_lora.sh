#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_SYOLO_s_m_l
ckpt1=./models/checkpoints/s_s50_one_x.pth
ckpt2=./models/checkpoints/m_s50_one_x.pth
ckpt3=./models/checkpoints/l_s50_one_x.pth
cur_exp_name=${today}_SYOLO_s_SYOLO_m_SYOLO_l_lora

python tools/train_dil.py -f $cur_cfg \
                          --ckpt1 $ckpt1 \
                          --ckpt2 $ckpt2 \
                          --ckpt3 $ckpt3 \
                          --router-mode max \
                          --lora --lora-rank 32 \
                          --experiment-name $cur_exp_name \
                          --logfile "train_log.txt" \
                          --eval-batch-size 4 \
                          -d 4 -b 4 --fp16

clear_gpu;
