#!/bin/bash
today=`date +%Y%m%d`
source ./evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_SYOLO_m
ckpt1=./models/checkpoints/s_s50_one_x.pth
ckpt2=./models/checkpoints/l_s50_one_x.pth
cur_exp_name=${today}_SYOLO_s_SYOLO_l_lora

python tools/train_dil.py -f $cur_cfg \
                          --ckpt1 $ckpt1 \
                          --ckpt2 $ckpt2 \
                          --router-mode max \
                          --experiment-name $cur_exp_name \
                          --logfile "train_log.txt" \
                          --eval-batch-size 4 \
                          -d 4 -b 4 --fp16

clear_gpu;
