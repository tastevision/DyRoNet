#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_DAMO_l
ckpt1=./models/checkpoints/streamnet_m.pth
ckpt2=./models/checkpoints/streamnet_l.pth
cur_exp_name=${today}_DAMO_m_DAMO_l_lora

python tools/train_dil.py -f $cur_cfg \
                          --ckpt1 $ckpt1 \
                          --ckpt2 $ckpt2 \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --router-mode max \
                          --lora --lora-rank 32 \
                          --experiment-name $cur_exp_name \
                          --logfile "train_log.txt" \
                          --eval-batch-size 4 \
                          -d 4 -b 4 --fp16

clear_gpu;
