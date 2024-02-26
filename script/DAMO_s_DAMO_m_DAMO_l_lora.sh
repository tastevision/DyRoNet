#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_DAMO_s_m_l
ckpt1=./models/checkpoints/streamnet_s.pth
ckpt2=./models/checkpoints/streamnet_m.pth
ckpt3=./models/checkpoints/streamnet_l.pth
cur_exp_name=${today}_DAMO_s_DAMO_m_DAMO_l_lora

python tools/train_dil.py -f $cur_cfg \
                          --ckpt1 $ckpt1 \
                          --ckpt2 $ckpt2 \
                          --ckpt3 $ckpt3 \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --router-mode max \
                          --lora --lora-rank 32 \
                          --experiment-name $cur_exp_name \
                          --logfile "train_log.txt" \
                          --eval-batch-size 4 \
                          -d 4 -b 4 --fp16

clear_gpu;
