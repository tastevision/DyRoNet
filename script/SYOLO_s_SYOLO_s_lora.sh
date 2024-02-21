#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_SYOLO_s_SYOLO_s
ckpt1=./models/checkpoints/s_s50_one_x.pth
ckpt2=./models/checkpoints/s_s50_one_x.pth
cur_exp_name=${today}_SYOLO_s_SYOLO_s_lora

python tools/train_dil.py -f $cur_cfg \
                          --ckpt1 $ckpt1 \
                          --ckpt2 $ckpt2 \
                          --router-mode max \
                          --lora --lora-rank 32 \
                          --experiment-name $cur_exp_name \
                          --logfile "train_log.txt" \
                          --eval-batch-size 4 \
                          -d 4 -b 4 --fp16

# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --router-mode max --lora --lora-rank 32 --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4;
eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             ./data/output/$cur_exp_name/latest_ckpt.pth;
clear_gpu;
