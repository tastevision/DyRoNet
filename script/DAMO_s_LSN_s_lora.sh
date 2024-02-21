#!/bin/bash
today=`date +%Y%m%d`
source ./evaluate_lora.sh

cur_cfg=../cfgs/streamdynamic_DAMO_s_LSN_s
cur_ckpt=../models/checkpoints/streamdynamic_untrained_DAMO_s_LSN_s.pth
cur_exp_name=${today}_${0%%.*}

# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --router-mode max --lora --lora-rank 32 --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
clear_gpu;

