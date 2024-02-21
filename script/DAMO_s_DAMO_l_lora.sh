#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_DAMO_m
cur_ckpt=./models/checkpoints/streamdynamic_DAMO_branch2_untrained_m.pth
# cur_exp_name=${today}_${0%%.*}
cur_exp_name=dynamic_DAMO_m_lora_20240110

# python tools/train_dil.py -f $cur_cfg \
#                           --ckpt1 ./models/checkpoints/streamnet_s.pth \
#                           --ckpt2 ./models/checkpoints/streamnet_l.pth \
#                           --logfile "train_log.txt" \
#                           -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --lora --lora-rank 32 \
#                           --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;

# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --router-mode max --lora --lora-rank 32 --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4;
eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
clear_gpu;

# rm -v ./data/output/${cur_exp_name}/after_router_train_router_online.txt
# CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $cur_cfg \
#                      -c $cur_ckpt \
#                      --experiment-name $cur_exp_name \
#                      --router-mode max \
#                      --lora \
#                      --lora-rank 32 \
#                      --lora-ckpt ./data/output/$cur_exp_name/latest_ckpt.pth \
#                      --logfile after_router_train_router_online.txt \
#                      -d 1 -b 1 --conf 0.01 --fp16 &

