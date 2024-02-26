#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_DAMO_m
ckpt1=./models/checkpoints/streamnet_s.pth
ckpt2=./models/checkpoints/streamnet_l.pth
cur_exp_name=${today}_DAMO_s_DAMO_l_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             ./data/output/$cur_exp_name/latest_ckpt.pth;

clear_gpu;
