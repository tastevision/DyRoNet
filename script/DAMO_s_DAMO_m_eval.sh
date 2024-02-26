#!/bin/bash
today=`date +%Y%m%d`
source ./evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_DAMO_s
ckpt1=./models/checkpoints/streamnet_s.pth
ckpt2=./models/checkpoints/streamnet_m.pth
cur_exp_name=${today}_DAMO_s_DAMO_m

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             ./data/output/$cur_exp_name/latest_ckpt.pth;

clear_gpu;
