#!/bin/bash
today=`date +%Y%m%d`
source ./evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_DAMO_s_LSN_l
ckpt1=./models/checkpoints/streamnet_s.pth
ckpt2=./models/checkpoints/longshortnet_l.pth
cur_exp_name=${today}_DAMO_s_LSN_l_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             ./data/output/$cur_exp_name/latest_ckpt.pth;

clear_gpu;

