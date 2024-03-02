#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_DAMO_l
ckpt1=./models/checkpoints/streamnet_m.pth
ckpt2=./models/checkpoints/streamnet_l.pth
lora_ckpt=./models/checkpoints/DAMO_m_DAMO_l_lora.pth
cur_exp_name=${today}_DAMO_m_DAMO_l_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             $lora_ckpt;
