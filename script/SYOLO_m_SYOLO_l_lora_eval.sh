#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_SYOLO_l
ckpt1=./models/checkpoints/m_s50_one_x.pth
ckpt2=./models/checkpoints/l_s50_one_x.pth
lora_ckpt=./models/checkpoints/SYOLO_m_SYOLO_l_lora.pth
cur_exp_name=${today}_SYOLO_m_SYOLO_l_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             $lora_ckpt;
