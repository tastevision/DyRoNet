#!/bin/bash
today=`date +%Y%m%d`
source ./evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_DAMO_s_LSN_l
ckpt1=./models/checkpoints/streamnet_s.pth
ckpt2=./models/checkpoints/longshortnet_l.pth
lora_ckpt=./models/checkpoints/DAMO_s_LSN_l_lora.pth
cur_exp_name=${today}_DAMO_s_LSN_l_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             $lora_ckpt;

