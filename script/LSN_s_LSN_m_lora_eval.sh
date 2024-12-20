#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_LSN_s
ckpt1=./models/checkpoints/longshortnet_s.pth
ckpt2=./models/checkpoints/longshortnet_m.pth
lora_ckpt=./models/checkpoints/LSN_s_LSN_m_lora.pth
cur_exp_name=${today}_LSN_s_LSN_m_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             $lora_ckpt;
