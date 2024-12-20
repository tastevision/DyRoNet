#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_LSN_l
ckpt1=./models/checkpoints/longshortnet_m.pth
ckpt2=./models/checkpoints/longshortnet_l.pth
lora_ckpt=./models/checkpoints/LSN_m_LSN_l_lora.pth
cur_exp_name=${today}_LSN_m_LSN_l_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             $lora_ckpt;
