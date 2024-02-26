#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_LSN_s_m_l
ckpt1=./models/checkpoints/longshortnet_s.pth
ckpt2=./models/checkpoints/longshortnet_m.pth
ckpt3=./models/checkpoints/longshortnet_l.pth
cur_exp_name=${today}_LSN_s_LSN_m_LSN_l_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             $ckpt3 \
             ./data/output/$cur_exp_name/latest_ckpt.pth;

clear_gpu;
