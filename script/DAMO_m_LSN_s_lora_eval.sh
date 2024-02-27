#!/bin/bash
today=`date +%Y%m%d`
source ./evaluate_lora.sh

cur_cfg=./cfgs/streamdynamic_DAMO_m_LSN_s
ckpt1=./models/checkpoints/streamnet_m.pth
ckpt2=./models/checkpoints/longshortnet_s.pth
cur_exp_name=${today}_DAMO_m_LSN_s_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             ./data/output/$cur_exp_name/latest_ckpt.pth;

clear_gpu;
