#!/bin/bash
today=`date +%Y%m%d`
source ./evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_DAMO_l
ckpt1=./models/checkpoints/streamnet_m.pth
ckpt2=./models/checkpoints/streamnet_l.pth
cur_exp_name=${today}_DAMO_m_DAMO_l

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             ./data/output/$cur_exp_name/latest_ckpt.pth;

clear_gpu;
