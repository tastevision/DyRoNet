#!/bin/bash
today=`date +%Y%m%d`
source ./evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_SYOLO_l
ckpt1=./models/checkpoints/m_s50_one_x.pth
ckpt2=./models/checkpoints/l_s50_one_x.pth
cur_exp_name=${today}_SYOLO_m_SYOLO_l

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             ./data/output/$cur_exp_name/latest_ckpt.pth;

clear_gpu;
