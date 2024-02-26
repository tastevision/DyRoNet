#!/bin/bash
today=`date +%Y%m%d`
source ./evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_LSN_l
ckpt1=./models/checkpoints/longshortnet_m.pth
ckpt2=./models/checkpoints/longshortnet_l.pth
cur_exp_name=${today}_LSN_m_LSN_l

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt1 \
             $ckpt2 \
             ./data/output/$cur_exp_name/latest_ckpt.pth;

clear_gpu;
