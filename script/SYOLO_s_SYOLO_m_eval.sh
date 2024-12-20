#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_SYOLO_s
ckpt=./models/checkpoints/SYOLO_s_SYOLO_m.pth
cur_exp_name=${today}_SYOLO_s_SYOLO_m_lora

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt;
