#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_SYOLO_l
ckpt=./models/checkpoints/SYOLO_m_SYOLO_l.pth
cur_exp_name=${today}_SYOLO_m_SYOLO_l

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt;
