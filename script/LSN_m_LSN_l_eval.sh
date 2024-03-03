#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_LSN_l
ckpt=./models/checkpoints/LSN_m_LSN_l.pth
cur_exp_name=${today}_LSN_m_LSN_l

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt;
