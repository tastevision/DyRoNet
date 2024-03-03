#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_DAMO_m
ckpt=./models/checkpoints/DAMO_s_DAMO_l.pth
cur_exp_name=${today}_DAMO_s_DAMO_l

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt;
