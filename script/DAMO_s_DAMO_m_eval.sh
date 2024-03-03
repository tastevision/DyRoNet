#!/bin/bash
today=`date +%Y%m%d`
source ./script/evaluate_full.sh

cur_cfg=./cfgs/streamdynamic_DAMO_s
ckpt=./models/checkpoints/DAMO_s_DAMO_m.pth
cur_exp_name=${today}_DAMO_s_DAMO_m

eval_process $cur_exp_name \
             $cur_cfg \
             $ckpt;
