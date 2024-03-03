#!/bin/bash
today=`date +%Y%m%d`

function clear_gpu() {
ps -ef | grep $USER | grep stream | grep python | cut -d ' ' -f 2 | xargs -n 1 kill -9;
ps -ef | grep $USER | grep stream | grep python | cut -d ' ' -f 3 | xargs -n 1 kill -9;
ps -ef | grep $USER | grep stream | grep python | cut -d ' ' -f 4 | xargs -n 1 kill -9;
}

function eval_process() {
EXP_DIR=$1
CFGS=$2
TRAINED_CKPT=$3

python ./tools/eval.py -f $CFGS \
    -c $TRAINED_CKPT \
    --experiment-name $EXP_DIR \
    --router-mode max \
    --logfile router_eval_${today}.txt \
    -d 1 -b 1 --conf 0.01 --fp16
}
