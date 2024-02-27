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
CKPT1=$3
CKPT2=$4
LORA_CKPT=$5

python ./tools/eval.py -f $CFGS \
    --ckpt1 $CKPT1 \
    --ckpt2 $CKPT2 \
    --experiment-name $EXP_DIR \
    --router-mode max \
    --lora \
    --lora-rank 32 \
    --lora-ckpt $LORA_CKPT \
    --logfile after_router_train_random_online_${today}.txt \
    -d 1 -b 1 --conf 0.01 --fp16 &

echo "done"
}
