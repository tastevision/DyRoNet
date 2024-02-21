#!/bin/bash
today=`date +%Y%m%d`

function clear_gpu() {
ps -ef | grep xiang.huang | grep stream | grep python | cut -d ' ' -f 2 | xargs -n 1 kill -9;
ps -ef | grep xiang.huang | grep stream | grep python | cut -d ' ' -f 3 | xargs -n 1 kill -9;
ps -ef | grep xiang.huang | grep stream | grep python | cut -d ' ' -f 4 | xargs -n 1 kill -9;
}

# evaluate过程
function eval_process() {
EXP_DIR=$1
CFGS=$2
CKPT1=$3
CKPT2=$4
TRAINED_CKPT=$5
# if [[ `cat ./data/output/$EXP_DIR/train_log.txt | grep "Training of experiment is done" | wc -l` == 0 ]]; then
#     echo "检测到实验未完成，停止测试" >> ./data/output/$EXP_DIR/alert.txt;
#     return
# fi

# before router train
CUDA_VISIBLE_DEVICES=0 python ./tools/eval.py -f $CFGS \
                     --ckpt1 $CKPT1 \
                     --ckpt2 $CKPT2 \
                     --experiment-name $EXP_DIR \
                     --router-mode random \
                     --logfile before_router_train_random_online_${today}.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

# CUDA_VISIBLE_DEVICES=1 python ./tools/eval.py -f $CFGS \
#                      --ckpt1 $CKPT1 \
#                      --ckpt2 $CKPT2 \
#                      --experiment-name $EXP_DIR \
#                      --router-mode max \
#                      --logfile before_router_train_router_online_${today}.txt \
#                      -d 1 -b 1 --conf 0.01 --fp16 &

CUDA_VISIBLE_DEVICES=1 python ./tools/eval.py -f $CFGS \
                     --ckpt1 $CKPT1 \
                     --ckpt2 $CKPT2 \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --branch 0 \
                     --logfile before_router_train_branch_0_online_${today}.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

CUDA_VISIBLE_DEVICES=2 python ./tools/eval.py -f $CFGS \
                     --ckpt1 $CKPT1 \
                     --ckpt2 $CKPT2 \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --branch 1 \
                     --logfile before_router_train_branch_1_online_${today}.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

# wait

# # after router train
# CUDA_VISIBLE_DEVICES=0 python ./tools/eval.py -f $CFGS \
#                      --c $TRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode random \
#                      --logfile after_router_train_random_online_${today}.txt \
#                      -d 1 -b 1 --conf 0.01 --fp16 &

CUDA_VISIBLE_DEVICES=3 python ./tools/eval.py -f $CFGS \
                     -c $TRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --logfile after_router_train_router_online_${today}.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

# CUDA_VISIBLE_DEVICES=2 python ./tools/eval.py -f $CFGS \
#                      --c $TRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode max \
#                      --branch 0 \
#                      --logfile after_router_train_branch_0_online_${today}.txt \
#                      -d 1 -b 1 --conf 0.01 --fp16 &
#
# CUDA_VISIBLE_DEVICES=3 python ./tools/eval.py -f $CFGS \
#                      --c $TRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode max \
#                      --branch 1 \
#                      --logfile after_router_train_branch_1_online_${today}.txt \
#                      -d 1 -b 1 --conf 0.01 --fp16 &

wait

echo "done"
}
