#!/bin/bash
function eval_process() {
EXP_DIR=$1
CFGS=$2
UNTRAINED_CKPT=$3
TRAINED_CKPT=$4

# before router train
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode random \
                     --logfile before_router_train_random.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

CUDA_VISIBLE_DEVICES=1 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --logfile before_router_train_router.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

CUDA_VISIBLE_DEVICES=2 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --branch 0 \
                     --logfile before_router_train_branch_0.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

CUDA_VISIBLE_DEVICES=3 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --branch 1 \
                     --logfile before_router_train_branch_1.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

wait

# after router train
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $CFGS \
                     -c $TRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode random \
                     --logfile after_router_train_random.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

CUDA_VISIBLE_DEVICES=1 python tools/eval.py -f $CFGS \
                     -c $TRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --logfile after_router_train_router.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

CUDA_VISIBLE_DEVICES=2 python tools/eval.py -f $CFGS \
                     -c $TRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --branch 0 \
                     --logfile after_router_train_branch_0.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

CUDA_VISIBLE_DEVICES=3 python tools/eval.py -f $CFGS \
                     -c $TRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --branch 1 \
                     --logfile after_router_train_branch_1.txt \
                     -d 1 -b 1 --conf 0.01 --fp16 &

wait

echo "done"
}

# eval_process dynamic_DAMO_s_20240104 \
#              cfgs/streamdynamic_DAMO_s.py \
#              ./models/checkpoints/streamdynamic_DAMO_branch2_untrained_s.pth \
#              ./data/output/dynamic_DAMO_s_20240104/latest_ckpt.pth

eval_process dynamic_DAMO_m_20240104 \
             cfgs/streamdynamic_DAMO_m.py \
             ./models/checkpoints/streamdynamic_DAMO_branch2_untrained_m.pth \
             ./data/output/dynamic_DAMO_m_20240104/latest_ckpt.pth

eval_process dynamic_DAMO_l_20240103 \
             cfgs/streamdynamic_DAMO_l.py \
             ./models/checkpoints/streamdynamic_DAMO_branch2_untrained_l.pth \
             ./data/output/dynamic_DAMO_l_20240103/latest_ckpt.pth

eval_process dynamic_LSN_s_20240106 \
             cfgs/streamdynamic_LSN_s.py \
             ./models/checkpoints/streamdynamic_LSN_branch2_untrained_s.pth \
             ./data/output/dynamic_LSN_s_20240106/latest_ckpt.pth

eval_process dynamic_LSN_m_20240106 \
             cfgs/streamdynamic_LSN_m.py \
             ./models/checkpoints/streamdynamic_LSN_branch2_untrained_m.pth \
             ./data/output/dynamic_LSN_m_20240106/latest_ckpt.pth

eval_process dynamic_LSN_l_20240106 \
             cfgs/streamdynamic_LSN_l.py \
             ./models/checkpoints/streamdynamic_LSN_branch2_untrained_l.pth \
             ./data/output/dynamic_LSN_l_20240106/latest_ckpt.pth
