#!/bin/bash
EXP_DIR=dynamic_DAMO_s_lora_20240108_01
CFGS=cfgs/streamdynamic_DAMO_s
UNTRAINED_CKPT=./models/checkpoints/streamdynamic_DAMO_branch2_untrained_s.pth
LORA_CKPT=./data/output/$EXP_DIR/latest_ckpt.pth

# # before router train
# CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $CFGS \
#                      -c $UNTRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode random \
#                      --logfile before_router_train_random.txt \
#                      -d 1 -b 1 --conf 0.01 & # --fp16 &
#
# CUDA_VISIBLE_DEVICES=1 python tools/eval.py -f $CFGS \
#                      -c $UNTRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode max \
#                      --logfile before_router_train_router.txt \
#                      -d 1 -b 1 --conf 0.01 & # --fp16 &
#
# CUDA_VISIBLE_DEVICES=2 python tools/eval.py -f $CFGS \
#                      -c $UNTRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode max \
#                      --branch 0 \
#                      --logfile before_router_train_branch_0.txt \
#                      -d 1 -b 1 --conf 0.01 & # --fp16 &
#
# wait

# CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $CFGS \
#                      -c $UNTRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode max \
#                      --branch 1 \
#                      --logfile before_router_train_branch_1.txt \
#                      -d 1 -b 1 --conf 0.01 & # --fp16 &

# # after router train
# CUDA_VISIBLE_DEVICES=1 python tools/eval.py -f $CFGS \
#                      -c $UNTRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode random \
#                      --lora \
#                      --lora-rank 32 \
#                      --lora-ckpt $LORA_CKPT \
#                      --logfile after_router_train_random.txt \
#                      -d 1 -b 1 --conf 0.01 & # --fp16 &

CUDA_VISIBLE_DEVICES=2 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --lora \
                     --lora-rank 32 \
                     --lora-ckpt $LORA_CKPT \
                     --logfile after_router_train_router.txt \
                     -d 1 -b 1 --conf 0.01 # & # --fp16 &

# # wait
#
# CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $CFGS \
#                      -c $UNTRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode max \
#                      --lora \
#                      --lora-rank 32 \
#                      --lora-ckpt $LORA_CKPT \
#                      --branch 0 \
#                      --logfile after_router_train_branch_0.txt \
#                      -d 1 -b 1 --conf 0.01 & # --fp16 &
#
# CUDA_VISIBLE_DEVICES=1 python tools/eval.py -f $CFGS \
#                      -c $UNTRAINED_CKPT \
#                      --experiment-name $EXP_DIR \
#                      --router-mode max \
#                      --lora \
#                      --lora-rank 32 \
#                      --lora-ckpt $LORA_CKPT \
#                      --branch 1 \
#                      --logfile after_router_train_branch_1.txt \
#                      -d 1 -b 1 --conf 0.01 & # --fp16 &

wait

echo "done"
