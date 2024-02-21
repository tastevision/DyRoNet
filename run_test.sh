#!/bin/bash
EXP_DIR=streamnet_s_20231229
ROUTER_TRAIN_DIR=streamnet_s_20231226
CFGS=cfgs/streamnet_s
CKPT=./models/checkpoints/streamdynamic_branch2_untrained_s.pth

# before router train
CUDA_VISIBLE_DEVICES=0 python tools/train_dil.py -f $CFGS \
                          -c $CKPT \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode random \
                          --logfile before_router_train_random.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

CUDA_VISIBLE_DEVICES=1 python tools/train_dil.py -f $CFGS \
                          -c $CKPT \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode max \
                          --logfile before_router_train_router.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

CUDA_VISIBLE_DEVICES=2 python tools/train_dil.py -f $CFGS \
                          -c $CKPT \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode max \
                          --branch 0 \
                          --logfile before_router_train_branch_0.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

CUDA_VISIBLE_DEVICES=3 python tools/train_dil.py -f $CFGS \
                          -c $CKPT \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode max \
                          --branch 1 \
                          --logfile before_router_train_branch_1.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

wait

# after router train
CUDA_VISIBLE_DEVICES=0 python tools/train_dil.py -f $CFGS \
                          -c ./data/output/$ROUTER_TRAIN_DIR/latest_ckpt.pth \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode random \
                          --logfile after_router_train_random.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

CUDA_VISIBLE_DEVICES=1 python tools/train_dil.py -f $CFGS \
                          -c ./data/output/$ROUTER_TRAIN_DIR/latest_ckpt.pth \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode max \
                          --logfile after_router_train_router.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

CUDA_VISIBLE_DEVICES=2 python tools/train_dil.py -f $CFGS \
                          -c ./data/output/$ROUTER_TRAIN_DIR/latest_ckpt.pth \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode max \
                          --branch 0 \
                          --logfile after_router_train_branch_0.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

CUDA_VISIBLE_DEVICES=3 python tools/train_dil.py -f $CFGS \
                          -c ./data/output/$ROUTER_TRAIN_DIR/latest_ckpt.pth \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode max \
                          --branch 1 \
                          --logfile after_router_train_branch_1.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

wait

# after branch train
CUDA_VISIBLE_DEVICES=0 python tools/train_dil.py -f $CFGS \
                          -c ./data/output/$EXP_DIR/latest_ckpt.pth \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode random \
                          --logfile after_branch_train_random.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

CUDA_VISIBLE_DEVICES=1 python tools/train_dil.py -f $CFGS \
                          -c ./data/output/$EXP_DIR/latest_ckpt.pth \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode max \
                          --logfile after_branch_train_router.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

CUDA_VISIBLE_DEVICES=2 python tools/train_dil.py -f $CFGS \
                          -c ./data/output/$EXP_DIR/latest_ckpt.pth \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode max \
                          --branch 0 \
                          --logfile after_branch_train_branch_0.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

CUDA_VISIBLE_DEVICES=3 python tools/train_dil.py -f $CFGS \
                          -c ./data/output/$EXP_DIR/latest_ckpt.pth \
                          -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --experiment-name $EXP_DIR \
                          --router-mode max \
                          --branch 1 \
                          --logfile after_branch_train_branch_1.txt \
                          --eval-batch-size 1 \
                          -d 1 -b 8 --fp16 &

wait

echo "done"
