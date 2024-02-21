#!/bin/bash
function clear_gpu() {
ps -ef | grep xiang.huang | grep stream | grep python | cut -d ' ' -f 2 | xargs -n 1 kill -9;
ps -ef | grep xiang.huang | grep stream | grep python | cut -d ' ' -f 3 | xargs -n 1 kill -9;
ps -ef | grep xiang.huang | grep stream | grep python | cut -d ' ' -f 4 | xargs -n 1 kill -9;
}

# evaluate过程
function eval_process() {
EXP_DIR=$1
CFGS=$2
UNTRAINED_CKPT=$3
LORA_CKPT=$4
# if [[ `cat ./data/output/$EXP_DIR/train_log.txt | grep "Training of experiment is done" | wc -l` == 0 ]]; then
#     echo "检测到实验未完成，停止测试" >> ./data/output/$EXP_DIR/alert.txt;
#     return
# fi

# before router train
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode random \
                     --logfile before_router_train_random.txt \
                     -d 1 -b 1 --conf 0.01 &

CUDA_VISIBLE_DEVICES=1 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --logfile before_router_train_router.txt \
                     -d 1 -b 1 --conf 0.01 &

CUDA_VISIBLE_DEVICES=2 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --branch 0 \
                     --logfile before_router_train_branch_0.txt \
                     -d 1 -b 1 --conf 0.01 &

CUDA_VISIBLE_DEVICES=3 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --branch 1 \
                     --logfile before_router_train_branch_1.txt \
                     -d 1 -b 1 --conf 0.01 &

# after router train
CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode random \
                     --lora \
                     --lora-rank 32 \
                     --lora-ckpt $LORA_CKPT \
                     --logfile after_router_train_random.txt \
                     -d 1 -b 1 --conf 0.01 &

CUDA_VISIBLE_DEVICES=1 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --lora \
                     --lora-rank 32 \
                     --lora-ckpt $LORA_CKPT \
                     --logfile after_router_train_router.txt \
                     -d 1 -b 1 --conf 0.01 &

CUDA_VISIBLE_DEVICES=2 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --lora \
                     --lora-rank 32 \
                     --lora-ckpt $LORA_CKPT \
                     --branch 0 \
                     --logfile after_router_train_branch_0.txt \
                     -d 1 -b 1 --conf 0.01 &

CUDA_VISIBLE_DEVICES=3 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --lora \
                     --lora-rank 32 \
                     --lora-ckpt $LORA_CKPT \
                     --branch 1 \
                     --logfile after_router_train_branch_1.txt \
                     -d 1 -b 1 --conf 0.01 &

wait

CUDA_VISIBLE_DEVICES=0 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --branch 2 \
                     --logfile before_router_train_branch_2.txt \
                     -d 1 -b 1 --conf 0.01 &

CUDA_VISIBLE_DEVICES=1 python tools/eval.py -f $CFGS \
                     -c $UNTRAINED_CKPT \
                     --experiment-name $EXP_DIR \
                     --router-mode max \
                     --lora \
                     --lora-rank 32 \
                     --lora-ckpt $LORA_CKPT \
                     --branch 2 \
                     --logfile after_router_train_branch_2.txt \
                     -d 1 -b 1 --conf 0.01 &

echo "done"
}

# SYOLO s + m + l
# cur_cfg=cfgs/streamdynamic_SYOLO_s_m_l.py
# cur_ckpt=./models/checkpoints/streamdynamic_untrained_SYOLO_s_m_l.pth
# cur_exp_name=dynamic_SYOLO_s_m_l_lora_20240114
# python tools/train_dil.py -f $cur_cfg \
#                           --ckpt1 ./models/checkpoints/s_s50_one_x.pth \
#                           --ckpt2 ./models/checkpoints/m_s50_one_x.pth \
#                           --ckpt3 ./models/checkpoints/l_s50_one_x.pth \
#                           --logfile "test.txt" \
#                           --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# mv -v ./data/output/$cur_exp_name/best_ckpt.pth $cur_ckpt
# rm -v ./data/output/$cur_exp_name/*
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --router-mode max --lora --lora-rank 32 --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;

# LSN s + m + l
# cur_cfg=cfgs/streamdynamic_LSN_s_m_l.py
# cur_ckpt=./models/checkpoints/streamdynamic_untrained_LSN_s_m_l.pth
# cur_exp_name=dynamic_LSN_s_m_l_lora_20240115
# python tools/train_dil.py -f $cur_cfg \
#                           --ckpt1 ./models/checkpoints/longshortnet_s.pth \
#                           --ckpt2 ./models/checkpoints/longshortnet_m.pth \
#                           --ckpt3 ./models/checkpoints/longshortnet_l.pth \
#                           --logfile "test.txt" \
#                           --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# mv -v ./data/output/$cur_exp_name/best_ckpt.pth $cur_ckpt
# rm -v ./data/output/$cur_exp_name/*
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --router-mode max --lora --lora-rank 32 --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;

# DAMO s + m + l
cur_cfg=cfgs/streamdynamic_DAMO_s_m_l.py
cur_ckpt=./models/checkpoints/streamdynamic_untrained_DAMO_s_m_l.pth
cur_exp_name=dynamic_DAMO_s_m_l_lora_20240115
# python tools/train_dil.py -f $cur_cfg \
#                           --ckpt1 ./models/checkpoints/streamnet_s.pth \
#                           --ckpt2 ./models/checkpoints/streamnet_m.pth \
#                           --ckpt3 ./models/checkpoints/streamnet_l.pth \
#                           --logfile "test.txt" \
#                           --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# mv -v ./data/output/$cur_exp_name/best_ckpt.pth $cur_ckpt
# rm -v ./data/output/$cur_exp_name/*
python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
                          --router-mode max --lora --lora-rank 32 --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
clear_gpu;

