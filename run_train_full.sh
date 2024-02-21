#!/bin/bash
# 训练代码

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
TRAINED_CKPT=$4
if [[ `cat ./data/output/$EXP_DIR/train_log.txt | grep "Training of experiment is done" | wc -l` == 0 ]]; then
    echo "检测到实验未完成，停止测试" >> ./data/output/$EXP_DIR/alert.txt;
    return
fi

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

cur_cfg=cfgs/streamdynamic_DAMO_s_LSN_s
cur_ckpt=./models/checkpoints/streamdynamic_untrained_DAMO_s_LSN_s.pth
cur_exp_name=dynamic_DAMO_s_LSN_s_20240112
# python tools/train_dil.py -f $cur_cfg \
#                           --ckpt1 ./models/checkpoints/streamnet_s.pth \
#                           --ckpt2 ./models/checkpoints/longshortnet_s.pth \
#                           --logfile "test.txt" \
#                           -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# mv -v ./data/output/$cur_exp_name/best_ckpt.pth $cur_ckpt
# rm -v ./data/output/$cur_exp_name/*

# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
clear_gpu;


# # StreamYOLO s + l
# cur_cfg=cfgs/streamdynamic_SYOLO_s_SYOLO_l_new
# cur_ckpt=./models/checkpoints/streamdynamic_SYOLO_branch2_untrained_m.pth
# cur_exp_name=dynamic_SYOLO_s_SYOLO_l_20240113
# # python tools/train_dil.py -f $cur_cfg -c $cur_ckpt --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# # clear_gpu;

# cur_cfg=cfgs/streamdynamic_SYOLO_s_SYOLO_l_new
# cur_ckpt=./models/checkpoints/streamdynamic_SYOLO_branch2_untrained_m.pth
# cur_exp_name=dynamic_SYOLO_s_SYOLO_l_20240113_01
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;

# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;

# # DAMO-StreamNet s + m
# cur_cfg=cfgs/streamdynamic_DAMO_s
# cur_ckpt=./models/checkpoints/streamdynamic_DAMO_branch2_untrained_s.pth
# cur_exp_name=dynamic_DAMO_s_20240110
# # python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
# #                           --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;
#
# # LongShortNet s + m
# cur_cfg=cfgs/streamdynamic_LSN_s
# cur_ckpt=./models/checkpoints/streamdynamic_LSN_branch2_untrained_s.pth
# cur_exp_name=dynamic_LSN_s_20240110
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;
#
# # StreamYOLO s + m
# cur_cfg=cfgs/streamdynamic_SYOLO_s
# cur_ckpt=./models/checkpoints/streamdynamic_SYOLO_branch2_untrained_s.pth
# cur_exp_name=dynamic_SYOLO_s_20240110
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;
#
# # DAMO-StreamNet s + l
# cur_cfg=cfgs/streamdynamic_DAMO_m
# cur_ckpt=./models/checkpoints/streamdynamic_DAMO_branch2_untrained_m.pth
# cur_exp_name=dynamic_DAMO_m_20240110
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;
#
# # LongShortNet s + l
# cur_cfg=cfgs/streamdynamic_LSN_m
# cur_ckpt=./models/checkpoints/streamdynamic_LSN_branch2_untrained_m.pth
# cur_exp_name=dynamic_LSN_m_20240110
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;
#
# # StreamYOLO s + l
# cur_cfg=cfgs/streamdynamic_SYOLO_m
# cur_ckpt=./models/checkpoints/streamdynamic_SYOLO_branch2_untrained_m.pth
# cur_exp_name=dynamic_SYOLO_m_20240110
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;
#
# # DAMO-StreamNet m + l
# cur_cfg=cfgs/streamdynamic_DAMO_l
# cur_ckpt=./models/checkpoints/streamdynamic_DAMO_branch2_untrained_l.pth
# cur_exp_name=dynamic_DAMO_l_20240110
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt -t ./models/teacher_models/l_s50_still_dfp_flip_ep8_4_gpus_bs_8/best_ckpt.pth \
#                           --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;
#
# # LongShortNet m + l
# cur_cfg=cfgs/streamdynamic_LSN_l
# cur_ckpt=./models/checkpoints/streamdynamic_LSN_branch2_untrained_l.pth
# cur_exp_name=dynamic_LSN_l_20240110
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;
#
# # StreamYOLO m + l
# cur_cfg=cfgs/streamdynamic_SYOLO_l
# cur_ckpt=./models/checkpoints/streamdynamic_SYOLO_branch2_untrained_l.pth
# cur_exp_name=dynamic_SYOLO_l_20240110
# python tools/train_dil.py -f $cur_cfg -c $cur_ckpt --router-mode max --experiment-name $cur_exp_name --eval-batch-size 4 -d 4 -b 4 --fp16;
# eval_process $cur_exp_name $cur_cfg $cur_ckpt ./data/output/$cur_exp_name/latest_ckpt.pth;
# clear_gpu;
#
# # 保存预训练权重的示例
# # python tools/train_dil.py -f cfgs/streamdynamic_LSN_m \
# #                           --ckpt1 ./models/checkpoints/longshortnet_s.pth \
# #                           --ckpt2 ./models/checkpoints/longshortnet_l.pth \
# #                           --router-mode max \
# #                           --experiment-name dynamic_LSN_m_20240106 \
# #                           --logfile "test.txt" \
# #                           --eval-batch-size 4 \
# #                           -d 4 -b 4 --fp16
# #
# # mv -v ./data/output/dynamic_LSN_m_20240106/best_ckpt.pth ./models/checkpoints/streamdynamic_LSN_branch2_untrained_m.pth
# # rm -v ./data/output/dynamic_LSN_m_20240106/*

