#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    get_local_rank,
    get_model_info,
    setup_logger
)

from functools import reduce
import loralib as lora
from torch import nn

def update_ckpt(ckpt):

    res_ckpt = ckpt.copy()

    for k, v in ckpt.items():
        if k.startswith("backbone"):
            res_ckpt[f'short_{k}'] = v
            res_ckpt[f'long_{k}'] = v
        if k.startswith("neck"):
            res_ckpt[f'backbone.{k}'] = v

    return res_ckpt

def resume_branch(model, branch_num, ckpt_file):
    """
    载入单个分支的权重
    """
    logger.info(f"loading checkpoint for branch {branch_num}")
    ckpt = torch.load(ckpt_file)["model"]
    ckpt = update_ckpt(ckpt)

    # model = load_ckpt(model, ckpt)
    # 手动载入单个分支的权重
    model_state_dict = model.state_dict()
    load_dict = {}

    for key_model, v in model_state_dict.items():
        # logger.info(key_model)
        key_name_list = key_model.split(".")
        if key_name_list[1] == str(branch_num):
            if len(key_name_list) == 2:
                new_key = key_name_list[0]
            else:
                new_key = ".".join([key_name_list[0]] + key_name_list[2:])
            if new_key in ckpt:
                v_ckpt = ckpt[new_key]
                if v.shape != v_ckpt.shape:
                    logger.warning(f"Shape of {key_model} in checkpoint is {v_ckpt.shape}, while shape of {key_model} in model is {v.shape}.")
                    continue
            else:
                logger.warning(f"{new_key} 不在 {ckpt_file} 中")
                continue
        else:
            if key_model not in ckpt:
                # logger.warning(f"{key_model} is not in the ckpt. Please double check and see if this is desired.")
                continue
            v_ckpt = ckpt[key_model]
            if v.shape != v_ckpt.shape:
                logger.warning(f"Shape of {key_model} in checkpoint is {v_ckpt.shape}, while shape of {key_model} in model is {v.shape}.")
                continue
        load_dict[key_model] = v_ckpt

    model.load_state_dict(load_dict, strict=False)

    return model

def make_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-c1", "--ckpt1", default=None, type=str, help="checkpoint file for branch 1")
    parser.add_argument("-c2", "--ckpt2", default=None, type=str, help="checkpoint file for branch 2")
    parser.add_argument("-c3", "--ckpt3", default=None, type=str, help="checkpoint file for branch 3")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--logfile",
        type=str,
        default="val_log.txt",
        help="日志文件名",
    )
    parser.add_argument(
        "--router-mode",
        type=str,
        default="max",
        help="router的分支选择模式",
    )
    parser.add_argument("--branch", default=None, type=int, help="branch to inference")

    parser.add_argument(
        "--lora",
        action="store_true",
        default=False,
        help="是否进行LoRA的训练",
    )
    parser.add_argument(
        "--lora-ckpt",
        type=str,
        default=None,
        help="lora文件位置",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="lora保存时的秩",
    )
    return parser


def convert_model_to_lora(model, lora_rank):
    """
    将model中的Conv2d和Linear层替换为lora类型的
    """
    pretrained_model_state_dict = model.state_dict()
    updated_layer = {}
    lora_ignore_keys = ["speed_detector"]
    for mod in model.named_modules():
        ignore = False
        for name in lora_ignore_keys:
            if name in mod[0]:
                ignore = True
        if ignore:
            continue
        name_list = mod[0].split(".")
        if name_list == ['']:
            continue
        layer = reduce(getattr, [model] + name_list)

        if isinstance(layer, nn.Conv2d): # 获取nn.Conv2d的全部参数
            new_layer = lora.Conv2d(
                in_channels  = layer.in_channels, 
                out_channels = layer.out_channels,
                kernel_size  = layer.kernel_size[0],
                stride       = layer.stride,
                padding      = layer.padding,
                dilation     = layer.dilation,
                groups       = layer.groups,
                bias         = isinstance(layer.bias, torch.Tensor),
                padding_mode = layer.padding_mode,
                r            = lora_rank,                             # rank of LoRA
                lora_alpha   = 2,
            )
            setattr(
                reduce(getattr, [model] + name_list[:-1]),
                name_list[-1],
                new_layer,
            )
            updated_layer[mod[0] + ".weight"] = mod[0] + ".conv.weight"
            if isinstance(layer.bias, torch.Tensor):
                updated_layer[mod[0] + ".bias"] = mod[0] + ".conv.bias"

        if isinstance(layer, nn.Linear):
            # 线性层不需要改变
            new_layer = lora.Linear(
                in_features  = layer.in_features,
                out_features = layer.out_features,
                bias         = isinstance(layer.bias, torch.Tensor),
                r            = lora_rank,                             # rank of LoRA
                lora_alpha   = 2,
            )
            setattr(
                reduce(getattr, [model] + name_list[:-1]),
                name_list[-1],
                new_layer,
            )

        # 因为名称的变化，经过LoRA替换的网络层将存在原checkpoint
        # 载入失败的问题，需要返回已经修改过的层的名称并对其进行
        # 特殊处理

        # if isinstance(layer, nn.Embedding):
        #     new_layer = lora.Embedding(
        #         num_embeddings     = layer.num_embeddings,
        #         embedding_dim      = layer.embedding_dim,
        #         padding_idx        = layer.padding_idx,
        #         max_norm           = layer.max_norm,
        #         norm_type          = layer.norm_type,
        #         scale_grad_by_freq = layer.scale_grad_by_freq,
        #         sparse             = layer.sparse,
        #         _weight            = layer._weight,
        #         _freeze            = layer._freeze,
        #     )
        #     setattr(
        #         reduce(getattr, [model] + name_list[:-1]),
        #         name_list[-1],
        #         new_layer,
        #     )

    for k, v in updated_layer.items():
        old_value = pretrained_model_state_dict[k]
        del pretrained_model_state_dict[k]
        pretrained_model_state_dict[v] = old_value

    model.load_state_dict(pretrained_model_state_dict, strict=False)


@logger.catch
def main(exp, args, num_gpu):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = num_gpu > 1

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()

    file_name = os.path.join(exp.output_dir, args.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename=args.logfile, mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    # logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    # logger.info("Model Structure:\n{}".format(str(model)))

    # evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test, args.legacy)
    evaluator = exp.get_evaluator(args.batch_size, is_distributed, args.test)
    evaluator.per_class_AP = True
    evaluator.per_class_AR = True

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    if not args.speed and not args.trt:
        if args.ckpt:
            ckpt_file = args.ckpt
            logger.info("loading checkpoint from {}".format(ckpt_file))
            loc = "cuda:{}".format(rank)
            # 载入lora
            ckpt = torch.load(ckpt_file, map_location=loc)
            model.load_state_dict(ckpt["model"], strict=False)
        else:
            assert args.ckpt1 and args.ckpt2, "当前的实验参数没有提供args.ckpt，同时也没有提供args.ckpt1和args.ckpt2"
            model = resume_branch(model, 0, args.ckpt1)
            model = resume_branch(model, 1, args.ckpt2)
            if args.ckpt3:
                model = resume_branch(model, 2, args.ckpt3)

        if args.lora and args.lora_ckpt:
            logger.info(f"load lora checkpoint from {args.lora_ckpt}")
            convert_model_to_lora(model, args.lora_rank)
            model.cuda(rank)
            loc = "cuda:{}".format(rank)
            # 载入lora的训练权重
            lora_ckpt = torch.load(args.lora_ckpt, map_location=loc)
            weight_set = set()
            for n, _ in model.named_parameters():
                weight_set.add(n)
            for k, _ in lora_ckpt["model"].items():
                if k not in weight_set:
                    logger.info(f"{k} 不存在于model定义中")
            model.load_state_dict(lora_ckpt["model"], strict=False)
            logger.info("lora checkpoint was loaded.")

        logger.info("checkpoint was loaded.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert (
            not args.fuse and not is_distributed and args.batch_size == 1
        ), "TensorRT model is not support model fusing and distributed inferencing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    # start evaluate
    *_, summary = evaluator.evaluate(
        model, is_distributed, args.fp16, trt_file=trt_file, decoder=decoder,
        test_size=exp.test_size, mode=args.router_mode, branch=args.branch,
    )
    logger.info("\n" + summary)


if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )
