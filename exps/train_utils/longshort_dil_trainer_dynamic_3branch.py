#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import datetime
import os
import time
from loguru import logger

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from torch.nn import functional as F
import numpy as np
from torch import nn

# from yolox.data import DataPrefetcher
from .longshort_data_prefetcher import DataPrefetcher
from .double_data_prefetcher import DataPrefetcher as SYOLODataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    # ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)
from .ema import ModelEMA

from functools import reduce
import loralib as lora


class Trainer:
    def __init__(self, exp: Exp, args, branch_num):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args
        self.branch_num = branch_num

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.speed_router_scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # 速度判断器的损失，需要在这里定义，在训练过程中生成必要的监督信息
        self.speed_lossfn = nn.KLDivLoss(reduction="batchmean", log_target=True)

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        self.ignore_keys = ["backbone_t", "head_t"]
        self.lora_ignore_keys = ["backbone_t", "head_t", "speed_detector"] # lora训练时需要忽略的模块

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename=self.args.logfile,
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        """
        在这里，根据self.epoch判断，训练2轮主结构(此时冻结speed_router)，训练1轮speed_router(此时冻结主结构)
        """

        # if self.epoch <= 4:
        #     logger.info("训练主结构，分支判断由随机数给出，保证基础的收敛状态")
        #     self.model.module.freeze_speed_detector()
        #     for self.iter in range(self.max_iter):
        #         self.before_iter()
        #         self.train_one_iter_mode_1()
        #         self.after_iter()
        #     self.model.module.unfreeze_speed_detector()

        if self.epoch % 2 == 0:
            logger.info("训练speed_router，冻结主结构")
            self.model.module.freeze_main_model()
            self.model.module.unfreeze_speed_detector()
            for n, p in self.model.module.named_parameters():
                logger.info(f"{n}: {p.requires_grad}")
            for self.iter in range(self.max_iter):
                self.before_iter()
                self.train_one_iter_mode_2()
                self.after_iter_2()
        else:
            logger.info("训练主结构，分支判断由speed_router给出，同时冻结speed_router")
            self.model.module.freeze_speed_detector()
            self.model.module.unfreeze_main_model()
            if self.args.lora:
                # 把两个分支中的lora部分视为可训练的
                logger.info("开启lora参数的训练")
                lora.mark_only_lora_as_trainable(self.model.module, bias='lora_only')
            for n, p in self.model.module.named_parameters():
                logger.info(f"{n}: {p.requires_grad}")
            for self.iter in range(self.max_iter):
                self.before_iter()
                self.train_one_iter_mode_3()
                self.after_iter()

    def train_one_iter_mode_1(self):
        """
        训练主结构，分支判断由随机数给出，保证基础的收敛状态
        """
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        # inps = inps.to(self.data_type)
        # targets = targets.to(self.data_type)
        # targets.requires_grad = False
        if isinstance(inps, torch.Tensor):
            inps = inps.to(self.data_type)
        else:
            inps = [inps[0].to(self.data_type), inps[1].to(self.data_type)]
        targets = (targets[0].to(self.data_type), targets[1].to(self.data_type))
        targets[0].requires_grad = False
        targets[1].requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets, train_mode=0)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def train_one_iter_mode_2(self):
        """
        训练speed_router，冻结主结构
        """
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        # inps = inps.to(self.data_type)
        # targets = targets.to(self.data_type)
        # targets.requires_grad = False
        if isinstance(inps, torch.Tensor):
            inps = inps.to(self.data_type)
        else:
            inps = [inps[0].to(self.data_type), inps[1].to(self.data_type)]
        targets = (targets[0].to(self.data_type), targets[1].to(self.data_type))
        targets[0].requires_grad = False
        targets[1].requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            with torch.no_grad():
                speed_supervision = self.model(inps, targets, train_mode=1)
                logger.info(f"speed router supervision: {speed_supervision}")

            speed_supervision = speed_supervision.to(self.device)
            speed_score = self.model.module.compute_speed_score(inps)
            speed_loss = self.speed_lossfn(speed_score, speed_supervision)

        self.speed_router_optimizer.zero_grad()
        self.speed_router_scaler.scale(speed_loss).backward()
        self.speed_router_scaler.step(self.speed_router_optimizer)
        self.speed_router_scaler.update()

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.speed_router_optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        logger.info(f"iter time: f{iter_end_time - iter_start_time}, speed router loss: f{speed_loss}")


    def train_one_iter_mode_3(self):
        """
        训练主结构，分支判断由speed_router给出，同时冻结speed_router
        """
        iter_start_time = time.time()
        inps, targets = self.prefetcher.next()
        # inps = inps.to(self.data_type)
        # targets = targets.to(self.data_type)
        # targets.requires_grad = False
        if isinstance(inps, torch.Tensor):
            inps = inps.to(self.data_type)
        else:
            inps = [inps[0].to(self.data_type), inps[1].to(self.data_type)]
        targets = (targets[0].to(self.data_type), targets[1].to(self.data_type))
        targets[0].requires_grad = False
        targets[1].requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets, train_mode=2)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        # logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size, ignore_keys=self.ignore_keys)
        self.speed_router_optimizer = torch.optim.SGD(model.speed_detector.parameters(), lr=0.01, momentum=0.9)

        # 需要在这里特别地将model.long_backbone加载到gpu上
        if hasattr(model, 'jian0'):
            for m in model.jian0:
                m.to(self.device)
            for m in model.jian1:
                m.to(self.device)
            for m in model.jian2:
                m.to(self.device)
        model.speed_detector.to(self.device)

        # logger.info(
        #     "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        # )
        model.to(self.device)

        # value of epoch will be set in `resume_train`
        if self.args.ckpt or self.args.resume:
            assert self.args.ckpt, "需要重新载入权重，但是没有提供args.ckpt"
            model = self.resume_train(model)
        else:
            assert self.args.ckpt1 and self.args.ckpt2 and self.args.ckpt3, "当前的实验参数没有提供args.ckpt，同时也没有提供args.ckpt1和args.ckpt2"
            model = self.resume_branch(model, 0, self.args.ckpt1)
            model = self.resume_branch(model, 1, self.args.ckpt2)
            model = self.resume_branch(model, 2, self.args.ckpt3)

        # LoRA状态的转换以及权重加载
        if self.args.lora:
            # if not self.args.resume: # 此时证明是首次的训练，需要初始化lora
            self.convert_model_to_lora(model)
            lora.mark_only_lora_as_trainable(model, bias='lora_only')

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        if self.args.logfile == "train_log.txt":
            logger.info("init prefetcher, this might take one minute or less...")
            if "SYOLO" in self.args.experiment_name:
                self.prefetcher = SYOLODataPrefetcher(self.train_loader) # 对于streamyolo来说，data prefetcher的定义是不同的
            else:
                self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        # self.speed_router_lr_scheduler = self.exp.get_router_lr_scheduler(
        #     self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        # )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False, find_unused_parameters=True)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998, ignore_keys=self.ignore_keys)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                wandb_params = dict()
                for k, v in zip(self.args.opts[0::2], self.args.opts[1::2]):
                    if k.startswith("wandb-"):
                        wandb_params.update({k.lstrip("wandb-"): v})
                self.wandb_logger = WandbLogger(config=vars(self.exp), **wandb_params)
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        # logger.info("\n{}".format(model))

    def after_train(self):
        if self.epoch < self.max_epoch - 1:
            logger.warning("出现错误，训练过程提前终止")
        else:
            logger.info(
                "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
            )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.args.logfile != "train_log.txt":
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()
        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                # TODO: 在这里，如果分支数增加的话，需要重新添加
                self.model.module.head[0].use_l1 = True
                self.model.module.head[1].use_l1 = True
                self.model.module.head[2].use_l1 = True
            else:
                self.model.head[0].use_l1 = True
                self.model.head[1].use_l1 = True
                self.model.head[2].use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        if self.args.logfile == "train_log.txt":
            logger.info("clear self.prefetcher and release the cache")
            del self.train_loader
            del self.prefetcher
            # torch.cuda.empty_cache()
            if (self.epoch + 1) % self.exp.eval_interval == 0:
                all_reduce_norm(self.model)
                self.evaluate_and_save_model()
            torch.cuda.empty_cache()
            logger.info("init prefetcher, this might take one minute or less...")
            self.train_loader = self.exp.get_data_loader(
                batch_size=self.args.batch_size,
                is_distributed=self.is_distributed,
                no_aug=self.no_aug,
                cache_img=self.args.cache,
            )
            if "SYOLO" in self.args.experiment_name:
                self.prefetcher = SYOLODataPrefetcher(self.train_loader) # 对于streamyolo来说，data prefetcher的定义是不同的
            else:
                self.prefetcher = DataPrefetcher(self.train_loader)

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "wandb":
                    self.wandb_logger.log_metrics({k: v.latest for k, v in loss_meter.items()})
                    self.wandb_logger.log_metrics({"lr": self.meter["lr"].latest})

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    def after_iter_2(self):
        """
        `after_iter` contains two parts of logic:
            * reset setting of resize
        """
        progress_str = "epoch: {}/{}, iter: {}/{}".format(
            self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
        )
        logger.info(progress_str)
        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_branch(self, model, branch_num, ckpt_file):
        """
        载入单个分支的权重
        """
        logger.info(f"loading checkpoint for branch {branch_num}")
        ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
        ckpt = self.update_ckpt(ckpt)
        if self.args.teacher_ckpt:
            teacher_ckpt_file = self.args.teacher_ckpt
            teacher_ckpt = torch.load(teacher_ckpt_file, map_location=self.device)["model"]

            for k, v in teacher_ckpt.items():
                if "head." in k:
                    k_new = f'head_t.{k[len("head."):]}'
                elif "backbone." in k:
                    k_new = f'backbone_t.{k[len("backbone."):]}'
                else:
                    raise Exception("Load teacher ckpt error.")
                ckpt[k_new] = v

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

        self.start_epoch = 0
        return model

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt
            ckpt = torch.load(ckpt_file, map_location=self.device)
            if self.args.teacher_ckpt:
                teacher_ckpt_file = self.args.teacher_ckpt
                teacher_ckpt = torch.load(teacher_ckpt_file, map_location=self.device)["model"]

                for k, v in teacher_ckpt.items():
                    if "head." in k:
                        k_new = f'head_t.{k[len("head."):]}'
                    elif "backbone." in k:
                        k_new = f'backbone_t.{k[len("backbone."):]}'
                    else:
                        raise Exception("Load teacher ckpt error.")
                    ckpt["model"][k_new] = v
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            if self.args.lora:
                lora_ckpt = torch.load(self.args.lora_ckpt, map_location=self.device)
                model.load_state_dict(lora_ckpt["model"], strict=False)
                self.optimizer.load_state_dict(lora_ckpt["optimizer"])
                self.speed_router_optimizer.load_state_dict(lora_ckpt["speed_router_optimizer"])
                self.best_ap = lora_ckpt.pop("best_ap", 0)
                start_epoch = (
                    self.args.start_epoch - 1
                    if self.args.start_epoch is not None
                    else lora_ckpt["start_epoch"]
                )
            else:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                self.speed_router_optimizer.load_state_dict(ckpt["speed_router_optimizer"])
                self.best_ap = ckpt.pop("best_ap", 0)
                start_epoch = (
                    self.args.start_epoch - 1
                    if self.args.start_epoch is not None
                    else ckpt["start_epoch"]
                )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            # 此时说明不是重载的训练，以下应该是没有lora参与的
            # 同时，在这里的参数载入也不需要考虑优化器的载入等问题
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                logger.info(f"正在载入 {self.args.ckpt}")
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                ckpt = self.update_ckpt(ckpt)
                if self.args.teacher_ckpt:
                    teacher_ckpt_file = self.args.teacher_ckpt
                    teacher_ckpt = torch.load(teacher_ckpt_file, map_location=self.device)["model"]

                    for k, v in teacher_ckpt.items():
                        if "head." in k:
                            k_new = f'head_t.{k[len("head."):]}'
                        elif "backbone." in k:
                            k_new = f'backbone_t.{k[len("backbone."):]}'
                        else:
                            raise Exception("Load teacher ckpt error.")
                        ckpt[k_new] = v
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.eval_batch_size, is_distributed=self.is_distributed
        )
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            ap50_95, ap50, summary = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, mode=self.args.router_mode, branch=self.args.branch,
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "epoch": self.epoch + 1,
                })
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")
        del self.evaluator

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        # 此处加入LoRA训练过程中的写法
        if self.rank != 0:
            return
        logger.info("Save weights to {}".format(self.file_name))
        save_model = self.ema_model.ema if self.use_model_ema else self.model

        if self.args.lora:
            # 在convert_model_to_lora函数中已经跳过了self.ignore_keys，不需重复
            logger.info(f"LoRA已经开启，正在保存LoRA权重")
            save_state_dict = lora.lora_state_dict(save_model) # 只保存lora生效的那些层
            # lora启用时，单独保存speed router的权重
            logger.info(f"LoRA已经开启，同时也保存speed router的权重")
            for k, v in save_model.state_dict().items():
                if "speed_detector" in k:
                    logger.info(f"正在保存 {k}")
                    save_state_dict[k] = v
        else:
            save_state_dict = dict()
            for k, v in save_model.state_dict().items():
                ig_flag = False
                for ig_k in self.ignore_keys:
                    if ig_k in k:
                        ig_flag = True
                        break
                if ig_flag:
                    continue
                save_state_dict[k] = v

        ckpt_state = {
            "start_epoch": self.epoch + 1,
            "model": save_state_dict,
            "optimizer": self.optimizer.state_dict(),
            "speed_router_optimizer": self.speed_router_optimizer.state_dict(),
            "best_ap": self.best_ap,
        }
        save_checkpoint(
            ckpt_state,
            update_best_ckpt,
            self.file_name,
            ckpt_name,
        )

        if self.args.logger == "wandb":
            self.wandb_logger.save_checkpoint(self.file_name, ckpt_name, update_best_ckpt)

    def update_ckpt(self, ckpt):

        res_ckpt = ckpt.copy()

        for k, v in ckpt.items():
            if k.startswith("backbone"):
                res_ckpt[f'short_{k}'] = v
                res_ckpt[f'long_{k}'] = v
            if k.startswith("neck"):
                res_ckpt[f'backbone.{k}'] = v

        return res_ckpt

    def convert_model_to_lora(self, model):
        """
        将model中的Conv2d和Linear层替换为lora类型的
        """
        pretrained_model_state_dict = model.state_dict()
        updated_layer = {}
        for mod in model.named_modules():
            # 不处理位于self.ignore_keys中的层
            ignore = False
            for name in self.lora_ignore_keys:
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
                    r            = self.args.lora_rank,                       # rank of LoRA
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
                    r            = self.args.lora_rank,                       # rank of LoRA
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
        model.to(self.device)

