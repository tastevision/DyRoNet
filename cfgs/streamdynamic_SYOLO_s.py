# encoding: utf-8
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from yolox.exp import Exp as MyExp
from loguru import logger
import math


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.backbone_depth = [0.33, 0.67]
        self.backbone_width = [0.50, 0.75]
        self.data_num_workers = 4
        self.num_classes = 8
        self.input_size = (600, 960)  # (h,w)
        self.random_size = (50, 70)
        self.test_size = (600, 960)
        #
        self.basic_lr_per_img = 0.001 / 64.0

        # 速度检测器的目标尺寸
        self.speed_detector_target_size = (100, 100)

        self.warmup_epochs = 1
        self.max_epoch = 5
        self.no_aug_epochs = self.max_epoch
        self.eval_interval = 1
        self.train_ann = 'train.json'
        self.val_ann = 'val.json'

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.output_dir = './data/output/'

        self.branch_num = len(self.backbone_depth) # 分支数

    def get_model(self):
        from exps.model.yolox import YOLOX
        from exps.model.dfp_pafpn import DFPPAFPN
        from exps.model.tal_head import TALHead
        import torch.nn as nn
        from exps.model.speed_detector import SpeedDetector

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = nn.ModuleList([
                DFPPAFPN(depth, width, in_channels=in_channels)
                for depth, width in zip(self.backbone_depth, self.backbone_width)
            ])
            head = nn.ModuleList([TALHead(
                self.num_classes,
                width,
                in_channels=in_channels,
                gamma=1.0,
                ignore_thr=0.5,
                ignore_value=1.5
            ) for width in self.backbone_width])

            speed_detector = SpeedDetector(
                target_size=self.speed_detector_target_size,
                branch_num=self.branch_num,
            )

            self.model = YOLOX(
                speed_detector,
                backbone,
                head,
            )

        self.model.apply(init_yolo)
        for i in range(self.branch_num):
            self.model.head[i].initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, local_rank=0, cache_img=False):
        from exps.dataset.tal_flip_one_future_argoversedataset import ONE_ARGOVERSEDataset
        from exps.data.tal_flip_mosaicdetection import MosaicDetection
        from exps.data.data_augment_flip import DoubleTrainTransform
        from yolox.data import (
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )

        dataset = ONE_ARGOVERSEDataset(
            data_dir='./data',
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=DoubleTrainTransform(max_labels=50, hsv=False, flip=True),
            cache=cache_img,
        )

        dataset = MosaicDetection(dataset,
                                  mosaic=not no_aug,
                                  img_size=self.input_size,
                                  preproc=DoubleTrainTransform(max_labels=120, hsv=False, flip=True),
                                  degrees=self.degrees,
                                  translate=self.translate,
                                  scale=self.mosaic_scale,
                                  shear=self.shear,
                                  perspective=0.0,
                                  enable_mixup=self.enable_mixup,
                                  mosaic_prob=self.mosaic_prob,
                                  mixup_prob=self.mixup_prob,
                                  )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from exps.dataset.tal_flip_one_future_argoversedataset import ONE_ARGOVERSEDataset
        from exps.data.data_augment_flip import DoubleValTransform

        valdataset = ONE_ARGOVERSEDataset(
            data_dir='./data',
            json_file='val.json',
            name='val',
            img_size=self.test_size,
            preproc=DoubleValTransform(),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler}
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        import random
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            if epoch >= self.max_epoch - 1:
                size = self.input_size
            else:
                size_factor = self.input_size[0] * 1.0 / self.input_size[1]
                size = random.randint(*self.random_size)
                size = (16 * int(size * size_factor), int(16 * size))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
    

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[0][..., 1::2] = targets[0][..., 1::2] * scale_x
            targets[0][..., 2::2] = targets[0][..., 2::2] * scale_y
            targets[1][..., 1::2] = targets[1][..., 1::2] * scale_x
            targets[1][..., 2::2] = targets[1][..., 2::2] * scale_y
        return inputs, targets

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        from exps.evaluators.onex_stream_evaluator import ONEX_COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev)
        evaluator = ONEX_COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator


    def get_trainer(self, args):
        from exps.train_utils.longshort_dil_trainer_dynamic import Trainer
        trainer = Trainer(self, args, self.branch_num)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False, mode="max", branch=None):
        return evaluator.evaluate(model, is_distributed, half, mode=mode, branch=branch)

    def get_optimizer(self, batch_size, ignore_keys=None):
        if "optimizer" not in self.__dict__:
            ignore_keys = ignore_keys if ignore_keys is not None else []
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                ig_flag = False
                for ig_k in ignore_keys:
                    if ig_k in k:
                        ig_flag = True
                        break
                if ig_flag:
                    continue

                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer
