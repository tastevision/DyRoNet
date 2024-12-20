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
        self.depth_t = 1
        self.width_t = 1
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

        self.short_cfg = [
            dict(frame_num=1, delta=1, with_short_cut=False, out_channels=[((
                round(128 * (self.backbone_width[0] / 1.0)),
                round(256 * (self.backbone_width[0] / 1.0)),
                round(512 * (self.backbone_width[0] / 1.0))
            ), 1), ],),
            dict(frame_num=1, delta=1, with_short_cut=False, out_channels=[((
                round(128 * (self.backbone_width[1] / 1.0)),
                round(256 * (self.backbone_width[1] / 1.0)),
                round(512 * (self.backbone_width[1] / 1.0))
            ), 1), ],),
            # dict(frame_num=1, delta=1, with_short_cut=False, out_channels=[((64, 128, 256), 1), ],),
            # dict(frame_num=1, delta=1, with_short_cut=False, out_channels=[((96, 192, 384), 1), ],),
        ]
        self.long_cfg = [
            dict(frame_num=3, delta=1, with_short_cut=False, include_current_frame=False, out_channels=[((
                round(42 * (self.backbone_width[0] / 1.0)),
                round(85 * (self.backbone_width[0] / 1.0)),
                round(170 * (self.backbone_width[0] / 1.0))
            ), 3), ],),
            dict(fusion_form='pure_concat', frame_num=3, delta=1, with_short_cut=False, include_current_frame=False, out_channels=[((
                round(42 * (self.backbone_width[1] / 1.0)),
                round(85 * (self.backbone_width[1] / 1.0)),
                round(170 * (self.backbone_width[1] / 1.0))
            ), 3), ],),
            # dict(frame_num=2, delta=1, with_short_cut=False, include_current_frame=False, out_channels=[((21, 42, 85), 2), ],),
            # dict(frame_num=3, delta=1, with_short_cut=False, include_current_frame=False, out_channels=[((32, 64, 128), 3), ],),
        ]
        self.branch_num = len(self.backbone_depth) # 分支数
        self.longest_frame_num = max([item["frame_num"] for item in self.long_cfg]) # 最长历史帧长度（用于数据集样本的获取）
        self.yolox_cfg = [
            dict(merge_form="long_fusion", with_short_cut=True),
            dict(merge_form="pure_concat", with_short_cut=True),
        ]
        self.neck_cfg = [
            {
                'depth': 1.0,
                'hidden_ratio': 0.75,
                'in_channels': [
                    round(256 * (self.backbone_width[0] / 1.0)),
                    round(512 * (self.backbone_width[0] / 1.0)),
                    round(1024 * (self.backbone_width[0] / 1.0)),
                ],
                'out_channels': [
                    round(256 * (self.backbone_width[0] / 1.0)),
                    round(512 * (self.backbone_width[0] / 1.0)),
                    round(1024 * (self.backbone_width[0] / 1.0)),
                ],
                'act': 'silu',
                'spp': False,
                'block_name': 'BasicBlock_3x3_Reverse',
                'dcn': True,
            },
        ]


    def get_model(self):
        from exps.model.dfp_pafpn import DFPPAFPN as TeacherBackbone
        from exps.model.yolox_DAMO_LSN import StreamDynamic_DAMO_LSN
        from exps.model.dynamic_long_backbone import DynamicLongBackbone as DAMOLongBackbone 
        from exps.model.dynamic_short_backbone import DynamicShortBackbone as DAMOShortBackbone
        from exps.model.neck_backbone import NeckBackbone as DAMONeckBackbone
        from exps.model.tal_head_dil import Head as TeacherHead
        from exps.model.tal_head_oddil import Head as StudentHead

        from exps.model.dfp_pafpn_long_v3 import DFPPAFPNLONGV3 as LSNLongBackbone
        from exps.model.dfp_pafpn_short_v3 import DFPPAFPNSHORTV3 as LSNShortBackbone
        from exps.model.longshort_backbone_neck_lsn import BACKBONENECK as LSNNeckBackbone
        from exps.model.tal_head import TALHead as LSNHead

        import torch.nn as nn
        from exps.model.speed_detector import SpeedDetector

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            long_backbone_s = nn.ModuleList([
                DAMOLongBackbone(self.backbone_depth[0], self.backbone_width[0], in_channels=in_channels, frame_num=self.long_cfg[0]["frame_num"],
                                 with_short_cut=self.long_cfg[0]["with_short_cut"], out_channels=self.long_cfg[0]["out_channels"]),
                LSNLongBackbone(self.backbone_depth[1], self.backbone_width[1], in_channels=in_channels, frame_num=self.long_cfg[1]["frame_num"],
                                with_short_cut=self.long_cfg[1]["with_short_cut"], out_channels=self.long_cfg[1]["out_channels"]),
            ])
            short_backbone_s = nn.ModuleList([
                DAMOShortBackbone(self.backbone_depth[0], self.backbone_width[0], in_channels=in_channels, frame_num=self.short_cfg[0]["frame_num"],
                                  with_short_cut=self.short_cfg[0]["with_short_cut"], out_channels=self.short_cfg[0]["out_channels"]),
                LSNShortBackbone(self.backbone_depth[1], self.backbone_width[1], in_channels=in_channels, frame_num=self.short_cfg[1]["frame_num"],
                                 with_short_cut=self.short_cfg[1]["with_short_cut"], out_channels=self.short_cfg[1]["out_channels"]),
            ])
            backbone_neck_s = nn.ModuleList([
                DAMONeckBackbone(self.backbone_depth[0], self.backbone_width[0], in_channels=in_channels, neck_cfg=self.neck_cfg[0]),
                LSNNeckBackbone(self.backbone_depth[1], self.backbone_width[1], in_channels=in_channels),
            ])

            head_s = nn.ModuleList([
                StudentHead(self.num_classes, self.backbone_width[0], in_channels=in_channels, gamma=1.0, ignore_thr=0.5, ignore_value=1.5,
                            hint_cls_factor=0.4, hint_obj_factor=0.4, hint_iou_factor=0.4, hint_cls_dil_enable=True, hint_obj_dil_enable=True,
                            hint_iou_dil_enable=True),
                LSNHead(self.num_classes, self.backbone_width[1], in_channels=in_channels, gamma=1.0, ignore_thr=0.5, ignore_value=1.5)
            ])

            # teacher model
            backbone_t = TeacherBackbone(self.depth_t, self.width_t, in_channels=in_channels)
            head_t = TeacherHead(self.num_classes, self.width_t, in_channels=in_channels, gamma=1.0,
                                ignore_thr=0.5, ignore_value=1.5, eval_decode=False)

            speed_detector = SpeedDetector(
                target_size=self.speed_detector_target_size,
                branch_num=self.branch_num,
            )

            self.model = StreamDynamic_DAMO_LSN(speed_detector,
                                                long_backbone_s, 
                                                short_backbone_s, 
                                                backbone_neck_s,
                                                head_s, 
                                                backbone_t=backbone_t,
                                                head_t=head_t,
                                                merge_form=[item['merge_form'] for item in self.yolox_cfg], 
                                                in_channels=in_channels, 
                                                width=self.backbone_width,
                                                with_short_cut=[item["with_short_cut"] for item in self.yolox_cfg],
                                                long_cfg=self.long_cfg)

            def para_count(block: nn.Module):
                cnt = 0
                for para in block.parameters():
                    cnt += para.view(-1).size(0)
                return cnt
            
            logger.info("模型参数量统计：")
            logger.info(f"speed_detector: {para_count(self.model.speed_detector.extractor)}")
            logger.info(f"speed_detector: {para_count(self.model.speed_detector.head)}")

            # self.model = YOLOXODDIL(backbone_s, head_s, backbone_t, head_t, 
            #     coef_cfg=self.coef_cfg, dil_neck_weight=0.5, still_teacher=False)

        self.model.apply(init_yolo)
        for i in range(self.branch_num):
            self.model.head[i].initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, local_rank=0, cache_img=False):
        from exps.dataset.longshort.tal_flip_long_short_argoversedataset import LONGSHORT_ARGOVERSEDataset
        from exps.dataset.distillation.longshort.tal_flip_long_short_argoversedataset_dil import LONGSHORT_Dil_ARGOVERSEDataset
        from exps.data.tal_flip_mosaicdetection import LongShortMosaicDetectionDil
        from exps.data.data_augment_flip import LongShortTrainTransformDil
        from yolox.data import (
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )

        dataset = LONGSHORT_Dil_ARGOVERSEDataset(
            data_dir='./data',
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=LongShortTrainTransformDil(max_labels=50, 
                                               hsv=False, 
                                               flip=True, 
                                               short_frame_num=self.short_cfg[-1]["frame_num"], 
                                               long_frame_num=self.longest_frame_num),
            cache=cache_img,
            short_cfg=self.short_cfg[-1],
            long_cfg=self.long_cfg[-1],
        )

        dataset = LongShortMosaicDetectionDil(dataset,
                                          mosaic=not no_aug,
                                          img_size=self.input_size,
                                          preproc=LongShortTrainTransformDil(max_labels=120, 
                                                                             hsv=False, 
                                                                             flip=True, 
                                                                             short_frame_num=self.short_cfg[-1]["frame_num"], 
                                                                             long_frame_num=self.longest_frame_num),
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
        # from exps.dataset.tal_flip_one_future_argoversedataset import ONE_ARGOVERSEDataset
        # from exps.data.data_augment_flip import DoubleValTransform
        from exps.dataset.longshort.tal_flip_long_short_argoversedataset import LONGSHORT_ARGOVERSEDataset
        from exps.data.data_augment_flip import LongShortValTransform

        valdataset = LONGSHORT_ARGOVERSEDataset(
            data_dir='./data',
            json_file='val.json',
            name='val',
            img_size=self.test_size,
            preproc=LongShortValTransform(short_frame_num=self.short_cfg[-1]["frame_num"],
                                          long_frame_num=self.longest_frame_num),
            short_cfg=self.short_cfg[-1],
            long_cfg=self.long_cfg[-1],
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
            inputs[0] = nn.functional.interpolate(
                inputs[0], size=tsize, mode="bilinear", align_corners=False
            )
            inputs[1] = nn.functional.interpolate(
                inputs[1], size=tsize, mode="bilinear", align_corners=False
            ) if inputs[1].ndim == 4 else inputs[1] # inputs[1].ndim != 4 为不使用long支路的情况
            targets[0][..., 1::2] = targets[0][..., 1::2] * scale_x
            targets[0][..., 2::2] = targets[0][..., 2::2] * scale_y
            targets[1][..., 1::2] = targets[1][..., 1::2] * scale_x
            targets[1][..., 2::2] = targets[1][..., 2::2] * scale_y
        return inputs, targets

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        # from exps.evaluators.onex_stream_evaluator import ONEX_COCOEvaluator
        from exps.evaluators.longshort_onex_stream_evaluator import LONGSHORT_ONEX_COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev)
        evaluator = LONGSHORT_ONEX_COCOEvaluator(
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
