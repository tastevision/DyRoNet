#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from loguru import logger

from exps.model.tal_head import TALHead
from exps.model.dfp_pafpn import DFPPAFPN

import numpy as np
from torch.nn import functional as F
import time

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(
        self,
        speed_detector: nn.Module, # 环境速度检测器
        backbone: nn.Module,
        head: nn.Module,
        ):
        super().__init__()
        if backbone is None:
            backbone = DFPPAFPN()
        if head is None:
            head = TALHead(20)

        self.speed_detector = speed_detector
        self.backbone = backbone
        self.head = head
        self.branch_num = len(self.backbone)

        self.inference_time = 0.0 # 统计推理时长

    def compute_speed_score(self, x):
        """
        计算速度评分
        """
        speed_score = self.speed_detector(x)
        return speed_score

    def freeze_speed_detector(self):
        """冻结speed_detector"""
        self._module_state_change(self.speed_detector, False)

    def unfreeze_speed_detector(self):
        """解冻speed_detector"""
        self._module_state_change(self.speed_detector, True)

    def freeze_main_model(self):
        """冻结主结构"""
        self._module_state_change(self.backbone, False)
        self._module_state_change(self.head, False)

    def unfreeze_main_model(self):
        """解冻主结构"""
        self._module_state_change(self.backbone, True)
        self._module_state_change(self.head, True)

    def freeze_head(self):
        """只冻结head"""
        self._module_state_change(self.head, False)

    def unfreeze_head(self):
        """只打开head"""
        self._module_state_change(self.head, True)

    def freeze_backbone(self):
        """冻结beackbone结构"""
        self._module_state_change(self.backbone, False)

    def unfreeze_backbone(self):
        """解冻backbone结构"""
        self._module_state_change(self.backbone, True)

    def _module_state_change(self, module, state: bool):
        """转换模块的梯度状态"""
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = state

    def forward(self, x, targets=None, buffer=None, mode='off_pipe', train_mode=2, router_mode="max", branch=None):
        # fpn output content features of [dark3, dark4, dark5]
        assert mode in ['off_pipe', 'on_pipe']

        if self.training:
            if train_mode == 0:
                return self.forward_train_offline_mode0(x, targets, buffer)
            elif train_mode == 1:
                return self.forward_train_offline_mode1(x, targets, buffer)
            elif train_mode == 2:
                return self.forward_train_offline_mode2(x, targets, buffer)
        else:
            if mode == "off_pipe":
                return self.forward_test_offline(x, targets, buffer, mode=router_mode, branch=branch)
            elif mode == "on_pipe":
                return self.forward_test_online(x, targets, buffer)

    def forward_process(self, x, N, buffer, test=False):
        """
        前向传播的主要过程
        """
        clock = time.time()
        fpn_outs = self.backbone[N](x, buffer=buffer, mode='off_pipe')
        self.inference_time += (time.time() - clock) / 2

        # logger.info(f"分支数：{N}, fpn_outs.size(): {fpn_outs[0].size()}, {fpn_outs[1].size()}, {fpn_outs[2].size()}")

        if not test:
            return fpn_outs
        else:
            clock = time.time()
            outputs = self.head[N](fpn_outs)
            self.inference_time += (time.time() - clock)
            return outputs

    def compute_loss(self, fpn_outs, x, targets, N):
        """
        损失的计算
        """
        assert targets is not None
        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head[N](
            fpn_outs, targets, x
        )
        outputs = {
            "total_loss": loss,
            "iou_loss": iou_loss,
            "l1_loss": l1_loss,
            "conf_loss": conf_loss,
            "cls_loss": cls_loss,
            "num_fg": num_fg,
        }
        return outputs

    def forward_train_offline_mode0(self, x, targets=None, buffer=None):
        """
        off_pipe, train_mode = 0
        1. 样本输入进来，以随机的方式输入到其中的任何一个分支中，正常输出 - train_mode=0
        """
        assert targets is not None
        beg = time.time()
        outputs = dict()
        # N = np.random.randint(0, len(self.backbone)) # 在分支中随机取一个
        idx = 0
        N = 0
        # if isinstance(x, torch.Tensor):
        #     idx = np.random.randint(x.size()[0])
        # else:
        #     idx = np.random.randint(x[0].size()[0])
        cur_x = x[idx:idx+1,:]
        cur_targets = (targets[0][idx:idx+1], targets[1][idx:idx+1])

        # import pdb; pdb.set_trace()
        # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
        fpn_outs = self.forward_process(cur_x, N, buffer)
        
        losses = self.compute_loss(fpn_outs, cur_x, cur_targets, N)
        outputs.update(losses)
        end = time.time()
        logger.info(f"branch: {N}, time: {end - beg}")

        return outputs

    def forward_train_offline_mode1(self, x, targets=None, buffer=None):
        """
        off_pipe, train_mode = 1
        2. 样本输入进来，拆分成单个样本，其中每个样本都经过每个分支，得到其损失和时间，输入一个speed_router_supervision - train_mode=1
        """
        assert targets is not None
        if isinstance(x, torch.Tensor):
            batch_size = x.size()[0]
        else:
            batch_size = x[0].size()[0]

        speed_router_supervision_time = []
        speed_router_supervision_loss = []

        for idx in range(batch_size):
            # import pdb; pdb.set_trace()
            # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
            time_list = torch.zeros(len(self.backbone))
            loss_list = torch.zeros(len(self.backbone))
            for N in range(len(self.backbone)):
                beg = time.time()
                cur_x = x[idx:idx+1,:]
                cur_targets = (targets[0][idx:idx+1], targets[1][idx:idx+1])
                fpn_outs = self.forward_process(cur_x, N, buffer)
                total_loss = self.compute_loss(fpn_outs, cur_x, cur_targets, N)["total_loss"]
                end = time.time()
                time_list[N] = end - beg
                loss_list[N] = total_loss

            speed_router_supervision_time.append(time_list)
            speed_router_supervision_loss.append(loss_list)
            logger.info(f"当前batch的损失值如为: {loss_list}")

        speed_router_supervision_time = torch.stack(speed_router_supervision_time)
        speed_router_supervision_loss = torch.stack(speed_router_supervision_loss)
        # speed_router_supervision = F.gumbel_softmax(speed_router_supervision_loss * speed_router_supervision_time, dim=1)
        # speed_router_supervision = F.softmax(speed_router_supervision_loss * speed_router_supervision_time, dim=1)
        speed_router_supervision = F.one_hot(
            torch.argmin(
                speed_router_supervision_loss * F.softmax(speed_router_supervision_time, dim=1),
                dim=1
            ),
            num_classes=len(self.backbone)
        ).to(x.dtype)

        return speed_router_supervision

    def forward_train_offline_mode2(self, x, targets=None, buffer=None):
        """
        off_pipe, train_mode = 2
        3. 样本输入进来，拆分成单个样本，用router分类，再按各个样本的分类情况输入到不同的分支进行计算，正常输出 - train_mode=2
        """
        assert targets is not None
        outputs = dict()
        # 由self.speed_detector给出每个样本的最优分支
        with torch.no_grad():
            speed_score = self.compute_speed_score(x)
            branch_num_list = speed_score.argmax(dim=1)

        # # 只随机选择batch中的一个样本进行网络训练
        # idx = np.random.randint(len(branch_num_list))
        # N = branch_num_list[idx]
        # cur_x = x
        # cur_targets = targets

        for idx, N in enumerate(branch_num_list):
            cur_x = x[idx:idx+1,:]
            cur_targets = (targets[0][idx:idx+1], targets[1][idx:idx+1])

            # import pdb; pdb.set_trace()
            # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
            fpn_outs = self.forward_process(cur_x, N, buffer)

            losses = self.compute_loss(fpn_outs, cur_x, cur_targets, N)
            if outputs == dict():
                outputs.update(losses)
            else:
                for k, v in outputs.items():
                    outputs[k] += losses[k]

        if isinstance(outputs, list):
            return torch.cat(outputs)
        else:
            for k, v in outputs.items():
                outputs[k] /= len(branch_num_list) * 1.0

        return outputs

    def forward_test_offline(self, x, targets=None, buffer=None, mode="max", branch=None):
        """
        off_pipe test
        mode: router的分支选择模式
        """
        self.inference_time = 0.0 # 清空单次结果

        if branch is None:
            # 由self.speed_detector给出每个样本的最优分支
            clock = time.time()
            speed_score = self.compute_speed_score(x)
            self.inference_time += (time.time() - clock)
            branch_num_list = speed_score.argmax(dim=1)
            logger.info(f"当前的分支选择策略为 {mode}")
            if mode == "max":
                N = max(branch_num_list) # 选择最大的索引
            elif mode == "min":
                N = min(branch_num_list) # 选择最小的索引
            elif mode == "random":
                N = np.random.randint(self.branch_num) # 随机选择一个分支
            elif mode == "most":
                unique_v, cnt = torch.unique(branch_num_list, return_counts=True)
                N = int(unique_v[cnt.argmax()])
            else:
                raise ValueError(f"当前输入的分支选择模式 {mode} 没有被定义")
        else:
            N = branch
            
        logger.info(f"当前正在选择第{N}个分支进行预测")

        # for idx, N in enumerate(branch_num_list):
        # import pdb; pdb.set_trace()
        # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
        output = self.forward_process(x, N, buffer, test=True)
        logger.info(f"Online时间统计：{self.inference_time}")
        return output

    def forward_test_online(self, x, targets=None, buffer=None):
        """
        on_pipe test (没有实现)
        """
        speed_score = self.compute_speed_score(x)
        branch_num_list = speed_score.argmax(dim=1)
        N = max(branch_num_list) # 选择最大的索引
        fpn_outs, buffer_ = self.backbone[N](x,  buffer=buffer, mode='on_pipe')
        outputs = self.head[N](fpn_outs)
        
        return outputs, buffer_






