#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from loguru import logger

from exps.model.tal_head import TALHead
from exps.model.dfp_pafpn_long import DFPPAFPNLONG
from exps.model.dfp_pafpn_short import DFPPAFPNSHORT
from exps.model.dfp_pafpn_short_v2 import DFPPAFPNSHORTV2

from yolox.models.network_blocks import BaseConv

import numpy as np
from torch.nn import functional as F
import time


class YOLOXLONGSHORTV3(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(
        self, 
        speed_detector: nn.Module, # 环境速度检测器
        long_backbone: nn.Module,
        short_backbone: nn.Module,
        backbone_neck: nn.Module,
        head: nn.Module,
        merge_form, 
        in_channels, 
        width, 
        act="silu",
        with_short_cut=False,
        long_cfg=None,
    ):
        """Summary
        
        Args:
            long_backbone (None, optional): Description
            short_backbone (None, optional): Description
            head (None, optional): Description
            merge_form (str, optional): "add" or "concat" or "pure_concat" or "long_fusion"
            in_channels (list, optional): Description
        """
        super().__init__()
        if short_backbone is None:
            short_backbone = DFPPAFPNSHORTV2()
        if head is None:
            head = TALHead(20)

        self.long_backbone = long_backbone
        self.short_backbone = short_backbone
        self.backbone = backbone_neck
        self.head = head
        self.speed_detector = speed_detector
        self.merge_form = merge_form
        self.in_channels = in_channels
        self.with_short_cut = with_short_cut
        self.long_cfg = long_cfg
        self.branch_num = len(self.backbone)

        assert long_cfg is not None and "out_channels" in long_cfg[-1]
        self.jian2 = nn.ModuleList([BaseConv(
            in_channels=sum([x[0][0]*x[1] for x in self.long_cfg[i]["out_channels"]]),
            out_channels=int(in_channels[0] * width[i]) // 2,
            ksize=1,
            stride=1,
            act=act,
        ) for i in range(len(self.long_cfg))])

        self.jian1 = nn.ModuleList([BaseConv(
            in_channels=sum([x[0][1]*x[1] for x in self.long_cfg[i]["out_channels"]]),
            out_channels=int(in_channels[1] * width[i]) // 2,
            ksize=1,
            stride=1,
            act=act,
        ) for i in range(len(self.long_cfg))])

        self.jian0 = nn.ModuleList([BaseConv(
            in_channels=sum([x[0][2]*x[1] for x in self.long_cfg[i]["out_channels"]]),
            out_channels=int(in_channels[2] * width[i]) // 2,
            ksize=1,
            stride=1,
            act=act,
        ) for i in range(len(self.long_cfg))])

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
        self._module_state_change(self.short_backbone, False)
        self._module_state_change(self.long_backbone, False)
        for item in self.backbone:
            self._module_state_change(item, False)
        self._module_state_change(self.head, False)

    def unfreeze_main_model(self):
        """解冻主结构"""
        self._module_state_change(self.short_backbone, True)
        self._module_state_change(self.long_backbone, True)
        for item in self.backbone:
            self._module_state_change(item, True)
        self._module_state_change(self.head, True)

    def freeze_head(self):
        """冻结head"""
        self._module_state_change(self.head, False)

    def unfreeze_head(self):
        """解冻head"""
        self._module_state_change(self.head, True)

    def freeze_backbone(self):
        """冻结beackbone结构"""
        self._module_state_change(self.short_backbone, False)
        self._module_state_change(self.long_backbone, False)
        for item in self.backbone:
            self._module_state_change(item, False)

    def unfreeze_backbone(self):
        """解冻backbone结构"""
        self._module_state_change(self.short_backbone, True)
        self._module_state_change(self.long_backbone, True)
        for item in self.backbone:
            self._module_state_change(item, True)

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

    def long_short_fusion(self, short_outs, long_outs, rurrent_pan_outs, jian0, jian1, jian2, branch, merge_form='long_fusion'):
        """
        long和short输出的融合过程
        """
        if isinstance(self.merge_form, str):
            merge_form = self.merge_form
        elif isinstance(self.merge_form, list):
            merge_form = self.merge_form[branch]
        else:
            logger.warning(f"self.merge_form类型为{type(self.merge_form)}，这不对")
            merge_form = "long_fusion"

        if isinstance(self.with_short_cut, bool):
            with_short_cut = self.with_short_cut
        elif isinstance(self.with_short_cut, list):
            with_short_cut = self.with_short_cut[branch]
        else:
            logger.warning(f"self.with_short_cut类型为{type(self.with_short_cut)}，这不对")
            with_short_cut = True

        if not self.with_short_cut:
            if self.long_backbone is None:
                fpn_outs = short_outs
            else:
                if merge_form == "long_fusion":
                    fpn_outs_2 = torch.cat([short_outs[0], jian2(long_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([short_outs[1], jian1(long_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([short_outs[2], jian0(long_outs[2])], dim=1)
                elif merge_form == "pure_concat":
                    fpn_outs_2 = torch.cat([short_outs[0], long_outs[0]], dim=1)
                    fpn_outs_1 = torch.cat([short_outs[1], long_outs[1]], dim=1)
                    fpn_outs_0 = torch.cat([short_outs[2], long_outs[2]], dim=1)
                else:
                    raise ValueError(f"merge_form {merge_form} 未定义")
                fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
        else:
            if self.long_backbone is None:
                fpn_outs = [x + y for x, y in zip(short_outs, rurrent_pan_outs)]
            else:
                if merge_form == "long_fusion":
                    fpn_outs_2 = torch.cat([short_outs[0], jian2(long_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([short_outs[1], jian1(long_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([short_outs[2], jian0(long_outs[2])], dim=1)
                elif merge_form == "pure_concat":
                    fpn_outs_2 = torch.cat([short_outs[0], long_outs[0]], dim=1)
                    fpn_outs_1 = torch.cat([short_outs[1], long_outs[1]], dim=1)
                    fpn_outs_0 = torch.cat([short_outs[2], long_outs[2]], dim=1)
                else:
                    raise ValueError(f"merge_form {merge_form} 未定义")
                fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
        return fpn_outs

    def forward_process(self, x, N, buffer, test=False):
        """
        前向传播的主要过程
        """
        clock = time.time()
        if x[0].size()[1] == 3:
            short_fpn_outs, rurrent_pan_outs = self.short_backbone[N](x[0], buffer=buffer, mode='off_pipe', backbone_neck=self.backbone[N])
        else:
            short_fpn_outs, rurrent_pan_outs = self.short_backbone[N](x[0][:,:-3,...], buffer=buffer, mode='off_pipe', backbone_neck=self.backbone[N])
        self.inference_time += (time.time() - clock) / self.short_backbone[N].frame_num
        long_fpn_outs = self.long_backbone[N](x[1][:,:self.long_cfg[N]["frame_num"] * 3,...], self.long_cfg[N]["frame_num"], buffer=buffer, mode='off_pipe', backbone_neck=self.backbone[N]) if self.long_backbone is not None else None
        clock = time.time()
        fpn_outs = self.long_short_fusion(short_fpn_outs, long_fpn_outs, rurrent_pan_outs,
                                          self.jian0[N], self.jian1[N], self.jian2[N],
                                          branch=N, merge_form=self.long_cfg[N]['fusion_form'])
        self.inference_time += (time.time() - clock) / self.short_backbone[N].frame_num
        if not test:
            return fpn_outs
        else:
            clock = time.time()
            outputs = self.head[N](fpn_outs)  # [1, 11850, 13]
            self.inference_time += (time.time() - clock) / self.short_backbone[N].frame_num
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
        beg = time.time()
        outputs = dict()
        assert self.long_cfg is not None
        N = np.random.randint(0, len(self.long_cfg)) # 在分支中随机取一个
        idx = np.random.randint(x[0].size()[0])
        cur_x = (x[0][idx:idx+1,:], x[1][idx:idx+1,:])
        cur_targets = (targets[0][idx:idx+1], targets[1][idx:idx+1])

        # import pdb; pdb.set_trace()
        # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
        assert targets is not None
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
        assert self.long_cfg is not None
        batch_size = x[0].size()[0]

        speed_router_supervision_time = []
        speed_router_supervision_loss = []

        for idx in range(batch_size):
            # import pdb; pdb.set_trace()
            # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
            time_list = torch.zeros(len(self.long_cfg))
            loss_list = torch.zeros(len(self.long_cfg))
            for N in range(len(self.long_cfg)):
                beg = time.time()
                cur_x = (x[0][idx:idx+1,:], x[1][idx:idx+1,:])
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
        speed_router_supervision = F.one_hot(
            torch.argmin(
                speed_router_supervision_loss * F.softmax(speed_router_supervision_time, dim=1),
                dim=1
            ),
            num_classes=len(self.backbone)
        ).to(x[0].dtype)

        return speed_router_supervision

    def forward_train_offline_mode2(self, x, targets=None, buffer=None):
        """
        off_pipe, train_mode = 2
        3. 样本输入进来，拆分成单个样本，用router分类，再按各个样本的分类情况输入到不同的分支进行计算，正常输出 - train_mode=2
        """
        outputs = dict()
        assert self.long_cfg is not None
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
            cur_x = (x[0][idx:idx+1,:], x[1][idx:idx+1,:])
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
        assert self.long_cfg is not None
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
        outputs = dict()
        fpn_outs, buffer_ = self.backbone[0](x,  buffer=buffer, mode='on_pipe')
        outputs = self.head[N](fpn_outs)
        
        return outputs, buffer_
