#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

from exps.model.tal_head import TALHead
from exps.model.dfp_pafpn_long import DFPPAFPNLONG
from exps.model.dfp_pafpn_short_v2 import DFPPAFPNSHORTV2
import time

from yolox.models.network_blocks import BaseConv
import numpy as np
from torch.nn import functional as F


class YOLOXLONGSHORTODDILDYNAMIC(nn.Module):
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
        backbone_t: nn.Module,
        head_t: nn.Module,
        merge_form="add", 
        in_channels=[256, 512, 1024], 
        width=1.0, 
        act="silu",
        with_short_cut=False,
        long_cfg=None,
        coef_cfg=None,
        dil_loc="head",
        # still_teacher=True,
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

        self.dil_loc = dil_loc 
        self.long_backbone = long_backbone
        self.short_backbone = short_backbone
        self.backbone = backbone_neck
        self.head = head
        self.backbone_t = backbone_t # teacher model
        self.head_t = head_t
        self.speed_detector = speed_detector
        self.merge_form = merge_form
        self.in_channels = in_channels
        self.with_short_cut = with_short_cut
        self._freeze_teacher_model()
        self._set_eval_teacher_model()
        # self.dil_loss = nn.MSELoss(reduction='sum')
        # coef_cfg = coef_cfg if coef_cfg is not None else dict()
        # self.dil_loss_coef = coef_cfg.get("dil_loss_coef", 1.)
        # self.det_loss_coef = coef_cfg.get("det_loss_coef", 1.)
        # self.reg_coef = coef_cfg.get("reg_coef", 1.)
        # self.cls_coef = coef_cfg.get("cls_coef", 1.)
        # self.obj_coef = coef_cfg.get("obj_coef", 1.)
        self.long_cfg = long_cfg

        if merge_form == "concat":
            self.jian2 = BaseConv(
                        in_channels=int(in_channels[0] * width),
                        out_channels=int(in_channels[0] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian1 = BaseConv(
                        in_channels=int(in_channels[1] * width),
                        out_channels=int(in_channels[1] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )

            self.jian0 = BaseConv(
                        in_channels=int(in_channels[2] * width),
                        out_channels=int(in_channels[2] * width) // 2,
                        ksize=1,
                        stride=1,
                        act=act,
                    )
        elif merge_form == "long_fusion":
            assert long_cfg is not None and "out_channels" in long_cfg
            self.jian2 = nn.ModuleList([BaseConv(
                in_channels=sum([x[0][0]*(i + 1) for x in self.long_cfg["out_channels"]]),
                out_channels=int(in_channels[0] * width) // 2,
                ksize=1,
                stride=1,
                act=act,
            ) for i in range(self.long_cfg["frame_num"])])

            self.jian1 = nn.ModuleList([BaseConv(
                in_channels=sum([x[0][1]*(i + 1) for x in self.long_cfg["out_channels"]]),
                out_channels=int(in_channels[1] * width) // 2,
                ksize=1,
                stride=1,
                act=act,
            ) for i in range(self.long_cfg["frame_num"])])

            self.jian0 = nn.ModuleList([BaseConv(
                in_channels=sum([x[0][2]*(i + 1) for x in self.long_cfg["out_channels"]]),
                out_channels=int(in_channels[2] * width) // 2,
                ksize=1,
                stride=1,
                act=act,
            ) for i in range(self.long_cfg["frame_num"])])


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
        self._module_state_change(self.backbone, False)
        self._module_state_change(self.head, False)

    def unfreeze_main_model(self):
        """解冻主结构"""
        self._module_state_change(self.short_backbone, True)
        self._module_state_change(self.long_backbone, True)
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
        self._module_state_change(self.short_backbone, False)
        self._module_state_change(self.long_backbone, False)
        self._module_state_change(self.backbone, False)

    def unfreeze_backbone(self):
        """解冻backbone结构"""
        self._module_state_change(self.short_backbone, True)
        self._module_state_change(self.long_backbone, True)
        self._module_state_change(self.backbone, True)

    def _module_state_change(self, module, state: bool):
        """转换模块的梯度状态"""
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = state

    def _module_state_change(self, module, state: bool):
        """转换模块的梯度状态"""
        if isinstance(module, nn.Module):
            for param in module.parameters():
                param.requires_grad = state

    def forward(self, x, targets=None, buffer=None, mode='off_pipe', train_mode=2):
        """
        设置三种训练模式：
        1. 样本输入进来，以随机的方式输入到其中的任何一个分支中，正常输出 - train_mode=0
        2. 样本输入进来，拆分成单个样本，其中每个样本都经过每个分支，得到其损失和时间，输入一个speed_router_supervision - train_mode=1
        3. 样本输入进来，拆分成单个样本，用router分类，再按各个样本的分类情况输入到不同的分支进行计算，正常输出 - train_mode=2
        """
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
                return self.forward_test_offline(x, targets, buffer)
            elif mode == "on_pipe":
                return self.forward_test_online(x, targets, buffer)

    def forward_train_offline_mode0(self, x, targets=None, buffer=None):
        """
        off_pipe, train_mode = 0
        1. 样本输入进来，以随机的方式输入到其中的任何一个分支中，正常输出 - train_mode=0
        """
        outputs = dict()
        assert self.long_cfg is not None
        N = np.random.randint(0, self.long_cfg["frame_num"]) # 在分支中随机取一个
        idx = np.random.randint(x[0].size()[0])
        cur_x = (x[0][idx:idx+1,:], x[1][idx:idx+1,:])
        cur_targets = (targets[0][idx:idx+1], targets[1][idx:idx+1])

        # import pdb; pdb.set_trace()
        # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
        short_fpn_outs, rurrent_pan_outs = self.short_backbone(cur_x[0][:,:-3,...], buffer=buffer, mode='off_pipe', backbone_neck=self.backbone)
        long_fpn_outs = self.long_backbone(cur_x[1][:,:(N + 1) * 3,...], N + 1, buffer=buffer, mode='off_pipe', backbone_neck=self.backbone) if self.long_backbone is not None else None
        fpn_outs_t = self.backbone_t(cur_x[0][:,-3:,...], buffer=buffer, mode='off_pipe')

        if not self.with_short_cut:
            if self.long_backbone is None:
                fpn_outs = short_fpn_outs
            else:
                if self.merge_form == "add":
                    fpn_outs = [x + y for x, y in zip(short_fpn_outs, long_fpn_outs)]
                elif self.merge_form == "concat":
                    fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                elif self.merge_form == "pure_concat":
                    fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                    fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                    fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                elif self.merge_form == "long_fusion":
                    fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                else:
                    raise Exception(f'merge_form must be in ["add", "concat"]')
        else:
            if self.long_backbone is None:
                fpn_outs = [x + y for x, y in zip(short_fpn_outs, rurrent_pan_outs)]
            else:
                if self.merge_form == "add":
                    fpn_outs = [x + y + z for x, y, z in zip(short_fpn_outs, long_fpn_outs, rurrent_pan_outs)]
                elif self.merge_form == "concat":
                    fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                elif self.merge_form == "pure_concat":
                    fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                    fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                    fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                elif self.merge_form == "long_fusion":
                    fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                else:
                    raise Exception(f'merge_form must be in ["add", "concat"]')

        assert targets is not None
        # (loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg), reg_outputs, obj_outputs, cls_outputs = self.head(
        #     fpn_outs, targets, x
        # )
        bbox_preds_t, obj_preds_t, cls_preds_t = self.head_t(fpn_outs_t)
        
        if self.dil_loc == "head":
            knowledge = (bbox_preds_t, obj_preds_t, cls_preds_t)
            (
                loss, 
                iou_loss, 
                conf_loss, 
                cls_loss, 
                l1_loss, 
                reg_dil_loss,
                obj_dil_loss,
                cls_dil_loss,
                loss_dil_hint,
                num_fg
            ) = self.head(fpn_outs, cur_targets, cur_x, knowledge=knowledge)

            losses = {
                "total_loss": loss,
                # "det_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                # "dil_loss": dil_loss,
                # "neck_dil_loss":neck_dil_loss,
                "reg_dil_loss": reg_dil_loss,
                "cls_dil_loss": cls_dil_loss,
                "obj_dil_loss": obj_dil_loss,
                "loss_dil_hint":loss_dil_hint,
                "num_fg": num_fg,
            }

            outputs.update(losses)

        else:
            outputs = self.head(fpn_outs)

        return outputs

    def forward_train_offline_mode1(self, x, targets=None, buffer=None):
        """
        off_pipe, train_mode = 1
        2. 样本输入进来，拆分成单个样本，其中每个样本都经过每个分支，得到其损失和时间，输入一个speed_router_supervision - train_mode=1
        """
        outputs = dict()
        assert self.long_cfg is not None
        batch_size = x[0].size()[0]

        speed_router_supervision_time = []
        speed_router_supervision_loss = []

        time_list_new = [0, 0, 0, 0] # total time, time of short_backbone, time of long_backbone, time of backbone_t
        for idx in range(batch_size):
            # import pdb; pdb.set_trace()
            # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
            time_list = torch.zeros(self.long_cfg["frame_num"])
            loss_list = torch.zeros(self.long_cfg["frame_num"])
            for N in range(self.long_cfg["frame_num"]):
                beg = time.time()
                cur_x = (x[0][idx:idx+1,:], x[1][idx:idx+1,:])
                cur_targets = (targets[0][idx:idx+1], targets[1][idx:idx+1])

                tick1 = time.time()
                short_fpn_outs, rurrent_pan_outs = self.short_backbone(cur_x[0][:,:-3,...], buffer=buffer, mode='off_pipe', backbone_neck=self.backbone)
                tick2 = time.time()
                fpn_outs_t = self.backbone_t(cur_x[0][:,-3:,...], buffer=buffer, mode='off_pipe')
                tick3 = time.time()
                long_fpn_outs = self.long_backbone(cur_x[1][:,:(N + 1) * 3,...], N + 1, buffer=buffer, mode='off_pipe', backbone_neck=self.backbone) if self.long_backbone is not None else None
                tick4 = time.time()

                time_list_new[1] += tick2 - tick1
                time_list_new[2] += tick4 - tick3
                time_list_new[3] += tick3 - tick2

                if not self.with_short_cut:
                    if self.long_backbone is None:
                        fpn_outs = short_fpn_outs
                    else:
                        if self.merge_form == "add":
                            fpn_outs = [x + y for x, y in zip(short_fpn_outs, long_fpn_outs)]
                        elif self.merge_form == "concat":
                            fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                            fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                            fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                            fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        elif self.merge_form == "pure_concat":
                            fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                            fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                            fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                            fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        elif self.merge_form == "long_fusion":
                            fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                            fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                            fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                            fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        else:
                            raise Exception(f'merge_form must be in ["add", "concat"]')
                else:
                    if self.long_backbone is None:
                        fpn_outs = [x + y for x, y in zip(short_fpn_outs, rurrent_pan_outs)]
                    else:
                        if self.merge_form == "add":
                            fpn_outs = [x + y + z for x, y, z in zip(short_fpn_outs, long_fpn_outs, rurrent_pan_outs)]
                        elif self.merge_form == "concat":
                            fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                            fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                            fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                            fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                            fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                        elif self.merge_form == "pure_concat":
                            fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                            fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                            fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                            fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                            fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                        elif self.merge_form == "long_fusion":
                            fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                            fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                            fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                            fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                            fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                        else:
                            raise Exception(f'merge_form must be in ["add", "concat"]')

                bbox_preds_t, obj_preds_t, cls_preds_t = self.head_t(fpn_outs_t)

                if self.dil_loc == "head":
                    knowledge = (bbox_preds_t, obj_preds_t, cls_preds_t)
                    total_loss = self.head(fpn_outs, cur_targets, cur_x, knowledge=knowledge)[0]
                else:
                    total_loss = self.head(fpn_outs)[0]

                end = time.time()
                time_list_new[0] += end - beg
                time_list[N] = 1.0 / (N + 1)
                loss_list[N] = total_loss

            speed_router_supervision_time.append(time_list)
            speed_router_supervision_loss.append(loss_list)

        speed_router_supervision_time = F.softmax(torch.stack(speed_router_supervision_time), dim=1)
        speed_router_supervision_loss = F.softmax(torch.stack(speed_router_supervision_loss), dim=1)
        speed_router_supervision = F.softmax(speed_router_supervision_loss * speed_router_supervision_time, dim=1)

        return speed_router_supervision, time_list_new

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
            branch_num_list = speed_score.argmin(dim=1)

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
            short_fpn_outs, rurrent_pan_outs = self.short_backbone(cur_x[0][:,:-3,...], buffer=buffer, mode='off_pipe', backbone_neck=self.backbone)
            fpn_outs_t = self.backbone_t(cur_x[0][:,-3:,...], buffer=buffer, mode='off_pipe')

            long_fpn_outs = self.long_backbone(cur_x[1][:,:(N + 1) * 3,...], N + 1, buffer=buffer, mode='off_pipe', backbone_neck=self.backbone) if self.long_backbone is not None else None

            if not self.with_short_cut:
                if self.long_backbone is None:
                    fpn_outs = short_fpn_outs
                else:
                    if self.merge_form == "add":
                        fpn_outs = [x + y for x, y in zip(short_fpn_outs, long_fpn_outs)]
                    elif self.merge_form == "concat":
                        fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                        fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                        fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    elif self.merge_form == "pure_concat":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    elif self.merge_form == "long_fusion":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    else:
                        raise Exception(f'merge_form must be in ["add", "concat"]')
            else:
                if self.long_backbone is None:
                    fpn_outs = [x + y for x, y in zip(short_fpn_outs, rurrent_pan_outs)]
                else:
                    if self.merge_form == "add":
                        fpn_outs = [x + y + z for x, y, z in zip(short_fpn_outs, long_fpn_outs, rurrent_pan_outs)]
                    elif self.merge_form == "concat":
                        fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                        fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                        fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                    elif self.merge_form == "pure_concat":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                    elif self.merge_form == "long_fusion":
                        fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                        fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                        fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                        fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                        fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                    else:
                        raise Exception(f'merge_form must be in ["add", "concat"]')

            bbox_preds_t, obj_preds_t, cls_preds_t = self.head_t(fpn_outs_t)

            if self.dil_loc == "head":
                knowledge = (bbox_preds_t, obj_preds_t, cls_preds_t)
                (
                    loss, 
                    iou_loss, 
                    conf_loss, 
                    cls_loss, 
                    l1_loss, 
                    reg_dil_loss,
                    obj_dil_loss,
                    cls_dil_loss,
                    loss_dil_hint,
                    num_fg
                ) = self.head(fpn_outs, cur_targets, cur_x, knowledge=knowledge)

                losses = {
                    "total_loss": loss,
                    # "det_loss": loss,
                    "iou_loss": iou_loss,
                    "l1_loss": l1_loss,
                    "conf_loss": conf_loss,
                    "cls_loss": cls_loss,
                    # "dil_loss": dil_loss,
                    # "neck_dil_loss":neck_dil_loss,
                    "reg_dil_loss": reg_dil_loss,
                    "cls_dil_loss": cls_dil_loss,
                    "obj_dil_loss": obj_dil_loss,
                    "loss_dil_hint":loss_dil_hint,
                    "num_fg": num_fg,
                }

                if outputs == dict():
                    outputs.update(losses)
                else:
                    for k, v in outputs.items():
                        outputs[k] += losses[k]
                loss_list.append(loss)
            else:
                if outputs == dict():
                    outputs = [self.head(fpn_outs)]
                else:
                    outputs = outputs.append(self.head(fpn_outs))

        if isinstance(outputs, list):
            return torch.cat(outputs)
        else:
            for k, v in outputs.items():
                outputs[k] /= len(branch_num_list) * 1.0

        return outputs

    def forward_test_offline(self, x, targets=None, buffer=None):
        """
        off_pipe test
        """
        outputs = dict()
        assert self.long_cfg is not None
        # 由self.speed_detector给出每个样本的最优分支
        speed_score = self.compute_speed_score(x)
        branch_num_list = speed_score.argmin(dim=1)
        N = max(branch_num_list)
        # print(f"选择第{N}个分支进行预测")

        # for idx, N in enumerate(branch_num_list):
        # import pdb; pdb.set_trace()
        # x[0] 0:3 channel 是t帧， 3:6 是t+1帧
        short_fpn_outs, rurrent_pan_outs = self.short_backbone(x[0], buffer=buffer, mode='off_pipe', backbone_neck=self.backbone)
        long_fpn_outs = self.long_backbone(x[1][:,:(N + 1) * 3,...], N + 1, buffer=buffer, mode='off_pipe', backbone_neck=self.backbone) if self.long_backbone is not None else None

        if not self.with_short_cut:
            if self.long_backbone is None:
                fpn_outs = short_fpn_outs
            else:
                if self.merge_form == "add":
                    fpn_outs = [x + y for x, y in zip(short_fpn_outs, long_fpn_outs)]
                elif self.merge_form == "concat":
                    fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                elif self.merge_form == "pure_concat":
                    fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                    fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                    fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                elif self.merge_form == "long_fusion":
                    fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                else:
                    raise Exception(f'merge_form must be in ["add", "concat"]')
        else:
            if self.long_backbone is None:
                fpn_outs = [x + y for x, y in zip(short_fpn_outs, rurrent_pan_outs)]
            else:
                if self.merge_form == "add":
                    fpn_outs = [x + y + z for x, y, z in zip(short_fpn_outs, long_fpn_outs, rurrent_pan_outs)]
                elif self.merge_form == "concat":
                    fpn_outs_2 = torch.cat([self.jian2[N](short_fpn_outs[0]), self.jian2[N](long_fpn_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([self.jian1[N](short_fpn_outs[1]), self.jian1[N](long_fpn_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([self.jian0[N](short_fpn_outs[2]), self.jian0[N](long_fpn_outs[2])], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                elif self.merge_form == "pure_concat":
                    fpn_outs_2 = torch.cat([short_fpn_outs[0], long_fpn_outs[0]], dim=1)
                    fpn_outs_1 = torch.cat([short_fpn_outs[1], long_fpn_outs[1]], dim=1)
                    fpn_outs_0 = torch.cat([short_fpn_outs[2], long_fpn_outs[2]], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                elif self.merge_form == "long_fusion":
                    fpn_outs_2 = torch.cat([short_fpn_outs[0], self.jian2[N](long_fpn_outs[0])], dim=1)
                    fpn_outs_1 = torch.cat([short_fpn_outs[1], self.jian1[N](long_fpn_outs[1])], dim=1)
                    fpn_outs_0 = torch.cat([short_fpn_outs[2], self.jian0[N](long_fpn_outs[2])], dim=1)
                    fpn_outs = (fpn_outs_2, fpn_outs_1, fpn_outs_0)
                    fpn_outs = [x + y for x, y in zip(fpn_outs, rurrent_pan_outs)]
                else:
                    raise Exception(f'merge_form must be in ["add", "concat"]')

        outputs = self.head(fpn_outs)  # [1, 11850, 13]

        return outputs

    def forward_test_online(self, x, targets=None, buffer=None):
        """
        on_pipe test (没有实现)
        """
        outputs = dict()
        fpn_outs, buffer_ = self.backbone(x,  buffer=buffer, mode='on_pipe')
        outputs = self.head(fpn_outs)
        
        return outputs, buffer_

    def _freeze_teacher_model(self):
        for name, param in self.backbone_t.named_parameters():
            param.requires_grad = False
        for name, param in self.head_t.named_parameters():
            param.requires_grad = False

    def _set_eval_teacher_model(self):
        self.backbone_t.eval()
        self.head_t.eval()

    def _get_tensors_numel(self, tensors):
        num = 0
        for t in tensors:
            num += torch.numel(t)
        return num


