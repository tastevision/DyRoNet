#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.short_next_input, self.long_next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.short_next_input = None
            self.long_next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = (self.next_target[0].cuda(non_blocking=True), self.next_target[1].cuda(non_blocking=True))
            # self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        short_input = self.short_next_input
        long_input = self.long_next_input
        target = self.next_target
        if short_input is not None:
            self.record_stream(short_input)
        if long_input is not None:
            self.record_stream(long_input)
        if target is not None:
            target[0].record_stream(torch.cuda.current_stream())
            target[1].record_stream(torch.cuda.current_stream())
            # target.record_stream(torch.cuda.current_stream())
        self.preload()
        return (short_input, long_input), (target[0], target[1])
        # return input, target

    def _input_cuda_for_image(self):
        self.short_next_input = self.short_next_input.cuda(non_blocking=True)
        self.long_next_input = self.long_next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())
