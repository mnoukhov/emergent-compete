import math

import gin
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss


class CirclePointsIter:
    def __init__(self, num_points, bias, batch_size, num_batches, device):
        self.num_points = num_points
        self.bias = bias
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device

        self.batches = 0

    def __next__(self):
        if self.batches >= self.num_batches:
            raise StopIteration()

        send_targets = torch.randint(self.num_points,
                                     size=(self.batch_size, 1),
                                     device=self.device,
                                     dtype=torch.float)
        recv_targets = (send_targets + self.bias) % self.num_points

        self.batches += 1

        return (send_targets, recv_targets)


@gin.configurable
class Game(DataLoader):
    def __init__(self, num_points, bias, batch_size, num_batches, device='cpu'):
        self.batch_size = batch_size
        self.num_points = num_points
        self.bias = bias
        self.num_batches = num_batches
        self.device = device

    def __iter__(self):
        return CirclePointsIter(self.num_points, self.bias, self.batch_size, self.num_batches,
                                self.device)


@gin.configurable
class CircleLoss(_Loss):
    def __init__(self, num_points):
        super().__init__(reduction=None)
        self.num_points = num_points

    def forward(self, output, target):
        # angle = 2 * math.pi * torch.abs(pred - target) / self.num_points
        # return torch.cos(angle)
        # diff[diff > self.num_points/2] = self.num_points - diff[diff > self.num_points/2]
        pred = output %  self.num_points
        diff = torch.abs(pred - target)
        counter_diff = self.num_points - diff
        min_diff = torch.min(diff, counter_diff)
        return min_diff

