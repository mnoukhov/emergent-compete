import math

import gin
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss


class CirclePointsIter:
    def __init__(self, num_points, bias, batch_size, num_batches, num_rounds,
                 device, training):
        self.num_points = num_points
        self.bias = bias
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_rounds = num_rounds
        self.device = device
        self.training = training

        self.batches = 0
        self.test_send_targets = self.test_targets()

    def test_targets(self):
        rounds = [torch.arange(0, self.num_points,
                               step=self.num_points / self.batch_size,
                               device=self.device).unsqueeze(1)]

        for _ in range(self.num_rounds - 1):
            rounds.append(rounds[-1].clone() + self.num_points / self.num_rounds)

        return torch.stack(rounds, dim=0)

    def __next__(self):
        if self.batches >= self.num_batches:
            raise StopIteration()

        if self.training:
            send_targets = self.num_points * torch.rand(size=(self.num_rounds, self.batch_size, 1),
                                                        device=self.device)
        else:
            send_targets = self.test_send_targets.clone()

        recv_targets = (send_targets + self.bias) % self.num_points

        self.batches += 1


        return (send_targets, recv_targets)


@gin.configurable
class Game(DataLoader):
    def __init__(self, num_points, bias, batch_size, num_batches, num_rounds,
                 device='cpu', training=True):
        self.batch_size = batch_size
        self.num_points = num_points
        self.bias = bias
        self.num_batches = num_batches
        self.num_rounds = num_rounds
        self.device = device
        self.training = training

    def __iter__(self):
        return CirclePointsIter(self.num_points, self.bias, self.batch_size,
                                self.num_batches, self.num_rounds, self.device,
                                self.training)


@gin.configurable
class CircleL1(_Loss):
    def __init__(self, num_points):
        super().__init__(reduction=None)
        self.num_points = num_points

    def forward(self, output, target):
        # torch.remainder has an issue
        pred = torch.abs(torch.fmod(output, self.num_points))
        diff = torch.abs(pred - target)
        counter_diff = self.num_points - diff
        min_diff = torch.min(diff, counter_diff)
        return min_diff


@gin.configurable
class CircleL2(_Loss):
    def __init__(self, num_points):
        super().__init__(reduction=None)
        self.num_points = num_points

    def forward(self, output, target):
        pred = torch.abs(torch.fmod(output, self.num_points))
        diff = (pred - target)**2
        counter_diff = (self.num_points - torch.abs(pred - target))**2
        min_diff = torch.min(diff, counter_diff)
        return min_diff * 2 / self.num_points


@gin.configurable
class CosineLoss(_Loss):
    def __init__(self, num_points):
        super().__init__(reduction=None)
        self.num_points = num_points

    def forward(self, output, target):
        output_theta = output * 2 * math.pi / self.num_points
        target_theta = target * 2 * math.pi / self.num_points
        cosine = torch.cos(output_theta - target_theta)
        return (1 - cosine) * self.num_points / 4
