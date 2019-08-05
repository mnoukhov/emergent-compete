import gin
import torch
from torch import nn
from torch.utils.data import DataLoader


class CirclePointsIter:
    def __init__(self, num_points, bias, batch_size, num_batches):
        self.num_points = num_points
        self.bias = bias
        self.batch_size = batch_size
        self.num_batches = num_batches

        self.batches = 0

    def __next__(self):
        if self.batches >= self.num_batches:
            raise StopIteration()

        send_targets = torch.randint(self.num_points,
                                     size=(self.batch_size, 1),
                                     dtype=torch.float)
        recv_targets = (send_targets + self.bias) % self.num_points

        self.batches += 1

        return (send_targets, recv_targets)


@gin.configurable
class Circle(DataLoader):
    def __init__(self, num_points, bias, batch_size, num_batches, device='cpu'):
        self.batch_size = batch_size
        self.num_points = num_points
        self.bias = bias
        self.num_batches = num_batches

    def __iter__(self):
        return CirclePointsIter(self.num_points, self.bias, self.batch_size, self.num_batches)


class CircleLoss(nn.Module):
    def __init__(self, num_points=36):
        super().__init__()
        self.num_points = 36

    def forward(self, pred, target):
        diff = torch.abs(pred - target)
        diff[diff > self.num_points/2] = self.num_points - diff[diff > self.num_points/2]
        return diff

