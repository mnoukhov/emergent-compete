import unittest

import torch

from src.game import IteratedSenderRecver
from src.utils import circle_diff

class TestReward(unittest.TestCase):
    def setUp(self):
        self.isr = IteratedSenderRecver(batch_size=1,
                                        num_rounds=5,
                                        num_targets=100,
                                        max_bias=10)
        self.isr.reset()
        self.isr.num_targets = 100
        self.isr.send_target = torch.tensor([50.])
        self.isr.recv_target = torch.tensor([45.])

    def test_commute(self):
        pred = torch.tensor(10.)
        target = torch.tensor(90.)
        reward = self.isr._reward(pred, target)

        pred = torch.tensor(90.)
        target = torch.tensor(10.)
        other_reward = self.isr._reward(pred, target)

        self.assertEqual(reward, other_reward)

    def test_shift(self):
        pred = torch.tensor(40.)
        target = torch.tensor(60.)
        reward = self.isr._reward(pred, target)

        pred += 20
        target += 20
        other_reward = self.isr._reward(pred, target)

        self.assertEqual(reward, other_reward)

    def test_circle(self):
        pred = torch.tensor(10.)
        target = torch.tensor(90.)
        reward = self.isr._reward(pred, target).item()

        pred = torch.tensor(0.)
        target = torch.tensor(20.)
        other_reward = self.isr._reward(pred, target).item()

        self.assertAlmostEqual(reward, other_reward, delta=1e-6)

    def test_min_is_opposite(self):
        rewards = []
        idx = 10
        target = torch.tensor(idx).float()
        opposite_idx = idx + self.isr.num_targets // 2
        for i in range(self.isr.num_targets):
            pred = torch.tensor(i).float()
            reward = self.isr._reward(pred, target)
            rewards.append(reward.item())

        self.assertEqual(min(rewards), rewards[opposite_idx])

    def test_max_is_same(self):
        rewards = []
        idx = 10
        target = torch.tensor(idx).float()
        for i in range(self.isr.num_targets):
            pred = torch.tensor(i).float()
            reward = self.isr._reward(pred, target)
            rewards.append(reward.item())

        self.assertEqual(max(rewards), rewards[idx])

    def test_min(self):
        target = torch.tensor(40.)
        pred = target + self.isr.num_targets // 2
        reward = self.isr._reward(pred, target)

        self.assertEqual(reward, - self.isr.num_targets // 2)

    def test_max(self):
        target = torch.tensor(40.)
        pred = target
        reward = self.isr._reward(pred, target)

        self.assertEqual(reward, 0.)


class TestBias(unittest.TestCase):
    def setUp(self):
        self.min_bias = 0
        self.max_bias = 10
        self.isr = IteratedSenderRecver(batch_size=1,
                                        num_rounds=5,
                                        num_targets=100,
                                        min_bias=self.min_bias,
                                        max_bias=self.max_bias)
        self.isr.reset()

    def test_round_bias_constant(self):
        bias = self.isr.bias
        action = torch.tensor([0.])
        done = False
        while not done:
            _, _, done = self.isr.step(action)
            self.assertEqual(bias, self.isr.bias)

    def test_episode_bias_changes(self):
        bias = self.isr.bias
        action = torch.tensor([0.])
        done = False

        self.isr.reset()
        self.assertNotEqual(bias, self.isr.bias)

    def test_bias_range(self):
        biases = []
        action = torch.tensor([0.])
        done = False

        for _ in range(1000):
            self.isr.reset()
            biases.append(self.isr.bias.item())

        self.assertEqual(min(biases), self.min_bias)
        self.assertEqual(max(biases), self.max_bias)
        avg_bias = self.min_bias + (self.max_bias - self.min_bias) / 2
        self.assertAlmostEqual(sum(biases) / 1000, avg_bias, delta=0.1)

    def test_bias_is_diff(self):
        send_target = self.isr.send_targets[0]
        recv_target = self.isr.recv_targets[0]
        diff = circle_diff(send_target, recv_target, self.isr.num_targets)
        self.assertEqual(self.isr.bias, diff)


class TestStep(unittest.TestCase):
    def setUp(self):
        self.min_bias = 0
        self.max_bias = 10
        self.isr = IteratedSenderRecver(batch_size=1,
                                        num_rounds=5,
                                        num_targets=100,
                                        min_bias=self.min_bias,
                                        max_bias=self.max_bias)
        self.isr.reset()

    def test_reward(self):
        action = torch.tensor([20.])
        send_target = self.isr.send_targets[0]
        recv_target = self.isr.recv_targets[0]
        _, rewards, _, = self.isr.step(action)
        send_reward, recv_reward = rewards

        exp_send_reward = self.isr._reward(send_target, action)
        exp_recv_reward = self.isr._reward(recv_target, action)

        self.assertEqual(send_reward, exp_send_reward)
        self.assertEqual(recv_reward, exp_recv_reward)


if __name__ == '__main__':
    unittest.main()
