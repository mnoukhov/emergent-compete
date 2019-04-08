import unittest

import torch

from src.game import IteratedSenderRecver

class TestStep(unittest.TestCase):
    def setUp(self):
        self.isr = IteratedSenderRecver(batch_size=1,
                                        num_rounds=5,
                                        num_targets=100,
                                        max_bias=10)
        self.isr.reset()
        self.isr.send_target = torch.tensor([50])
        self.isr.recv_target = torch.tensor([45])

    def test_dist_regular(self):
        pred = torch.tensor(40)
        target = torch.tensor(60)
        dist = self.isr._dist(pred, target)
        self.assertEqual(dist, target - pred)

    def test_dist_circle(self):
        pred = torch.tensor(10)
        target = torch.tensor(90)
        dist = self.isr._dist(pred, target)
        self.assertEqual(dist, 20)

    def test_dist_to_reward(self):
        self.isr.num_targets = 300
        dist = 30
        exp_reward = -30 / 150
        self.assertEqual(self.isr.dist_to_reward(dist), exp_reward)

    def test_min_reward(self):
        action = self.isr.send_target + self.isr.num_targets/2
        _, rewards, _ = self.isr.step(action)
        send_reward, _ = rewards
        self.assertEqual(send_reward.item(), -1)

    def test_max_reward(self):
        action = self.isr.send_target
        _, rewards, _ = self.isr.step(action)

        send_reward, _ = rewards
        self.assertEqual(send_reward.item(), 0)

    def test_reward(self):
        action = torch.tensor([40])
        send_target = self.isr.send_target
        recv_target = self.isr.recv_target
        _, rewards, _ = self.isr.step(action)

        exp_send_reward = self.isr.dist_to_reward(send_target - action)
        exp_recv_reward = self.isr.dist_to_reward(recv_target - action)

        send_reward, recv_reward = rewards
        self.assertEqual(send_reward, exp_send_reward)
        self.assertEqual(recv_reward, exp_recv_reward)


if __name__ == '__main__':
    unittest.main()
