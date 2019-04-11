import unittest

import torch

from src.game import IteratedSenderRecver

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

    # def test_reward(self):
        # action = torch.tensor([40.])
        # send_target = self.isr.send_target
        # recv_target = self.isr.recv_target
        # _, rewards, _ = self.isr.step(action)

        # exp_send_reward = self.isr.dist_to_reward(send_target - action)
        # exp_recv_reward = self.isr.dist_to_reward(recv_target - action)

        # send_reward, recv_reward = rewards
        # self.assertEqual(send_reward, exp_send_reward)
        # self.assertEqual(recv_reward, exp_recv_reward)


if __name__ == '__main__':
    unittest.main()
