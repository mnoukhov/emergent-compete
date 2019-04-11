import unittest
from unittest.mock import Mock

import torch

from src.agents import PolicyGradient

class TestPolicyGradient(unittest.TestCase):
    def setUp(self):
        self.policy = PolicyGradient(n=100, lr=100, gamma=0.99,
                                     std=3, mode=0)

    def test_action_std0(self):
        policy_action = 0.5
        self.policy.std = 0
        self.policy.policy = Mock(return_value=policy_action)

        action = self.policy.action([1,2,3])
        exp_action = policy_action * self.policy.n
        self.assertEqual(exp_action, action.item())

    def test_update_loss_basic(self):
        self.policy.gamma = 1.0
        self.policy.rewards = [torch.full((3,1),10, requires_grad=True)
                               for _ in range(5)]
        self.policy.log_probs = [torch.ones(3,1, requires_grad=True)
                                 for _ in range(5)]
        self.policy.update()

        self.assertEqual(self.policy.losses[0].item(), 30.)
