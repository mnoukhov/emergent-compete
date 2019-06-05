from collections import namedtuple, deque
import random

import gin
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


Experience = namedtuple('Experience', ('state', 'send_action', 'recv_action',
                                       'send_reward', 'recv_reward', 'next_state'))

#TODO change to deque?
@gin.configurable
class ReplayBuffer(Dataset):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]

    def push(self, state, send_action, recv_action,
             send_reward, recv_reward, next_state):
        batch_size = state.size(0)
        for i in range(batch_size):
            exp = Experience(state[i], send_action[i], recv_action[i],
                             send_reward[i], recv_reward[i], next_state[i])
            self.memory.append(exp)

    def sample(self, num_samples):
        indices = torch.randint(len(self.memory), size=(num_samples,), dtype=torch.int64).tolist()
        batch = default_collate([self.memory[i] for i in indices])
        return batch
