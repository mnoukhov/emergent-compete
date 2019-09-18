import torch

from src.agents import Policy

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.policy = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size))

    def forward(self, input_):
        return self.policy(input_)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, output_size):
        self.policy = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size))

    def forward(self, state, action):
        return self.policy(torch.cat((state,action), dim=1))


@gin.configurable
class DDPG(Policy):
    def __init__(self, input_size, output_size, hidden_size,
                 actor_lr, critic_lr, ent_reg, tau, warmup_episodes, **kwargs):
        super().__init__(**kwargs)
        self.actor = Actor(input_size, hidden_size, output_size)
        self.actor_target = Actor(input_size, hidden_size, output_size)

        self.critic = Critic(7,1).to(device)
        self.critic_target = deepcopy(self.critic).to(device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = ReplayBuffer()
        self.noise = Normal(0, scale=0.1)

        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.warmup_episodes = warmup_episodes

    def action(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            action = self.actor(state).squeeze().cpu()

        if self.training:
            batch_size = state.shape[0]
            action += self.noise.sample(sample_shape=(batch_size,))

        return action % self.num_actions

    def update(self, ep, rewards, log, **kwargs):
        _, logs = super().update(ep, rewards, log)
        if ep < self.warmup_episodes:
            return

        # hardcoded for recver
        state, _, action, _, reward, next_state = self.memory.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)

        current_Q = self.critic(state, action)
        next_action = self.actor_target(next_state).squeeze()
        next_Q = self.critic_target(next_state, next_action)
        target_Q = reward.unsqueeze(1) + self.gamma * next_Q
        critic_loss = F.mse_loss(current_Q, target_Q)

        current_action = self.actor(state).squeeze()
        critic_reward = self.critic(state, current_action)
        actor_loss = -critic_reward.mean()

        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.actor_target, self.tau)

        logs['loss'] = actor_loss.item() + critic_loss.item()

        return actor_loss + critic_loss, logs


    def soft_update(src, trg, tau):
        for trg_param, src_param in zip(trg.parameters(), src.parameters()):
            trg_param.data.copy_((1 - tau) * trg_param.data + tau * src_param.data)

