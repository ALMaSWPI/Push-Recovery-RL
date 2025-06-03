import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical

class ActorCritic(nn.Module):
    def __init__(self, in_dim, out_dim, continuous=True, layer_norm=True):
        super(ActorCritic, self).__init__()

        self.continuous = continuous

        # Actor network
        self.actor_fc1 = nn.Linear(in_dim, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_fc3 = nn.Linear(64, out_dim)

        # Learnable log_std for diagonal covariance (continuous only)
        if self.continuous:
            self.log_std = nn.Parameter(torch.zeros(out_dim))
        else:
            self.softmax = nn.Softmax(dim=-1)

        # Critic network
        self.critic_fc1 = nn.Linear(in_dim, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_fc3 = nn.Linear(64, 1)

        # Optional weight normalization
        if layer_norm:
            self.normalization(self.actor_fc1, std=1.0)
            self.normalization(self.actor_fc2, std=1.0)
            self.normalization(self.actor_fc3, std=1.0)
            self.normalization(self.critic_fc1, std=1.0)
            self.normalization(self.critic_fc2, std=1.0)
            self.normalization(self.critic_fc3, std=1.0)

    @staticmethod
    def normalization(layer, std=1.0, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)

    def _forward_actor(self, states):
        x = F.relu(self.actor_fc1(states))
        x = F.relu(self.actor_fc2(x))
        x = self.actor_fc3(x)
        if self.continuous:
            mean = x
            std = self.log_std.exp().expand_as(mean)
            return mean, std
        else:
            return self.softmax(x)

    def _forward_critic(self, states):
        x = F.relu(self.critic_fc1(states))
        x = F.relu(self.critic_fc2(x))
        return self.critic_fc3(x)

    def forward(self, states):
        if self.continuous:
            mean, std = self._forward_actor(states)
            value = self._forward_critic(states)
            return mean, std, value
        else:
            probs = self._forward_actor(states)
            value = self._forward_critic(states)
            return probs, value

    def sample_action(self, mean, std):
        if self.continuous:
            dist = MultivariateNormal(mean, torch.diag_embed(std))
        else:
            dist = Categorical(mean)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob

    def get_logproba(self, states, actions):
        if self.continuous:
            mean, std = self._forward_actor(states)
            dist = MultivariateNormal(mean, torch.diag_embed(std))
        else:
            probs = self._forward_actor(states)
            dist = Categorical(probs)
        return dist.log_prob(actions)

    def evaluate_actions(self, states, actions):
        logprobs = self.get_logproba(states, actions)
        values = self._forward_critic(states).squeeze(-1)
        entropy = -torch.mean(torch.exp(logprobs) * logprobs)
        return logprobs, values, entropy
