import torch
import torch.nn.functional as F
import copy
from tools.utils import collect_trajectories, compute_gae

class MetaPPo:
    def __init__(self, meta_policy, env=None, inner_lr=0.1, outer_lr=0.001, gamma=0.99, tau=0.95, continuous=True):
        self.meta_policy = meta_policy
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.gamma = gamma
        self.tau = tau
        self.continuous = continuous
        self.device = torch.device("cpu")
        self.optimizer = torch.optim.Adam(self.meta_policy.parameters(), lr=outer_lr)

    def clone_policy(self):
        return copy.deepcopy(self.meta_policy)

    def inner_update(self, env, policy, num_steps=1000):
        obs, actions, rewards, dones, log_probs, values = collect_trajectories(env, policy, num_steps, self.device)
        returns, advantages = compute_gae(rewards, dones, values, self.gamma, self.tau)
        returns, advantages = map(lambda x: x.detach(), (returns, advantages))

        # Do 1 PPO update
        new_log_probs, new_values, entropy = policy.evaluate_actions(obs, actions)
        ratio = (new_log_probs - log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns)
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

        grads = torch.autograd.grad(loss, policy.parameters(), create_graph=True)
        with torch.no_grad():
            for p, g in zip(policy.parameters(), grads):
                p.data -= self.inner_lr * g

    def compute_meta_loss(self, env, adapted_policy, num_steps=1000):
        obs, actions, rewards, dones, log_probs, values = collect_trajectories(env, adapted_policy, num_steps, self.device)
        returns, advantages = compute_gae(rewards, dones, values, self.gamma, self.tau)

        # Meta loss is computed w.r.t. original meta policy
        log_probs_meta, values_meta, entropy_meta = self.meta_policy.evaluate_actions(obs, actions)
        policy_loss = -(log_probs_meta * advantages.detach()).mean()
        value_loss = F.mse_loss(values_meta, returns.detach())
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_meta.mean()

        grads = torch.autograd.grad(loss, self.meta_policy.parameters())
        return loss.item(), grads

    def outer_update(self, grads_list):
        # Average gradients
        with torch.no_grad():
            for i, param in enumerate(self.meta_policy.parameters()):
                grad = torch.stack([grads[i] for grads in grads_list]).mean(dim=0)
                param.grad = grad
        self.optimizer.step()
        self.optimizer.zero_grad()
