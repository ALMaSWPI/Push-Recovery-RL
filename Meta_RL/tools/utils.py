import torch
from torch.distributions import MultivariateNormal


def collect_trajectories(env, policy, num_steps, device):
    obs = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    state = torch.tensor(env.reset(), dtype=torch.float32).to(device)
    # Inside collect_trajectories
    for _ in range(num_steps):
        mean, std, value = policy(state.unsqueeze(0))
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, _ = env.step(action.cpu().numpy())

        obs.append(state)
        actions.append(action.squeeze(0))
        rewards.append(torch.tensor(reward, dtype=torch.float32))
        dones.append(torch.tensor(done, dtype=torch.float32))
        log_probs.append(log_prob)
        values.append(value.squeeze(0))

        state = torch.tensor(next_state, dtype=torch.float32).to(device)
        if done:
            state = torch.tensor(env.reset(), dtype=torch.float32).to(device)


    return (
        torch.stack(obs),
        torch.stack(actions),
        torch.tensor(rewards, dtype=torch.float32).to(device),
        torch.tensor(dones, dtype=torch.float32).to(device),
        torch.stack(log_probs),
        torch.stack(values)
    )

def compute_gae(rewards, dones, values, gamma, tau):
    rewards = rewards.detach()
    dones = dones.detach()
    values = values.detach().squeeze(-1)  # ✅ Make sure values is [T]

    advantages = torch.zeros_like(rewards)
    gae = 0.0

    # Append zero for bootstrap
    values = torch.cat([values, torch.tensor([0.0], device=values.device)])

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * tau * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return returns, advantages
