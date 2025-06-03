import os
os.environ['MUJOCO_GL'] = 'osmesa'  # headless rendering

import time
import torch
import numpy as np
import imageio
from meta_inverted_pendulum_env import MetaInvertedPendulumEnv
from meta_ppo import MetaPPo
from ac_network import ActorCritic
from torch.distributions import MultivariateNormal

# Load trained meta-policy
meta_policy = ActorCritic(in_dim=4, out_dim=1, continuous=True)
meta_policy.load_state_dict(torch.load("meta_policy_iter_10.pth"))
meta_policy.eval()

# Meta agent
agent = MetaPPo(meta_policy, continuous=True)

# Create new test task
test_task = {"pole_length": 1.75}
env = MetaInvertedPendulumEnv().make_env(test_task)

# Adapt meta-policy to task
adapted_policy = agent.clone_policy()
agent.inner_update(env, adapted_policy, num_steps=1000)

# Run adapted policy and record video
obs = env.reset()
done = False

frames = []
max_steps = 1000000  # in case the episode is too long

for _ in range(max_steps):
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        mean, std, _ = adapted_policy(obs_tensor)
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        action = dist.mean.cpu().numpy().squeeze()
    obs, _, done, _ = env.step(action)
    
    frame = env.render(mode='rgb_array')
    frames.append(frame)

    if done:
        break

env.close()

# Save video using imageio
video_path = "adapted_policy_demo_1M.mp4"
imageio.mimsave(video_path, frames, fps=30)
print(f"✅ Saved video: {video_path}")
