import os
import imageio
import numpy as np
from stable_baselines3 import PPO
from pendulum_env import InvertedPendulumEnv
from pendulum_env import InvertedPendulumEnvCustom
import time

# Make output folder if needed
video_dir = "./recordings/"
os.makedirs(video_dir, exist_ok=True)

# New pendulum parameters
test_length = 1  # 1.5 meters long instead of 1.0
test_mass = 0.1   # heavier

# Load trained model and environment
env = InvertedPendulumEnvCustom(render_mode="rgb_array",pole_length=test_length, pole_mass=test_mass)  # must support 'rgb_array'
#env = InvertedPendulumEnvCustom(render_mode="rgb_array", pole_length=1.5, pole_mass=0.2)
model = PPO.load("ppo_inverted_pendulum_custom", env=env)


frames = []
obs, _ = env.reset()

# Run for N steps and record frames
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)

    # Get RGB frame from env
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    if done or truncated:
        obs, _ = env.reset()

env.close()

# Save video
output_path = os.path.join(video_dir, "inverted_pendulum_run_custom35k.mp4")
imageio.mimwrite(output_path, frames, fps=30, codec="libx264")

print(f"🎥 Video saved to: {output_path}")
