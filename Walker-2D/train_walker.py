import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make("Walker2d-v4")

# Create the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=1_000_000)

# Save the trained model
model.save("ppo_walker2d")

# Optional: Evaluate it
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()
