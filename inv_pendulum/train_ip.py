from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from pendulum_env import InvertedPendulumEnvCustom

# Create environment
env = InvertedPendulumEnvCustom(pole_length=1.0, pole_mass=0.1)
check_env(env, warn=True)

# Create PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=35_000)

# Save the model
model.save("ppo_inverted_pendulum_custom2")

# Optional: Watch the trained agent
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()

# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
# from pendulum_env import InvertedPendulumEnv

# env_fn = lambda: InvertedPendulumEnv(render_mode="rgb_array")
# env = DummyVecEnv([env_fn])

# env = VecVideoRecorder(
#     env,
#     video_folder="./videos/",
#     record_video_trigger=lambda x: x % 10000 == 0,  # record every 10k steps
#     video_length=500,
#     name_prefix="inverted_pendulum"
# )

# # 2. Train model
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=80000)

# # 3. Save final video at the end
# obs = env.reset()
# for _ in range(500):
#     action, _ = model.predict(obs)
#     obs, _, dones, _ = env.step(action)
# env.close()

