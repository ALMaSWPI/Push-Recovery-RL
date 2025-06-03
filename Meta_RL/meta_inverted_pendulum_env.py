import gym
import numpy as np

class MetaInvertedPendulumEnv:
    def __init__(self, pole_length_range=(0.5, 2.0)):
        self.pole_length_range = pole_length_range

    def sample_task(self):
        pole_length = np.random.uniform(*self.pole_length_range)
        return {"pole_length": pole_length}

    def make_env(self, task):
        env = gym.make("InvertedPendulum-v2")
        env.env.length = task["pole_length"]
        return env
