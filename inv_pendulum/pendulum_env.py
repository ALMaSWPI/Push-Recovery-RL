# import gymnasium as gym
# import numpy as np
# import mujoco
# from mujoco import viewer
# from gymnasium import spaces
# from mujoco import MjRenderer


# class InvertedPendulumEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

#     def __init__(self, render_mode=None):
#         self.model = mujoco.MjModel.from_xml_path("inverted_pendulum.xml")
#         self.data = mujoco.MjData(self.model)

#         self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

#         self.render_mode = render_mode
#         self.viewer = None
#         self._frame = None  # store rendered frame if needed

#         self.ctrl_range = self.model.actuator_ctrlrange[0]


#     def step(self, action):
#         action = np.clip(action, -1.0, 1.0)
#         self.data.ctrl[:] = action * self.ctrl_range[1]  # scale to actual control range

#         for _ in range(5):  # simulate 5 steps to make motion smoother
#             mujoco.mj_step(self.model, self.data)

#         obs = self._get_obs()
#         angle = obs[1]
#         ang_vel = obs[3]
#         reward = 1.0 - 0.1 * (angle ** 2 + 0.01 * ang_vel ** 2)
#         done = bool(abs(angle) > 0.5)  # end episode if pendulum falls too much

#         return obs, reward, done, False, {}

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         qpos = np.random.uniform(low=-0.01, high=0.01, size=self.model.nq)
#         qvel = np.random.uniform(low=-0.01, high=0.01, size=self.model.nv)
#         self.data.qpos[:] = qpos
#         self.data.qvel[:] = qvel
#         mujoco.mj_forward(self.model, self.data)
#         return self._get_obs(), {}  # <- must return (obs, info)


#     def _get_obs(self):
#         return np.concatenate([self.data.qpos, self.data.qvel])

#     def render(self):
#         if self.render_mode == "rgb_array":
#             if self.viewer is None:
#                 self.viewer = MjRenderer(self.model)
#             self.viewer.update_scene(self.data)
#             frame = self.viewer.render()
#             return frame  # shape (H, W, 3), dtype=np.uint8

#         elif self.render_mode == "human":
#             # Optional: real-time viewer
#             if self.viewer is None:
#                 from mujoco.viewer import launch_passive
#                 self.viewer = launch_passive(self.model, self.data)
#             self.viewer.sync()


#     def close(self):
#         if hasattr(self, "viewer") and self.viewer is not None:
#             self.viewer.close()
#             self.viewer = None
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class InvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path("inverted_pendulum.xml")
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        self.ctrl_range = self.model.actuator_ctrlrange[0]
        self.render_mode = render_mode
        self.viewer = None  # used only in human mode
        self.renderer = None  # used for rgb_array

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action * self.ctrl_range[1]

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        angle = obs[1]
        ang_vel = obs[3]
        reward = 1.0 - 0.1 * (angle ** 2 + 0.01 * ang_vel ** 2)
        terminated = bool(abs(angle) > 0.5)

        return obs, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        qpos = np.random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = np.random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                from mujoco import Renderer
                self.renderer = Renderer(self.model)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        elif self.render_mode == "human":
            from mujoco.viewer import launch_passive
            if self.viewer is None:
                self.viewer = launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

class InvertedPendulumEnvCustom(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, pole_length=1.0, pole_mass=0.1):
        # Load model
        self.model = mujoco.MjModel.from_xml_path("inverted_pendulum.xml")
        self.data = mujoco.MjData(self.model)

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        # Control range
        self.ctrl_range = self.model.actuator_ctrlrange[0]

        # Rendering
        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

        # --- Set pendulum custom params ---
        self._set_pendulum_params(pole_length, pole_mass)

    def _set_pendulum_params(self, pole_length, pole_mass):
        # Get IDs
        pole_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pole_geom")
        pole_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pole")

        # Set geom size: [radius, half-length]
        self.model.geom_size[pole_geom_id][0] = 0.05  # radius stays 0.05
        self.model.geom_size[pole_geom_id][1] = 0.5 * pole_length  # half-length

        # Set mass
        self.model.body_mass[pole_body_id] = pole_mass

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomize initial state
        qpos = np.random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = np.random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action * self.ctrl_range[1]

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        angle = obs[1]
        ang_vel = obs[3]

        # Reward: encourage uprightness and low velocity
        reward = 1.0 - 0.1 * (angle ** 2 + 0.01 * ang_vel ** 2)

        # Terminate if fallen too far
        terminated = bool(abs(angle) > 0.5)
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        elif self.render_mode == "human":
            if self.viewer is None:
                from mujoco.viewer import launch_passive
                self.viewer = launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

