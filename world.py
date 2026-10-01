from typing import Optional
import gymnasium as gym
import numpy as np
import random as rd
import mujoco.viewer

class WorldEnv(gym.Env):
    def __init__(self, XML: bytes):
        super().__init__()

        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)

        # agent_x, agent_y, agent_vel_x, agent_vel_y, target_x, target_y
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32,
        )

        # slide_x [-1, 1] slide_y [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.agent_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "agent",
        )

        self.target_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target",
        )

        self.max_steps = 1000
        self.current_step = 0

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.cam.distance = 5
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -30

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.model.body_pos[self.agent_id] = np.array([0, 0, 0.1])

        self.model.body_pos[self.target_id] = np.array([
            rd.randint(-25, 25),
            rd.randint(-25, 25),
            0
        ], dtype=float)

        self.current_step = 0

        observation = self._get_obs()

        return observation

    def _get_obs(self):
        agent_pos = self.data.xpos[self.agent_id][:2]
        agent_vel = self.data.cvel[self.agent_id][3:5]
        target_pos = self.data.xpos[self.target_id][:2]

        return np.array(agent_pos + agent_vel + target_pos)

    def _get_distance(self):
        return np.linalg.norm(self.data.xpos[self.target_id][:2] - self.data.xpos[self.agent_id][:2])

    def _at_target(self):
        distance = np.linalg.norm(self.data.xpos[self.target_id][:2] - self.data.xpos[self.agent_id][:2])
        velocity = np.linalg.norm(self.data.cvel[self.agent_id][3:5])

        return distance <= 0.1 and velocity <= 0.1

    def step(self, action):
        self.current_step += 1

        self.viewer.cam.lookat[:] = self.data.xpos[self.agent_id]

        x_command = action[0]
        y_command = action[1]

        self.data.ctrl[0] = x_command
        self.data.ctrl[1] = y_command

        distance = self._get_distance()

        mujoco.mj_step(self.model, self.data)

        reward = distance - self._get_distance()
        reward += 5 if self._at_target() else 0

        terminated = self.current_step == self.max_steps

        observation = self._get_obs()

        self.viewer.sync()

        return observation, reward, terminated

    def get_viewer(self):
        return self.viewer
