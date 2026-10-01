import gymnasium as gym
import numpy as np
import mujoco

class WorldEnv(gym.Env):
    def __init__(self, XML: str):
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

    def reset(self):
        pass
        # TODO: create reset
        

    def step(self, action):
        x_command = action[0]
        y_command = action[1]

        self.data.ctrl[0] = x_command
        self.data.ctrl[1] = y_command

        mujoco.mj_step(self.model, self.data)

        # TODO: I don't know if this needs more or not
