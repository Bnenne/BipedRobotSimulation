import mujoco
import mujoco.viewer
import numpy as np
import random as rd

XML = r"""
<mujoco>
    <worldbody>
        <body name="robot" pos="0 0 0.1">
            <joint name="slide_x" type="slide" axis="1 0 0"/>
            <joint name="slide_y" type="slide" axis="0 1 0"/>
        
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        
        <body name="target" pos="0 0 0">
            <geom type="sphere" size="0.1" rgba="1 0 0 1" contype="0" conaffinity="0"/>
        </body>
        
        <geom type="plane" size="25 25 0.1"/>
    </worldbody>
  
    <actuator>
        <motor name="motor_x" joint="slide_x" ctrlrange="-1 1"/>
        <motor name="motor_y" joint="slide_y" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""

def at_target(target, current_pos, current_vel):
    distance = np.linalg.norm(target[:2] - current_pos[:2])
    velocity = np.linalg.norm(current_vel)

    return distance <= 0.1 and velocity <= 0.1

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")

target = np.array([
    rd.randint(-25, 25),
    rd.randint(-25, 25),
    0
], dtype=float)
model.body_pos[target_id] = target
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -30

    while viewer.is_running():
        mujoco.mj_step(model, data)

        robot_pos = data.xpos[robot_id]

        viewer.cam.lookat[:] = robot_pos

        if at_target(target, data.xpos[robot_id], data.cvel[robot_id][3:]):
            target = np.array([
                rd.randint(-25, 25),
                rd.randint(-25, 25),
                0
            ], dtype=float)

            model.body_pos[target_id] = target
            mujoco.mj_forward(model, data)

        if target[0] > robot_pos[0]:
            data.ctrl[0] = 0.01
        elif target[0] < robot_pos[0]:
            data.ctrl[0] = -0.01
        else:
            data.ctrl[0] = 0

        if target[1] > robot_pos[1]:
            data.ctrl[1] = 0.01
        elif target[1] < robot_pos[1]:
            data.ctrl[1] = -0.01
        else:
            data.ctrl[1] = 0

        print(
            "Target:",
            target,
            "Robot:",
            np.round(robot_pos, 1)
        )

        viewer.sync()