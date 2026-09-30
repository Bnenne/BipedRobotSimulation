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
        
        <geom name="target" type="sphere" size="0.1" pos="0 0 0" rgba="1 0 0 1" contype="0" conaffinity="0"/>
        
        <geom type="plane" size="50 50 0.1"/>
    </worldbody>
  
    <actuator>
        <motor name="motor_x" joint="slide_x" ctrlrange="-1 1"/>
        <motor name="motor_y" joint="slide_y" ctrlrange="-1 1"/>
    </actuator>
</mujoco>
"""

def at_target(target, current_pos, current_vel):
    distance = np.linalg.norm(target - current_pos)
    velocity = np.linalg.norm(current_vel)

    return distance <= 0.1

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

target = np.array([rd.randint(-50, 50), rd.randint(-50, 50), 0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        if at_target(target, data.geom_xpos[2], data.cvel[1][3:]):
            target = np.array([rd.randint(-50, 50), rd.randint(-50, 50), rd.randint(-50, 50)])

        if target[0] > data.geom_xpos[2][0]:
            data.ctrl[0] = 0.01
        elif target[0] < data.geom_xpos[2][0]:
            data.ctrl[0] = -0.01
        else:
            data.ctrl[0] = 0

        if target[1] > data.geom_xpos[2][1]:
            data.ctrl[1] = 0.01
        elif target[1] < data.geom_xpos[2][1]:
            data.ctrl[1] = -0.01
        else:
            data.ctrl[1] = 0

        print(target, np.round(data.geom_xpos[2], decimals=1))

        viewer.sync()