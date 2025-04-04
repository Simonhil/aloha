
"""ment to visualise the current mujoco environment"""


import cv2
import time
import sys
import IPython
import mujoco.renderer
import torch


e = IPython.embed
from constants import MASTER2PUPPET_JOINT_FN, DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE
#from robot_utils import torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions

from pathlib import Path
from typing import Optional, Sequence
from data_collection.config import BaseConfig as bc
import mujoco
import mujoco.viewer
import numpy as np
#from loop_rate_limiters import RateLimiter


import mink

_HERE = Path(__file__).parent.parent
_XML = _HERE / 'mujoco_assets' / "box_transfer.xml"

# Single arm joint names.
_JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

# Single arm velocity limits, taken from:
# https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
_VELOCITY_LIMITS = {k: np.pi for k in _JOINT_NAMES}


model = mujoco.MjModel.from_xml_path(str(_XML))
data = mujoco.MjData(model)




# Bodies for which to apply gravity compensation.
left_subtree_id = model.body("left/base_link").id
right_subtree_id = model.body("right/base_link").id

# Get the dof and actuator ids for the joints we wish to control.
joint_names: list[str] = []
velocity_limits: dict[str, float] = {}

#joint_names.append("box_joint")
#velocity_limits["box_joint"] = 0

for prefix in ["left", "right"]:
    for n in _JOINT_NAMES:
        name = f"{prefix}/{n}"
        joint_names.append(name)
        velocity_limits[name] = _VELOCITY_LIMITS[n]
dof_ids = np.array([model.joint(name).id for name in joint_names])
actuator_ids = np.array([model.actuator(name).id for name in joint_names])

configuration = mink.Configuration(model)

tasks = [
    l_ee_task := mink.FrameTask(
        frame_name="left/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    ),
    r_ee_task := mink.FrameTask(
        frame_name="right/gripper",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    ),
    posture_task := mink.PostureTask(model, cost=1e-4),
]


# Create a renderer
width, height = 250, 250  # Image resolution


l_mid = model.body("left/target").mocapid[0]
r_mid = model.body("right/target").mocapid[0]

mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")



# Launch the MuJoCo viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    #viewer.cam.fixedcamid = 4  # Use the first camera (change index as needed)
    #viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # Use a fixed camera
    renderer = mujoco.Renderer(model, bc.IMAGE_WIDTH,bc.IMAGE_HIGHT)
    while viewer.is_running():

        mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")
        #ee pos
        l_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "left/target"))
        r_ee_task.set_target(mink.SE3.from_mocap_name(model, data, "right/target"))

        #get simulated img
        camera_name = "wrist_cam_left"
        renderer.update_scene(data, camera=camera_name)
        img = renderer.render()

        
        #imageio.imwrite(f"{camera_name}.png", img)
        # Save the image
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = img_bgr[100:,:,:]
        img_bgr=cv2.resize(img_bgr, (bc.END_WIDTH, bc.END_HIGHT))
       
        cv2.imwrite(f"{camera_name}.png", img_bgr)

        # Create an offscreen renderer
          # Set resolution

        # List of camera names to capture
        #camera_names = ["wrist_cam_left","wrist_cam_right"]

        # Capture and save images
        # for camera_name in camera_names:


           

        #     #imageio.imwrite(f"{camera_name}.png", img)
        #     # Save the image
        #     # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        #     # img_bgr = img_bgr[100:200, 100:200, :]
        #     # img_bgr=cv2.resize(img_bgr, (width, height))
        #     # cv2.imwrite(camera_name +"mujoco_camera_image.png", img_bgr)
        #     left_frame_id = model.site("left/gripper").id
        #     print(data.subtree_linvel[left_frame_id])
        mujoco.mj_step(model, data)  # Step the simulation
        viewer.sync()
        time.sleep(0.01)  # Control the simulation speed
