import threading
import time
import sys
import IPython
import cv2
import torch
e = IPython.embed

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from aloha_scripts.constants import MASTER2PUPPET_JOINT_FN, DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE
from aloha_scripts.robot_utils import torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions
from data_collection.config import BaseConfig as bc
from pathlib import Path
from typing import Optional, Sequence
from scipy.spatial.transform import Rotation as R







def press_to_start(master_bot):
    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot.dxl.robot_torque_enable("single", "gripper", False)
    print(f'Close the gripper to start')
    close_thresh = -0.3
    pressed = False
    while not pressed:
        gripper_pos = get_arm_gripper_positions(master_bot)
        if gripper_pos < close_thresh:
            pressed = True
        time.sleep(DT/10)
    torque_off(master_bot)
    print(f'Started!')


def prep_robots(master_bot, puppet_bot, master_only):
    # reboot gripper motors, and set operating modes for all motors
    start_arm_qpos = START_ARM_POSE[:6]
    if not master_only:
        puppet_bot.dxl.robot_reboot_motors("single", "gripper", True)
        puppet_bot.dxl.robot_set_operating_modes("group", "arm", "position")
        puppet_bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
        torque_on(puppet_bot)
        move_arms([puppet_bot], [start_arm_qpos] * 2, move_time=1)
        move_grippers([puppet_bot], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)

    master_bot.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot.dxl.robot_set_operating_modes("single", "gripper", "position")
    # puppet_bot.dxl.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit
   
    torque_on(master_bot) 

    # move arms to starting position
    move_arms([master_bot], [start_arm_qpos] * 2, move_time=1)
    # move grippers to starting position
    move_grippers([master_bot], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)



def arm_teleop_task(master_bot, puppet_bot, master_only, stop_event):
    try:
        press_to_start(master_bot)
        gripper_command = JointSingleCommand(name="gripper")
        while not stop_event.is_set():


            master_state_joints = master_bot.dxl.joint_states.position[:6]
            master_gripper_joint = master_bot.dxl.joint_states.position[6]
            # sync joint positions
            if not master_only:
                puppet_bot.arm.set_joint_positions(master_state_joints, blocking=False)
            # sync gripper positions
                puppet_gripper_joint_target = MASTER2PUPPET_JOINT_FN(master_gripper_joint)
                gripper_command.cmd = puppet_gripper_joint_target
                puppet_bot.gripper.core.pub_single.publish(gripper_command)
            # sleep DT
            time.sleep(DT)
    except KeyboardInterrupt:
        print("Teleop task stopped")




def teleop(robot_side,master_only, init_node, stop_event):
    """ A standalone function for experimenting with teleoperation. No data recording. """

    puppet_bot =None
    if  not master_only: #allowing compatibility with simuation only
        if init_node:
            puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', init_node=False)
        else:
            puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', init_node=False)
    master_bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_{robot_side}', init_node=False)

    prep_robots(master_bot, puppet_bot, master_only)
 

    ### Teleoperation loop

    thread = threading.Thread(target=arm_teleop_task, args=(master_bot, puppet_bot, master_only, stop_event))
    thread.start()
    return master_bot,puppet_bot, thread


def reset(left_master, left_puppet, right_master, right_puppet,master_only,stop_event):
        prep_robots(left_master, left_puppet, master_only)
        prep_robots(right_master, right_puppet, master_only)
        left_thread = threading.Thread(target=arm_teleop_task, args=(left_master, left_puppet, master_only, stop_event))
        left_thread.start()
        right_thread = threading.Thread(target=arm_teleop_task, args=(right_master, right_puppet, master_only, stop_event))
        right_thread.start()
        return left_thread, right_thread

def get_params(robot):
    joint_state= robot.dxl.joint_states

    joint_pose = torch.Tensor(joint_state.position[:6])
    joint_vel=torch.Tensor(joint_state.velocity[:6])

    ee_pose = torch.Tensor(robot.arm.get_ee_pose())
    #TODO ee_velocity
    ee_vel = torch.zeros(6)

    #do be corrected
    gripper_params = get_gripper_params(robot)

    return [joint_pose, joint_vel, ee_pose, ee_vel, gripper_params]


def get_gripper_params(robot):
    
    joint = robot.dxl.joint_states.position[6]
    width = robot.gripper.gripper_value
    #TODO nachmessen da maxwidth 250
    thresh = 250 / 2

    if width < thresh:
        state -1.0
    else:
        state = 1.0
    params = [width, state, joint]
    return torch.tensor(params)


def get_pair_params_aloha(left, right):
    left_params = get_params(left)
    right_params = get_params(right)
    joint_pos = torch.concat((left_params[0], right_params[0]))
    joint_vel = torch.concat((left_params[1], right_params[1]))
    ee_pose = torch.concat((left_params[2], right_params[2]))
    ee_vel = torch.concat((left_params[3], right_params[3]))
    gripper_state = torch.concat((torch.tensor([left_params[4][0]]), torch.tensor([right_params[4][0]])))
    gripper_width = torch.concat((torch.tensor([left_params[4][1]]), torch.tensor([right_params[4][1]])))
    gripper_joint = torch.concat((torch.tensor([left_params[4][2]]), torch.tensor([right_params[4][2]])))

    return [joint_pos, joint_vel, ee_pose, ee_vel, gripper_state, gripper_width, gripper_joint]




def crop_img(img, cam_name):
    img = img
    if cam_name == 'cam_high':
        
        img = img[:350,50:500,:]#[80:,50:630,:] #[:,:,:]
        img=cv2.resize(img, (420, 340))
    elif cam_name == 'cam_left_wrist':
        img = img[:,:,:]#[:,:,:]
        img=cv2.resize(img, (224, 224))
    elif cam_name == 'cam_right_wrist':
        img = img[:,:,:]#[:,:,:]
        img=cv2.resize(img, (224, 224))
    else:
        raise NotImplementedError
    return img


def store_and_capture_cams_real(recorder, img_dir, step):
    
    imgs = recorder.get_images()
    for camera_name in bc.REALCAMS:
          
            #imageio.imwrite(f"{camera_name}.png", img)
            # Save the image
            img = imgs[camera_name]
            img = crop_img(img, camera_name)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            dir = f"{img_dir}/{camera_name}_orig/"
            cv2.imwrite(dir + str(step) + ".jpg", img_bgr)

def get_images(recorder):
    imgs = recorder.get_images()
    images = {}
    for camera_name in bc.REALCAMS:
          
            #imageio.imwrite(f"{camera_name}.png", img)
            # Save the image
            img = imgs[camera_name]
            img = crop_img(img, camera_name)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images[camera_name] = img_bgr
    return images

def get_observations(env, recorder):
    images = get_images(recorder)
    observation = env.get_observation()
    observation['images'] = images
    return observation

def step(action , env, recorder):
    (step_type,
    reward,
    discount,
    observation) = env.step(action)

    observation = get_observations(env, recorder)
    reward = reward
    done = False
    return observation, reward , done