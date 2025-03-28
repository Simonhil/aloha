import threading
import time
import sys
import IPython
import torch
e = IPython.embed

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from aloha_scripts.constants import MASTER2PUPPET_JOINT_FN, DT, START_ARM_POSE, MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE
from aloha_scripts.robot_utils import torque_on, torque_off, move_arms, move_grippers, get_arm_gripper_positions

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
    start_arm_qpos = START_ARM_POSE[:6]
    move_arms([master_bot], [start_arm_qpos] * 2, move_time=1)
    # move grippers to starting position
    move_grippers([master_bot], [MASTER_GRIPPER_JOINT_MID, PUPPET_GRIPPER_JOINT_CLOSE], move_time=0.5)



def arm_teleop_task(master_bot, puppet_bot, master_only):
    try:
        press_to_start(master_bot)
        gripper_command = JointSingleCommand(name="gripper")
        while True:
            # sync joint positions
            master_state_joints = master_bot.dxl.joint_states.position[:6]
            puppet_bot.arm.set_joint_positions(master_state_joints, blocking=False)
            # sync gripper positions
            master_gripper_joint = master_bot.dxl.joint_states.position[6]
            puppet_gripper_joint_target = MASTER2PUPPET_JOINT_FN(master_gripper_joint)
            gripper_command.cmd = puppet_gripper_joint_target
            puppet_bot.gripper.core.pub_single.publish(gripper_command)
            # sleep DT
            time.sleep(DT)
    except KeyboardInterrupt:
        print("Teleop task stopped")




def teleop(robot_side,master_only = False, init_node = False):
    """ A standalone function for experimenting with teleoperation. No data recording. """

    if  not master_only: #allowing compatibility with simuation only
        if init_node:
            puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', init_node=True)
        else:
            puppet_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_{robot_side}', init_node=False)
    master_bot = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper", robot_name=f'master_{robot_side}', init_node=False)

    prep_robots(master_bot, puppet_bot, master_only)


    ### Teleoperation loop
    if not master_only:
        threading.Thread(target=arm_teleop_task, args=(master_bot, puppet_bot, master_only)).start()
    return master_bot,puppet_bot

def get_params(robot):
    joint_state= robot.dxl.joint_states

    joint_pose = joint_state.position[:6]
    joint_vel=joint_state.velocity[:6]

    ee_pose = robot.arm.get_ee_pose()
    #TODO ee_velocity
    ee_vel = []

    #do be corrected
    gripper_state = joint_state.velocity[6]

    return[joint_pose, joint_vel, ee_pose, ee_vel, gripper_state]


def get_gripper_params(robot):
    
    joint = get_arm_gripper_positions(robot)
    width = robot.gripper.gripper_value
    #TODO nachmessen da machwidth 250
    thresh = 250 / 2

    if width < thresh:
        state -1
    else:
        state = 1
    return {"state": state, "width":width, "joint":joint}



def get_pair_params_aloha(right, left):
    left_params = get_params(left)
    right_params = get_params(right)
    joint_pos = torch.concat((left_params[0], right_params[0]))
    joint_vel = torch.concat((left_params[1], right_params[1]))
    ee_pose = torch.concat((left_params[2], right_params[2]))
    ee_vel = torch.concat((left_params[3], right_params[3]))
    gripper_state = torch.concat((left_params[4], right_params[4]))

    return [joint_pos, joint_vel, ee_pose, ee_vel, gripper_state]