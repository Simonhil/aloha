import glob
import os
from pathlib import Path
import time

import cv2
import natsort

import mink
import mujoco
import numpy as np
import torch
import imageio
from data_collection.mujoco_helper import get_ctrl_id_list, get_gripper_params, get_joint_params, mujoco_setup, store_and_capture_cams_mujoco
from data_collection.config import BaseConfig as bc
import matplotlib.pyplot as plt


class JointReplay:

    def __init__(
        self,
        xml_path,
        data_dir,
        leader:bool,
        cam_record:bool,
        stepsize,
        reward,
      
    ):
        self.data_dir = data_dir
        self.stepsize = stepsize
        self.cam_record = cam_record
        (self.viewer, self.right_gripper_actuator, self.right_joint_actuator, self.left_gripper_actuator, self.left_joint_actuator,
                    self.posture_task, self.r_ee_task, self.l_ee_task, 
                    self.configuration, self.data_diractuator_ids,
                   self. model, self.data,self.renderer)=mujoco_setup(xml_path)
        self.leader = leader
        
        self.jointpositions, self.gripper_joints= self.unpack(data_dir)
 
      

    def unpack(self, episode_path):
        joints = self.unpack_single_param(episode_path,"joint_pos")
        gripper_joints = self.unpack_single_param(episode_path,"gripper_joint")
        return joints, gripper_joints

    def unpack_single_param(self, episode_path, param):
        if self.leader:
            file = os.path.join(episode_path, f'leader_{param}.pt')
        else:
            file = os.path.join(episode_path, f'follower_{param}.pt')
        #path = os.path.join(episode_path, "*.pickle")
      
            # Keys contained in .pickle:
            # 'joint_state', 'joint_state_velocity', 'des_joint_state', 'des_joint_vel', 'end_effector_pos', 'end_effector_ori', 'des_gripper_width', 'delta_joint_state',
            # 'delta_des_joint_state', 'delta_end_effector_pos', 'delta_end_effector_ori', 'language_description', 'traj_length'
            #pt_file_path = os.path.join(episode_path, file)
        return torch.load(file)



    def joint_move(self, poses, gripper_joints):
        left_ids = get_ctrl_id_list(self.model, "left")
        right_ids =get_ctrl_id_list(self.model, "right")
        # for n in bc.JOINT_NAMES:
        #     name = f"{side}/{n}"
        #     joint_names.append(name)
        # joint_ids = np.array([self.model.joint(name).id for name in joint_names])
        #poses = torch.concat((poses, torch.tensor([0])))
        for i in range(6):
            self.data.ctrl[left_ids[i]] = poses[left_ids[i]]
            self.data.ctrl[right_ids[i]] = poses[right_ids[i]-1]
        self.data.ctrl[self.left_gripper_actuator] = gripper_joints[0]
        self.data.ctrl[self.right_gripper_actuator] = gripper_joints[1]
    
    def move_robot_joint(self, plot):

        new_joint_positions = []
        new_gripper_joints = []

        
        for i in range(0,len(self.jointpositions),self.stepsize):
            self.joint_move(self.jointpositions[i], self.gripper_joints[i])
            mujoco.mj_step(self.model, self.data)  # Step the simulation
            self.viewer.sync()


            #cam
            if self.cam_record:
                img_dir = self.data_dir +"/images"
                store_and_capture_cams_mujoco(self.data, self.renderer, bc.SIMCAMS, img_dir, i)






            time.sleep(bc.STEPSPEED)  # Control the simulation speed
            left_pos, _ = get_joint_params(self.model, self.data, "left")
            right_pos, _ = get_joint_params(self.model, self.data, "right")
            this_joint_pos = torch.concat((left_pos,right_pos))


            left_g = torch.tensor([get_gripper_params(self.model, self.data, "left")[2]])
            right_g = torch.tensor([get_gripper_params(self.model, self.data, "right")[2]])
            this_gripper_joint = torch.concat((left_g, right_g))


            new_gripper_joints.append( this_gripper_joint)
            new_joint_positions.append(this_joint_pos)
            mink.move_mocap_to_frame(self.model, self.data, "left/target", "left/gripper", "site")
            mink.move_mocap_to_frame(self.model, self.data, "right/target", "right/gripper", "site")

        time.sleep(1)
        if plot:
            self.plot_joints(self.jointpositions, np.array(new_joint_positions))
            self.plot_gripper(self.gripper_joints, np.array(new_gripper_joints))
            plt.show()
        self.viewer.close()





    def plot_joints(self, first, second):
        # Number of positions in each inner array
       

        # Create subplots
        num_plots = 4
        num_colums = 3
        fig, sup = plt.subplots(num_plots, num_colums, figsize=(6 * num_colums, 4 * num_plots))
        # Plot each position separately
        fig.set_label("")
        fig.tight_layout(pad=4.0, h_pad=20) 
        fig.subplots_adjust(wspace=1, hspace=5)
        for i in range(num_plots):
            for j in range(num_colums):
                index = j + i*num_colums
                sup[i][j].plot(first[:, index], label=f'Joints, Position {index}', marker='o')
                sup[i][j].plot(second[:, index], label=f'Second, Position {index}', marker='s')
                sup[i][j].set_xticks(np.arange(0, len(first), 200))
                sup[i][j].set_title(f'Plot for Position {index}')
                sup[i][j].set_xlabel('Index')
                sup[i][j].set_ylabel('Value')
                sup[i][j].legend()
                sup[i][j].grid(True)
           
        # Adjust layout
        plt.tight_layout()
        


    def plot_gripper(self, first, second):
        # Number of positions in each inner array
       

        # Create subplots
        num_plots = 2
        num_colums = 1
        fig, sup = plt.subplots(num_plots, num_colums, figsize=(6 * num_colums, 4 * num_plots))
        # Plot each position separately
        fig.set_label("gripper")
        fig.tight_layout(pad=4.0, h_pad=20) 
        fig.subplots_adjust(wspace=1, hspace=5)
        for i in range(num_plots):
                index = i
                sup[i].plot(first[:,i], label=f'gripper, Position {index}', marker='o')
                sup[i].plot(second[:,i], label=f'Second, Position {index}', marker='s')
                sup[i].set_xticks(np.arange(0, len(first), 200))
                sup[i].set_title(f'Plot for Position {index}')
                sup[i].set_xlabel('Index')
                sup[i].set_ylabel('Value')
                sup[i].legend()
                sup[i].grid(True)
           
        # Adjust layout
        plt.tight_layout()


def create_img_vector(img_folder_path):
    cam_list = []
    img_paths = glob.glob(os.path.join(img_folder_path, '*.png'))
    img_paths = natsort.natsorted(img_paths)
    #assert len(img_paths)==trajectory_length, "Number of images does not equal trajectory length!"

    for img_path in img_paths:
        img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        cam_list.append(img_array)
    return cam_list

def make_video(img_dir, name, dir):
    print("img  " + str(img_dir))
    frames = create_img_vector(img_dir)
    imageio.mimsave(f"{dir}/{name}.mp4", np.stack(frames), fps=25)

def single_replay(replay, video, leader,cam, step, reward, dir, plot):
    if replay :
        xml_path= _HERE / 'mujoco_assets' / "box_transfer.xml",
        data_dir= "/home/sihi/Desktop/2025_04_01-10_17_50",
        rp = JointReplay(
            # xml_path="/home/sihi/Desktop/Bachelor/aloha/mujoco_assets/box_transfer.xml",
            # data_dir="/home/sihi/delete/download/EXAMPLE",
            xml_path="/home/i53/student/shilber/aloha/mujoco_assets/box_transfer.xml",
            data_dir= dir,
            # xml_path="/home/simonhilber/aloha/mujoco_assets/box_transfer.xml",
            # data_dir="/home/simonhilber/delete/2025_04_03-09_26_22",
            leader=leader, cam_record = cam,stepsize=step, reward=reward)
        rp.move_robot_joint(plot)
    if video :
        make_video(dir + str("/images/overhead_cam_orig"), "top",dir)
        #make_video(dir + str ("/images/wrist_cam_left_orig"), "left[100:,:,:]",dir)
        #make_video(dir+ str ("/images/wrist_cam_right_orig"), "right[100:,:,:]", dir)

if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent.parent
    replay = False
    video = True
    data_path = "/home/i53/student/shilber/Downloads/first10_50HZ"
    #single_replay(replay, video=video, leader=True, cam=True,step=1,  reward=None, dir= data_path)
    for name in os.listdir(data_path):
        dir = data_path + "/" + str(name)
        print(name)
        single_replay(replay=replay, video=video, leader=True, cam=True,step=1, reward=None, dir= dir, plot = False)

    
