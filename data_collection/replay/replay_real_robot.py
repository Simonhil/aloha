import glob
import os
from pathlib import Path
import time

import cv2
import imageio
from matplotlib import pyplot as plt
import natsort
import numpy as np
import torch
from aloha_scripts import real_env
from aloha_scripts.constants import PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN
from data_collection.config import BaseConfig as bc
from data_collection.teleop_helper import get_params

class JointReplayReal:

    def __init__(
        self,
        data_dir,
        leader:bool,
       
        reward,
        pos
      
    ):
        self.leader = leader
        self.data_dir = data_dir
      
        
        self.jointpositions, self.gripper_joints= self.unpack(data_dir,pos)
 
        self.env = real_env.make_real_env(init_node=True)

    def unpack(self, episode_path, pos):
        if not pos:
            joints = self.unpack_single_param(episode_path,"joint_vel")
        else:  
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



   
    def move_robot_joint(self, plot):

        new_joint_positions = []
        new_gripper_joints = []

        self.env.reset()
        for i in range(0,len(self.jointpositions)):
            action_all_joint = torch.zeros((1,14))
            action_all_joint[0,:6] = self.jointpositions[i][:6]
            action_all_joint[0,6] = self.gripper_joints[i][0]
            action_all_joint[0,7:13] = self.jointpositions[i][6:]
            action_all_joint[0,13] = self.gripper_joints[i][1]

            print(i)
            self.env.step( action_all_joint)
           
            l_jp,l_jv,l_ep,l_ev,l_g= get_params(self.env.puppet_bot_left, False)
            r_jp,r_jv,r_ep,r_ev,r_g= get_params(self.env.puppet_bot_right, False)
            this_joint_pos = torch.concat((l_jp,r_jp))


            this_gripper_joint = torch.concat((l_g, r_g))


            new_gripper_joints.append( this_gripper_joint)
            new_joint_positions.append(this_joint_pos)
            time.sleep(bc.STEPSPEED)  # Control the simulation speed
        time.sleep(1)
        if plot:
            self.plot_joints(self.jointpositions, np.array(new_joint_positions))
            self.plot_gripper(self.gripper_joints, np.array(new_gripper_joints))
            plt.show()
 





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
                sup[i].plot(second[:,i], label=f'replay, Position {index}', marker='s')
                sup[i].set_xticks(np.arange(0, len(first), 200))
                sup[i].set_title(f'Plot for Position {index}')
                sup[i].set_xlabel('Index')
                sup[i].set_ylabel('Value')
                sup[i].legend()
                sup[i].grid(True)
           
        # Adjust layout
        plt.tight_layout()





def create_img_vector(img_folder_path, trajectory_length):
    cam_list = []
    img_paths = glob.glob(os.path.join(img_folder_path, '*.png'))
    img_paths = natsort.natsorted(img_paths)
    #assert len(img_paths)==trajectory_length, "Number of images does not equal trajectory length!"

    for img_path in img_paths:
        img_array = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR)
        cam_list.append(img_array)
    return cam_list



# takes the images from img_fir and saves them as video in dir
def make_video(img_dir, name, dir):
    print("img  " + str(img_dir))
    frames = create_img_vector(img_dir)
    imageio.mimsave(f"{dir}/{name}.mp4", np.stack(frames), fps=25)





def single_replay(replay, video, leader, reward, dir, plot,pos):
    if replay :
        

        rp = JointReplayReal(
            # xml_path="/home/sihi/Desktop/Bachelor/aloha/mujoco_assets/box_transfer.xml",
            # data_dir="/home/sihi/delete/download/EXAMPLE",
            #xml_path="/home/i53/student/shilber/aloha/mujoco_assets/box_transfer.xml",
            data_dir= dir,
            # data_dir="/home/simonhilber/delete/2025_04_03-09_26_22",
            leader=leader, reward=reward,
            pos=pos)
        

        rp.move_robot_joint(plot)
    if video :
        make_video(dir + str("/images/cam_high_orig"), "top",dir)
        #make_video(dir + str ("/images/wrist_cam_left_orig"), "left[100:,:,:]",dir)
        #make_video(dir+ str ("/images/wrist_cam_right_orig"), "right[100:,:,:]", dir)

if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent.parent
    replay = True
    video = False
    #data_path = "/home/i53/student/shilber/Downloads/first10_50HZ"
    data_path = "/home/simonhilber/delete/2025_04_14-10_39_41"
    single_replay(replay, video=video, leader=True,  reward=None, dir= data_path, plot=True, pos= True)
  

    
