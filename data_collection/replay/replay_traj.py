import glob
import os
from pathlib import Path
import time

import mink
import mujoco
import numpy as np
import torch
from data_collection.mujoco_helper import get_ctrl_id_list, get_joint_params, mujoco_setup
from data_collection.config import BaseConfig as bc
import matplotlib.pyplot as plt


class JointReplay:

    def __init__(
        self,
        xml_path,
        data_dir,
        leader:bool,
        reward,
      
    ):
        print(data_dir)
        (self.viewer, self.right_gripper_actuator, self.right_joint_actuator, self.left_gripper_actuator, self.left_joint_actuator,
                    self.posture_task, self.r_ee_task, self.l_ee_task, 
                    self.configuration, self.data_diractuator_ids,
                   self. model, self.data)=mujoco_setup(xml_path)
        self.leader = leader
        self.jointpositions = self.unpack_joint_positions(data_dir)
      


    def unpack_joint_positions(self, episode_path):
        if self.leader:
            file = os.path.join(episode_path, 'leader_joint_pos.pt')
        else:
            file = os.path.join(episode_path, 'follower_joint_pos.pt')
        #path = os.path.join(episode_path, "*.pickle")
      
            # Keys contained in .pickle:
            # 'joint_state', 'joint_state_velocity', 'des_joint_state', 'des_joint_vel', 'end_effector_pos', 'end_effector_ori', 'des_gripper_width', 'delta_joint_state',
            # 'delta_des_joint_state', 'delta_end_effector_pos', 'delta_end_effector_ori', 'language_description', 'traj_length'
            #pt_file_path = os.path.join(episode_path, file)
        return torch.load(file)



    def move_one_side(self, side:str, poses):
        joint_names = []
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

           
    
    def move_robot_joint(self):

        new_joint_positions = []
        for step in self.jointpositions:
          
            # self.data.ctrl[self.left_gripper_actuator] = self.master_left.dxl.joint_states.position[6]
            # self.data.ctrl[self.right_gripper_actuator] = self.master_right.dxl.joint_states.position[6]
            self.move_one_side("left", step)
            self.move_one_side("right", step)
            mujoco.mj_step(self.model, self.data)  # Step the simulation
            self.viewer.sync()

            time.sleep(0.1)  # Control the simulation speed
            left_pos, _ = get_joint_params(self.model, self.data, "left")
            right_pos, _ = get_joint_params(self.model, self.data, "right")
            this_joint_pos = torch.concat((left_pos,right_pos))
            new_joint_positions.append(this_joint_pos)
            mink.move_mocap_to_frame(self.model, self.data, "left/target", "left/gripper", "site")
            mink.move_mocap_to_frame(self.model, self.data, "right/target", "right/gripper", "site")

        time.sleep(1)
        self.plot_joints(self.jointpositions, np.array(new_joint_positions))

        self.viewer.close()

    def plot_joints(self, first, second):
        # Number of positions in each inner array
       

        # Create subplots
        num_plots = 4
        num_colums = 3
        fig, axes = plt.subplots(num_plots, num_colums, figsize=(4 * num_colums, 4 * num_plots))
        # Plot each position separately
        fig.set_label("")
        fig.tight_layout(pad=4.0, h_pad=20) 
        fig.subplots_adjust(wspace=1, hspace=5)
        for i in range(num_plots):
            for j in range(num_colums):
                index = j + i*num_colums
                axes[i][j].plot(first[:, index], label=f'Joints, Position {index}', marker='o')
                axes[i][j].plot(second[:, index], label=f'Second, Position {index}', marker='s')
                axes[i][j].set_xticks(np.arange(0, 20, 1))
                axes[i][j].set_title(f'Plot for Position {index}')
                axes[i][j].set_xlabel('Index')
                axes[i][j].set_ylabel('Value')
                axes[i][j].legend()
                axes[i][j].grid(True)
           
        # Adjust layout
        plt.tight_layout()
        plt.show()






if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent.parent
    xml_path= _HERE / 'mujoco_assets' / "box_transfer.xml",
    data_dir= "/home/sihi/Desktop/2025_04_01-10_17_50",
    rp = JointReplay(
        xml_path="/home/sihi/Desktop/Bachelor/aloha/mujoco_assets/box_transfer.xml",
        data_dir="/home/sihi/Desktop/2025_04_01-10_17_50",
        leader=False, reward=None)
    rp.move_robot_joint()
