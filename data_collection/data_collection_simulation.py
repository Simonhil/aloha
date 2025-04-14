
from datetime import datetime
import threading
import time
import mujoco
import mujoco.viewer
import torch
import numpy as np
from pathlib import Path
import mink
import cv2
import shutil
from enum import Enum, auto
from aloha_scripts.robot_utils import Recorder, ImageRecorder
#from aloha_scripts.real_aloha_example import get_ctrl_id_list
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from data_collection.mujoco_helper import get_pair_params_mujoco, mujoco_setup, store_and_capture_cams_mujoco
from data_collection.teleop_helper import  get_action, get_pair_params_aloha, opening_ceremony, reset, store_and_capture_cams_real, teleop, prep_robots
from data_collection.config import BaseConfig as bc
import aloha_scripts.real_env as real_env
from utils.keyboard import KeyManager
class TeleoperationType(Enum):
    JOINT_SPACE = auto()
    TASK_SPACE = auto()


class DataCollectionManager:
    def __init__(
        self,
        xml_path,
        data_dir,
        cam_names,
        reward_func,
        simulation= False
      
    ):
        self.xml_path = xml_path
        self.step = 0
        self.cam_names = cam_names
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)
        self.is_simulation= simulation
        if not simulation:
            self.master_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=True)
            self.master_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)
            self.env = real_env.make_real_env(init_node=False)
            
        else:
            self.master_left, self.puppet_left, self.left_thread= teleop("left",self.is_simulation, True, bc.STOPEVENT)
            self.master_right, self.puppet_right, self.right_thread= teleop("right",self.is_simulation, False, bc.STOPEVENT)
        print("setting up")

        print("setting up bots")
        # self.master_left, self.puppet_left = teleop("left",simulation, True)
        # self.master_right, self.puppet_right = teleop("right",simulation, False)
        print("bots set up")


    

    


    def reset(self):
        bc.STOPEVENT.clear()
        #reset real robots
        
        self.env.reset()
        if self.is_simulation:
            self.left_thread, self.right_thread = reset(self.master_left, self.puppet_left, self.master_right, self.puppet_right,self.is_simulation, bc.STOPEVENT)
            (self.viewer, self.right_gripper_actuator, self.right_joint_actuator, self.left_gripper_actuator, self.left_joint_actuator,
                    self.posture_task, self.r_ee_task, self.l_ee_task, 
                    self.configuration, self.data_diractuator_ids,
                   self. model, self.data, self.renderer)=mujoco_setup(self.xml_path)
        else: 
            opening_ceremony(self.master_left, self.master_right, self.env.puppet_bot_left, self.env.puppet_bot_right)

        print("reset complete")
    #currently missing implementation for real cameras
    def start_key_listener(self):
        km = KeyManager()
        print("Press 'n' to collect new data or 'q' to quit data collection")
        #opening_ceremony(self.master_left, self.master_right, self.env.puppet_bot_left, self.env.puppet_bot_right)

        while km.key != "q":
            if km.key == "n":
                print()
                print("Preparing for new data collection")
                self.reset()
                if self.is_simulation:
                    self.l_ee_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, "left/target"))
                    self.r_ee_task.set_target(mink.SE3.from_mocap_name(self.model, self.data, "right/target"))
                self.__create_new_recording_dir()
                self.__create_empty_data()
                

                print("Start! Press 's' to save collected data or 'd' to discard.")

                self.timestep = 0
                while km.key not in ["s", "d"]:
                    self.__collection_step(self.timestep)
                    self.timestep += 1
                    km.pool()

                else:
                    print("stopping")
                    bc.STOPEVENT.set() 
                    if self.is_simulation:
                        self.viewer.close()
                        del self.model
                        del self.data
                        time.sleep(0.1)
                    else:
                        # self.left_thread.join()
                        # self.right_thread.join()
                        pass
                    if km.key == "s":
                        print()
                        print("Saving data")

                        self.__save_data()

                        print("Saved!")
                    elif km.key == "d":
                        print()
                        print("Discarding data")
                        shutil.rmtree(self.record_dir)
                        print("Discarded!")

                    print(
                        "Press 'n' to collect new data or 'q' to quit data collection"
                    )

            km.pool()

        print()
        print("Ending data collection...")
        km.close()
        #self.__close_hardware_connections()


    def __create_new_recording_dir(self):
        self.record_dir = self.data_dir / datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.record_dir.mkdir()

        self.image_dir = self.record_dir / "images"
        self.image_dir.mkdir()
        if self.is_simulation:
            for name in self.cam_names:
                device_dir = self.image_dir / f"{name}_orig"
                device_dir.mkdir()
        else:
            for device in bc.REALCAMS:
                device_dir = self.image_dir / f"{device}_orig"
                device_dir.mkdir()
            
            # for device in self.continuous_devices:
            #     device_dir = self.image_dir / f"{device.name}_orig"
            #     device_dir.mkdir()

    def __create_empty_data(self):
        self.leader_joint_pos_list = []
        self.leader_joint_vel_list = []
        self.leader_ee_pos_list = []
        self.leader_ee_vel_list = []
        self.leader_gripper_state_list = []
        self.leader_gripper_width_list = []
        self.leader_gripper_joint_list = []
        self.leader_time =[]
        
        self.follower_joint_pos_list = []
        self.follower_joint_vel_list = []
        self.follower_ee_pos_list = []
        self.follower_ee_vel_list = []
        self.follower_gripper_state_list = []
        self.follower_gripper_width_list = []
        self.follower_gripper_joint_list = []
        self.follower_time = []


        self.image_times = []

    def collection(self):
        timestep = 0
        leader_time = time.time()
        leader_params = get_pair_params_aloha(self.master_left, self.master_right, is_master=True)
        
        if self.is_simulation:
            follower_time = time.time()
            follower_params = get_pair_params_mujoco(self.model, self.data)
            

        else:
            follower_time = time.time()
            follower_params = get_pair_params_aloha(self.env.puppet_bot_left, self.env.puppet_bot_right, is_master=False)
        
        

        # if self.is_simulation:
        #     store_and_capture_cams_mujoco(self.data, self.renderer, self.cam_names,self.image_dir, timestep)
        #     pass

        self.leader_joint_pos_list.append(leader_params[0])
        self.leader_joint_vel_list.append(leader_params[1])
        self.leader_ee_pos_list.append(leader_params[2])
        self.leader_ee_vel_list.append(leader_params[3])
        self.leader_gripper_state_list.append(leader_params[4])
        self.leader_gripper_width_list.append(leader_params[5])
        self.leader_gripper_joint_list.append(leader_params[6])
        self.leader_time.append(leader_time)
        
        self.follower_joint_pos_list.append(follower_params[0])
        self.follower_joint_vel_list.append(follower_params[1])
        self.follower_ee_pos_list.append(follower_params[2])
        self.follower_ee_vel_list.append(follower_params[3])
        self.follower_gripper_state_list.append(follower_params[4])
        self.follower_gripper_width_list.append(follower_params[5])
        self.follower_gripper_joint_list.append(follower_params[6])
        self.follower_time.append(follower_time)
        timestep += 1

        
    def __collection_step(self, timestep: int):
        if self.is_simulation:
        #teleoperation to mujoco
            for index in range(6):
                self.data.ctrl[self.left_joint_actuator[index]] = self.master_left.dxl.joint_states.position[index]
                self.data.ctrl[self.right_joint_actuator[index]] = self.master_right.dxl.joint_states.position[index]
            self.data.ctrl[self.left_gripper_actuator] = self.master_left.dxl.joint_states.position[6]
            self.data.ctrl[self.right_gripper_actuator] = self.master_right.dxl.joint_states.position[6]
            mujoco.mj_step(self.model, self.data)  # Step the simulation
            self.viewer.sync()
            self.collection()
            
            mink.move_mocap_to_frame(self.model, self.data, "left/target", "left/gripper", "site")
            mink.move_mocap_to_frame(self.model, self.data, "right/target", "right/gripper", "site")

        else :
            action = torch.zeros((1,14))
            action[0]=torch.tensor(get_action(self.master_left, self.master_right))
            self.env.step(action)
            self.collection()
            images_times = store_and_capture_cams_real(self.env.image_recorder, self.image_dir,self.timestep)
            self.image_times.append(images_times)
        #time.sleep(bc.FREQ)

    def __save_data(self):
        leader_joint_pos_list = torch.stack(self.leader_joint_pos_list)
        leader_joint_vel_list = torch.stack(self.leader_joint_vel_list)
        leader_ee_pos_list = torch.stack(self.leader_ee_pos_list)
        leader_ee_vel_list = torch.stack(self.leader_ee_vel_list)
        leader_gripper_state_list = torch.stack(self.leader_gripper_state_list)
        leader_gripper_width_list = torch.stack(self.leader_gripper_width_list)
        leader_gripper_joint_list = torch.stack(self.leader_gripper_joint_list)
        
        follower_joint_pos_list = torch.stack(self.follower_joint_pos_list)
        follower_joint_vel_list = torch.stack(self.follower_joint_vel_list)
        follower_ee_pos_list = torch.stack(self.follower_ee_pos_list)
        follower_ee_vel_list = torch.stack(self.follower_ee_vel_list)
        follower_gripper_state_list = torch.stack(self.follower_gripper_state_list)
        follower_gripper_width_list = torch.stack(self.follower_gripper_width_list)
        follower_gripper_joint_list = torch.stack(self.follower_gripper_joint_list)

        torch.save(leader_joint_pos_list, self.record_dir / "leader_joint_pos.pt")
        torch.save(leader_joint_vel_list, self.record_dir / "leader_joint_vel.pt")
        torch.save(leader_ee_pos_list, self.record_dir / "leader_ee_pos.pt")
        torch.save(leader_ee_vel_list, self.record_dir / "leader_ee_vel.pt")
        torch.save(leader_gripper_state_list, self.record_dir / "leader_gripper_state.pt")
        torch.save(leader_gripper_width_list, self.record_dir / "leader_gripper_width.pt")
        torch.save(leader_gripper_joint_list, self.record_dir / "leader_gripper_joint.pt")
        torch.save(self.leader_time, self.record_dir / "leader_time.pt")

        torch.save(follower_joint_pos_list, self.record_dir / "follower_joint_pos.pt")
        torch.save(follower_joint_vel_list, self.record_dir / "follower_joint_vel.pt")
        torch.save(follower_ee_pos_list, self.record_dir / "follower_ee_pos.pt")
        torch.save(follower_ee_vel_list, self.record_dir / "follower_ee_vel.pt")
        torch.save(follower_gripper_state_list, self.record_dir / "follower_gripper_state.pt")
        torch.save(follower_gripper_width_list, self.record_dir / "follower_gripper_width.pt")
        torch.save(follower_gripper_joint_list, self.record_dir / "follower_gripper_joint.pt")
        torch.save(self.follower_time, self.record_dir / "follower_time.pt")

        cam_time_top = []
        cam_time_left = []
        cam_time_right = []

        print(self.image_times[0].keys())
        for step in self.image_times:
            cam_time_top.append( step['cam_high'])
            cam_time_left.append(step['cam_left_wrist'])
            cam_time_right.append(step['cam_right_wrist'])

        torch.save(cam_time_top, self.record_dir / "cam_time_top.pt")
        torch.save(cam_time_left, self.record_dir / "cam_time_left.pt")
        torch.save(cam_time_right, self.record_dir / "cam_time_right.pt")
        #torch.save(self.im, self.record_dir / "follower_time.pt")

    # def __close_hardware_connections(self):
    #     self.follower_gripper.close()
    #     self.follower_arm.close()
    #     self.leader_gripper.close()
    #     self.leader_arm.close()

    #     for device in self.discrete_devices:
    #         device.close()
        
    #     for device in self.continuous_devices:
    #         device.close()

#from real_robot_env.robot.hardware_azure import Azure
#from real_robot_env.robot.hardware_depthai import DepthAI
#from real_robot_env.robot.hardware_realsense import RealSense
#from real_robot_env.robot.hardware_gopro import GoPro
#from real_robot_env.robot.hardware_audio import AudioInterface


#rewards
from rewards import place_holder


if __name__ == "__main__":
    _HERE = Path(__file__).parent.parent
    cam_names=[str]
    data_collection_manager = DataCollectionManager(
        xml_path= _HERE / 'mujoco_assets' / "box_transfer.xml",
        data_dir=Path("/home/simonhilber/delete"),
        cam_names = bc.SIMCAMS,
        reward_func = place_holder,
        simulation= False
       
    )

    data_collection_manager.start_key_listener()