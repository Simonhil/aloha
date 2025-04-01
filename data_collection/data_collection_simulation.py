
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
#from aloha_scripts.real_aloha_example import get_ctrl_id_list
from data_collection.mujoco_helper import get_pair_params_mujoco, store_and_capture_cams_mujoco
from data_collection.teleop_helper import  get_pair_params_aloha, teleop, prep_robots
from data_collection.config import BaseConfig as bc
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
        print("setting up")

        print("setting up bots")
        # self.master_left, self.puppet_left = teleop("left",simulation, True)
        # self.master_right, self.puppet_right = teleop("right",simulation, False)
        print("bots set up")

    def get_ctrl_id_list(self,model, name):
        id_list = []
        for joint_name in bc.JOINT_NAMES:
            id_list.append(model.actuator(f"{name}/{joint_name}").id)
        return id_list


    def mujoco_setup(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        model = self.model

        # Bodies for which to apply gravity compensation.
        left_subtree_id = self.model.body("left/base_link").id
        right_subtree_id = self.model.body("right/base_link").id

        # Get the dof and actuator ids for the joints we wish to control.
        joint_names: list[str] = []
        velocity_limits: dict[str, float] = {}
        for prefix in ["left", "right"]:
            for n in bc.JOINT_NAMES:
                name = f"{prefix}/{n}"
                joint_names.append(name)
                velocity_limits[name] = bc.VELOCITY_LIMITS[n]
        dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.data_diractuator_ids = np.array([self.model.actuator(name).id for name in joint_names])

        self.configuration = mink.Configuration(self.model)

        self.l_ee_task = mink.FrameTask(
            frame_name="left/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        self.r_ee_task = mink.FrameTask(
            frame_name="right/gripper",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )

        self.posture_task = mink.PostureTask(model, cost=1e-4)

        tasks = [self.l_ee_task, self.r_ee_task, self.posture_task]

         # Enable collision avoidance between the following geoms.
        l_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("left/wrist_link").id)
        r_wrist_geoms = mink.get_subtree_geom_ids(model, model.body("right/wrist_link").id)
        l_geoms = mink.get_subtree_geom_ids(model, model.body("left/upper_arm_link").id)
        r_geoms = mink.get_subtree_geom_ids(model, model.body("right/upper_arm_link").id)
        frame_geoms = mink.get_body_geom_ids(model, model.body("metal_frame").id)
        collision_pairs = [
            (l_wrist_geoms, r_wrist_geoms),
            (l_geoms + r_geoms, frame_geoms + ["table"]),
        ]
        collision_avoidance_limit = mink.CollisionAvoidanceLimit(
            model=model,
            geom_pairs=collision_pairs,  # type: ignore
            minimum_distance_from_collisions=0.05,
            collision_detection_distance=0.1,
        )

        limits = [
            mink.ConfigurationLimit(model=model),
            mink.VelocityLimit(model, velocity_limits),
            collision_avoidance_limit,
        ]

        l_mid = model.body("left/target").mocapid[0]
        r_mid = model.body("right/target").mocapid[0]
        solver = "quadprog"
        pos_threshold = 5e-3
        ori_threshold = 5e-3
        max_iters = 5



        #find all actuator ids
        self.left_joint_actuator = self.get_ctrl_id_list(model, "left")
        self.left_gripper_actuator = model.actuator("left/gripper").id
        self.right_joint_actuator = self.get_ctrl_id_list(model, "right")
        self.right_gripper_actuator = model.actuator("right/gripper").id

        self.viewer = mujoco.viewer.launch_passive(
        model=model, 
        data=self.data, 
        show_left_ui=False, 
        show_right_ui=False
        )

        print("mojoco")
        mujoco.mj_resetDataKeyframe(model, self.data, model.key("neutral_pose").id)
        self.configuration.update(self.data.qpos)
        mujoco.mj_forward(model, self.data)
        self.posture_task.set_target_from_configuration(self.configuration)

        # Initialize mocap targets at the end-effector site.
        mink.move_mocap_to_frame(model, self.data, "left/target", "left/gripper", "site")
        mink.move_mocap_to_frame(model, self.data, "right/target", "right/gripper", "site")




        self.left_frame_id = model.site("left/gripper").id
        self.right_frame_id = model.site("right/gripper").id
        self.left_mocap_id =  model.body_mocapid[model.body("left/target").id]
        self.right_mocap_id =  model.body_mocapid[model.body("right/target").id]
      
        self.renderer = mujoco.Renderer(model,1000,1000)

        print("mujoco ready")




    def reset(self):
        
        #reset real robots
        self.master_left, self.puppet_left = teleop("left",self.is_simulation, True)
        self.master_right, self.puppet_right = teleop("right",self.is_simulation, False)
        if self.is_simulation:
            self.mujoco_setup(self.xml_path)
            print("reset complete")

    #currently missing implementation for real cameras
    def start_key_listener(self):
        km = KeyManager()
        print("Press 'n' to collect new data or 'q' to quit data collection")

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
                collection = threading.Thread(target=self.collection, args=())


                self.stop_event = threading.Event()
                collection.start()
                

                print("Start! Press 's' to save collected data or 'd' to discard.")

                self.timestep = 0
                while km.key not in ["s", "d"]:
                    self.__collection_step(self.timestep)
                    self.timestep += 1
                    km.pool()

                else:
                    print("stopping")
                    self.viewer.close()
                    
                    self.stop_event.set() 
                    collection.join()
                    del self.model
                    del self.data
                    time.sleep(0.1)
                   
                    if km.key == "s":
                        print()
                        print("Saving data")

                        self.__save_data()

                        print("Saved!")
                    elif km.key == "d":
                        print()
                        print("Discarding data")

                        self.record_dir.rmdir()
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
            for device in self.discrete_devices:
                device_dir = self.image_dir / f"{device.name}_orig"
                device_dir.mkdir()
            
            for device in self.continuous_devices:
                device_dir = self.image_dir / f"{device.name}_orig"
                device_dir.mkdir()

    def __create_empty_data(self):
        self.leader_joint_pos_list = []
        self.leader_joint_vel_list = []
        self.leader_ee_pos_list = []
        self.leader_ee_vel_list = []
        self.leader_gripper_state_list = []
        
        self.follower_joint_pos_list = []
        self.follower_joint_vel_list = []
        self.follower_ee_pos_list = []
        self.follower_ee_vel_list = []
        self.follower_gripper_state_list = []

    def collection(self):
        timestep = 0
        while not self.stop_event.is_set():
            leader_params = get_pair_params_aloha(self.master_left, self.master_right)
            
            if self.is_simulation:
                follower_params = get_pair_params_mujoco(self.model, self.data)
                

            else:
                follower_params = get_pair_params_aloha(self.puppet_left, self.puppet_right)
            
            

            if self.is_simulation:
                store_and_capture_cams_mujoco(self.data, self.renderer, self.cam_names,self.image_dir, timestep)

            self.leader_joint_pos_list.append(leader_params[0])
            self.leader_joint_vel_list.append(leader_params[1])
            self.leader_ee_pos_list.append(leader_params[2])
            self.leader_ee_vel_list.append(leader_params[3])
            self.leader_gripper_state_list.append(leader_params[4])
            
            self.follower_joint_pos_list.append(follower_params[0])
            self.follower_joint_vel_list.append(follower_params[1])
            self.follower_ee_pos_list.append(follower_params[2])
            self.follower_ee_vel_list.append(follower_params[3])
            self.follower_gripper_state_list.append(follower_params[4])
            timestep += 1
            time.sleep(1/bc.FREQ)

        
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
            time.sleep(0.01)  # Control the simulation speed
            
            mink.move_mocap_to_frame(self.model, self.data, "left/target", "left/gripper", "site")
            mink.move_mocap_to_frame(self.model, self.data, "right/target", "right/gripper", "site")



    def __save_data(self):
        leader_joint_pos_list = torch.stack(self.leader_joint_pos_list)
        leader_joint_vel_list = torch.stack(self.leader_joint_vel_list)
        leader_ee_pos_list = torch.stack(self.leader_ee_pos_list)
        leader_ee_vel_list = torch.stack(self.leader_ee_vel_list)
        #leader_gripper_state_list = torch.Tensor(self.leader_gripper_state_list)
        
        follower_joint_pos_list = torch.stack(self.follower_joint_pos_list)
        follower_joint_vel_list = torch.stack(self.follower_joint_vel_list)
        follower_ee_pos_list = torch.stack(self.follower_ee_pos_list)
        follower_ee_vel_list = torch.stack(self.follower_ee_vel_list)
        #follower_gripper_state_list = torch.Tensor(self.follower_gripper_state_list)

        torch.save(leader_joint_pos_list, self.record_dir / "leader_joint_pos.pt")
        torch.save(leader_joint_vel_list, self.record_dir / "leader_joint_vel.pt")
        torch.save(leader_ee_pos_list, self.record_dir / "leader_ee_pos.pt")
        torch.save(leader_ee_vel_list, self.record_dir / "leader_ee_vel.pt")
        #torch.save(leader_gripper_state_list, self.record_dir / "leader_gripper_state.pt")

        torch.save(follower_joint_pos_list, self.record_dir / "follower_joint_pos.pt")
        torch.save(follower_joint_vel_list, self.record_dir / "follower_joint_vel.pt")
        torch.save(follower_ee_pos_list, self.record_dir / "follower_ee_pos.pt")
        torch.save(follower_ee_vel_list, self.record_dir / "follower_ee_vel.pt")
        #torch.save(follower_gripper_state_list, self.record_dir / "follower_gripper_state.pt")

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
        simulation= True
       
    )

    data_collection_manager.start_key_listener()