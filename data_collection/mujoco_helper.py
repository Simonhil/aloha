

import cv2
import mink
import mujoco
import mujoco.viewer
import numpy as np
import torch
from data_collection.config import BaseConfig as bc


def get_ctrl_id_list(model, name):
    id_list = []
    for joint_name in bc.JOINT_NAMES:
        id_list.append(model.actuator(f"{name}/{joint_name}").id)
    return id_list


def mujoco_setup(xml_path):
            #mujoco setup
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)


            # Bodies for which to apply gravity compensation.
            left_subtree_id = model.body("left/base_link").id
            right_subtree_id = model.body("right/base_link").id

            # Get the dof and actuator ids for the joints we wish to control.
            joint_names: list[str] = []
            velocity_limits: dict[str, float] = {}
            for prefix in ["left", "right"]:
                for n in bc.JOINT_NAMES:
                    name = f"{prefix}/{n}"
                    joint_names.append(name)
                    velocity_limits[name] = bc.VELOCITY_LIMITS[n]
            dof_ids = np.array([model.joint(name).id for name in joint_names])
            data_diractuator_ids = np.array([model.actuator(name).id for name in joint_names])

            configuration = mink.Configuration(model)

            l_ee_task = mink.FrameTask(
                frame_name="left/gripper",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )

            r_ee_task = mink.FrameTask(
                frame_name="right/gripper",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )

            posture_task = mink.PostureTask(model, cost=1e-4)

            tasks = [l_ee_task, r_ee_task, posture_task]

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

            # Create a renderer

 


            renderer = mujoco.Renderer(model, bc.IMAGE_WIDTH,bc.IMAGE_HIGHT)
 


            #find all actuator ids
            left_joint_actuator = get_ctrl_id_list(model, "left")
            left_gripper_actuator = model.actuator("left/gripper").id
            right_joint_actuator = get_ctrl_id_list(model, "right")
            right_gripper_actuator = model.actuator("right/gripper").id

            viewer = mujoco.viewer.launch_passive(
            model=model, 
            data=data, 
            show_left_ui=False, 
            show_right_ui=False
            )

            mujoco.mj_resetDataKeyframe(model, data, model.key("neutral_pose").id)
            configuration.update(data.qpos)
            mujoco.mj_forward(model, data)
            posture_task.set_target_from_configuration(configuration)



            model.opt.timestep = bc.PHYSICSTIME # Increase to skip some calculations


            # Initialize mocap targets at the end-effector site.
            mink.move_mocap_to_frame(model, data, "left/target", "left/gripper", "site")
            mink.move_mocap_to_frame(model, data, "right/target", "right/gripper", "site")

            return (viewer, right_gripper_actuator, right_joint_actuator, left_gripper_actuator, left_joint_actuator,
                    posture_task, r_ee_task, l_ee_task, 
                    configuration, data_diractuator_ids,
                    model, data, renderer)








def get_ee_params(model, data, side):

    posid = model.body(f"{side}/target").id
    velid = model.body(f"{side}/gripper_base").id
    pos = torch.Tensor(data.xpos[posid])
    quat = torch.Tensor(data.xquat[posid])

   
    position = torch.concat((pos, quat)) #dim 7
    velocity = torch.Tensor(data.cvel[velid]) #dim6
    return position, velocity


def get_gripper_params(model, data,side ):
        
    id = model.joint(f"{side}/left_finger").id
    
    joint = data.qpos[id]
    
    idleft = model.body(f"{side}/left_finger_link").id
    idright = model.body(f"{side}/right_finger_link").id
    right = data.xpos[idright]
    left = data.xpos[idleft]

    state = 0
    
    width = abs(left - right)

    #by mesuring
    thresh = 0.07

    if width[1] < thresh:
        state -1.0
    else:
        state = 1.0
    params = [width[1], state, joint]
    return torch.tensor(params)
     
def get_joint_params(model,data, side:str):
    joint_names: list[str] = []
     #get all the ids
    for n in bc.JOINT_NAMES:
        name = f"{side}/{n}"
        joint_names.append(name)
    joint_ids = np.array([model.joint(name).id for name in joint_names])


    joint_pos = []
    joint_vel = []
    for i in joint_ids:
        joint_pos.append(data.qpos[i])
        joint_vel.append(data.qvel[i])

    return torch.Tensor(joint_pos), torch.Tensor(joint_vel)


def get_params(model, data, side:str):
    joint_pos, joint_vel = get_joint_params(model, data, side)
    ee_pos, ee_vel = get_ee_params(model, data, side)
    gripper_state = get_gripper_params(model, data, side)

    return [joint_pos, joint_vel, ee_pos, ee_vel, gripper_state]

def get_pair_params_mujoco(model,data):

    left_params = get_params(model, data, "left")
    right_params = get_params(model, data, "right")

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
    if cam_name == "overhead_cam":
        img = img[200:620,:,:]#[80:,50:630,:] #[:,:,:]
    elif cam_name == "wrist_cam_left":
        img = img[100:,:,:]#[:,:,:]
    elif cam_name == "wrist_cam_right":
        img = img[100:,:,:]#[:,:,:]
    else:
        raise NotImplementedError
    return img

def store_and_capture_cams_mujoco(data, renderer, names, img_dir, step):
    

    for camera_name in names:
               
            renderer.update_scene(data, camera=camera_name)
            img = renderer.render()

            
            #imageio.imwrite(f"{camera_name}.png", img)
            # Save the image
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bgr = crop_img(img_bgr, camera_name)
            img_bgr=cv2.resize(img_bgr, (bc.END_WIDTH, bc.END_HIGHT))
            dir = f"{img_dir}/{camera_name}_orig/"
            cv2.imwrite(dir + str(step) + ".jpg", img_bgr)