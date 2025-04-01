

import cv2
import mink
import mujoco
import numpy as np
import torch
from data_collection.config import BaseConfig as bc


def get_ctrl_id_list(model, name):
    id_list = []
    for joint_name in bc.JOINT_NAMES:
        id_list.append(model.actuator(f"{name}/{joint_name}").id)
    return id_list


def get_ee_params(model, data, side):

    posid = model.body(f"{side}/target").id
    velid = model.body(f"{side}/gripper_base").id
    pos = torch.Tensor(data.xpos[posid])
    quat = torch.Tensor(data.xquat[posid])

   
    position = torch.concat((pos, quat)) #dim 7
    velocity = torch.Tensor(data.cvel[velid]) #dim6
    return position, velocity


def get_gripper_params(model, data,side ):
        


    # id = model.joint(F"{side}/gripper").id
    # joint = data.qpos[id]
    
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
    params = [width[1], state]
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
    gripper_params = torch.concat((left_params[4], right_params[4]))


    return [joint_pos, joint_vel, ee_pose, ee_vel, gripper_params]


def crop_img(img, cam_name):
    img = img
    if cam_name is "overhead_cam":
        img = img[300:1100,100:900,:]
    elif cam_name is "wrist_cam_left":
        img = img[:,:,:]
    elif cam_name is "wrist_cam_right":
        img = img[:,:,:]
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
            img_bgr=cv2.resize(img_bgr, (bc.IMAGE_WIDTH, bc.IMAGE_HIGHT))
            dir = f"{img_dir}/{camera_name}_orig/"
            cv2.imwrite(dir + str(step) + ".png", img_bgr)