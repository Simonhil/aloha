import cv2
from matplotlib import pyplot as plt
import torch
from data_collection.config import BaseConfig as bc
from aloha_scripts.robot_utils import Recorder, ImageRecorder
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

def  save(img, name):

 
         
            img = crop_img(img, name)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            dir = f"{name}_orig"
            cv2.imwrite(dir + str(1) + ".jpg", img_bgr)
#recorder_left = Recorder('left', init_node=True)
#recorder_right = Recorder('right', init_node=False)
image_recorder = ImageRecorder(init_node=True)
img = image_recorder.get_images()
name = 'cam_high'
save(img[name], name)
# data = {}
# data.update({"test" : torch.load("/home/simonhilber/delete/2025_04_08-08_57_31/follower_joint_pos.pt")})
# print(len(data["test"]))