import numpy as np


class BaseConfig:
    
    JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate", 
    ]

    # Single arm velocity limits, taken from:
    # https://github.com/Interbotix/interbotix_ros_manipulators/blob/main/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro
    VELOCITY_LIMITS = {k: np.pi for k in JOINT_NAMES}
    IMAGE_WIDTH= 680
    IMAGE_HIGHT=680
    END_WIDTH = 224
    END_HIGHT = 224
    FREQ = 0.02
    PHYSICSTIME = 0.005
    STEPSPEED = 0.02
    SIMCAMS=["wrist_cam_left","wrist_cam_right", "overhead_cam"]