import os
import json
import numpy as np
from scipy.spatial.transform import Rotation 
import cv_utils

def get_swisscube_keypoints():
    keypoints = np.array([
        [-331113, -50545, 28775],
        [312419, -52650, -129375],
        [232469, -49250, 126875],
        [330069, -49250, 126875],
        [331269, 49549, 126875],
        [232469, 50750, 126875],
        [232469, -49250, 15875],
        [330069, -49250, 15875],
        [330069, 50750, 15875],
        [231269, 49549, 15875]
    ])
    # Onshape moves the origin frustratingly
    offset_vector = np.array([
        (283019+279528)/2,
        1464.2,
        71375
    ])
    keypoints = keypoints-offset_vector
    # Units in mm while we expect m
    keypoints_m = keypoints / (1000**2)
    return keypoints_m

def get_airbus_keypoints():
    """
    Returns the handpicked keypoints for the Airbus dataset as a numpy array.

    The keypoints are converted from their original units (assumed to be in mm)
    to meters, and any required offsets can be added below.
    """
    old_keypoints = np.array([
        [48, -150, 32], #0
        [-48, -150, 32],#1
        [-48, 150, 32],#2
        [48, 150, 32],#3
        [48, -149, -43],#4
        [-48, -149, -43],#5
        [-48, 150, -24],#6
        [48, 150, -24],#7
        [170, -57.68, -53.68],#8
        [-170, -57.68, -53.68],#9
        [-170, -14, -32],#10
        [170, -14, -32],#11
        [-38, 0, 52],#12
        [6, 168, -29.3],#13
        [0, -122, 32],#14
        [-33, 90, -52]#15
    ])
    new_interior_kps = \
        [
            [
                -38.562981349241085,
                -58.675975449879274,
                -26.043238039266498
            ],
            [
                27.180319328236322,
                145.97236413150176,
                -29.45654029820852
            ],
            [
                152.77222574968226,
                -43.78254136048497,
                -44.59678267565188
            ],
            [
                -22.718095076765906,
                51.222005425052544,
                32.23576720124868
            ],
            [
                36.07431123648729,
                -150.94825818620063,
                -20.00192533026587
            ],
            [
                -141.5526048784538,
                -55.3429741238336,
                -55.570505490746065
            ],
            [
                35.598605126531794,
                11.941874617500332,
                -33.34923517508748
            ],
            [
                -41.335185314367976,
                -138.06096812777457,
                20.38839329350627
            ],
            [
                27.56496334595886,
                -79.56127301467201,
                28.661188388477612
            ],
            [
                -48.49815499804893,
                108.01366876803081,
                -17.774989619018996
            ],
            [
                72.37496941888134,
                -55.342943047382064,
                -51.22115846943848
            ],
            [
                33.65238854702943,
                86.15142411371158,
                16.74221104523356
            ]
            ]
    # They are rotated from the 3D model
    rel_rot = Rotation.from_euler("zyx",[0,0,np.pi/2])
    keypoints = rel_rot.apply(new_interior_kps)
    
    # Convert units from mm to meters (if needed)
    keypoints_m = keypoints / 1000.0

    return keypoints_m

def get_urso_keypoints():
    keypoints = [[ -0.71362391, -14.54631772,  24.19320329],
       [-13.10359625,   6.37924592,  59.17365457],
       [  4.7355621 , -13.7027351 ,  24.38852812],
       [ 14.90181027,  -2.91704135, 118.39920778],
       [ 13.73380317,  -3.97265318,  22.51637237],
       [ 14.61588334,  -8.54422369,  15.63742796],
       [ -1.93913102, -19.01668725,  39.64490589],
       [  4.34983766,   7.33871215,  66.05660989],
       [  6.19697642,  -0.31165698,  60.64222991],
       [-13.68952837,   7.05421578,  37.01900278]]
    return keypoints


def write_keypoints_to_file(keypoints, object_name):
    # Save keypoints to a JSON file
    if type(keypoints) == np.ndarray:
        keypoints_list = keypoints.tolist()
    else:
        keypoints_list = keypoints
    output_path = f"assets/keypoints/{object_name}_keypoints.json"
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir,exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(keypoints_list, f, indent=2)
    print(f"Keypoints saved to {output_path}")

if __name__ == "__main__":
    #write_keypoints_to_file(get_swisscube_keypoints(),'swisscube')
    write_keypoints_to_file(get_urso_keypoints(),'urso_interior')