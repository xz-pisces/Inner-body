import cv2
import json
import numpy as np
import tensorflow as tf
import keras.backend as K
from typing import Dict
from smpl.batch_smpl import batch_rodrigues

# LABELS_FULL = {
#     'Sunglasses': [170, 0, 51],
#     'LeftArm': [51, 170, 221],
#     'RightArm': [0, 255, 255],
#     'LeftLeg': [85, 255, 170],
#     'RightLeg': [170, 255, 85],
#     'LeftShoe': [255, 255, 0],
#     'RightShoe': [255, 170, 0],
# }
#
# LABELS_CLOTHING= {
#     'Face': [0, 0, 255],
#     'Arms': [51, 170, 221],
#     'Legs': [85, 255, 170],
#     'Shoes': [255, 255, 0]
# }

# def read_segmentation(file):
#     segm = cv2.imread(file)[:, :, ::-1]
#
#     segm[np.all(segm == LABELS_FULL['Sunglasses'], axis=2)] = LABELS_CLOTHING['Face']
#     segm[np.all(segm == LABELS_FULL['LeftArm'], axis=2)] = LABELS_CLOTHING['Arms']
#     segm[np.all(segm == LABELS_FULL['RightArm'], axis=2)] = LABELS_CLOTHING['Arms']
#     segm[np.all(segm == LABELS_FULL['LeftLeg'], axis=2)] = LABELS_CLOTHING['Legs']
#     segm[np.all(segm == LABELS_FULL['RightLeg'], axis=2)] = LABELS_CLOTHING['Legs']
#     segm[np.all(segm == LABELS_FULL['LeftShoe'], axis=2)] = LABELS_CLOTHING['Shoes']
#     segm[np.all(segm == LABELS_FULL['RightShoe'], axis=2)] = LABELS_CLOTHING['Shoes']
#
#     return segm[:, :, ::-1] / 255.

LABELS_FULL = {
    'Hat': [128,0,0],
    'Hair': [255,0,0],
    'Glove': [0,85,0],

    'Sunglasses': [170, 0, 51],

    'Upper-clothes':[255,85,0],
    'Dress':[0,0,85],
    'Coat':[0,119,221],
    'Socks':[85,85,0],
    'Pants':[0,85,85],
    'Torso-Skin':[85,51,0],
    'Scarf':[52,86,128],
    'Skirt':[0,128,0],
    'Face':[0,0,255],

    'LeftArm': [51, 170, 221],
    'RightArm': [0, 255, 255],

    'LeftLeg': [85, 255, 170],
    'RightLeg': [170, 255, 85],

    'LeftShoe': [255, 255, 0],
    'RightShoe': [255, 170, 0],
}

LABELS_CLOTHING= {
    'Hair': [255,0,0],
    'Face': [0, 0, 255],
    'Torso-Skin':[85,51,0],
    'Upper-clothes':[255,85,0],
    'Pants':[0,85,85],
    'Arms': [51, 170, 221],
    'Legs': [85, 255, 170],
    'Shoes': [255, 255, 0]
}


def read_segmentation(file, CONFIG_SIZE:(Dict[str, int] or None)==None):
    segm = cv2.imread(file, 1)[:, :, ::-1]
    if CONFIG_SIZE is not None :
        # assert segm.shape[0] == segm.shape[1] and segm.shape[0] == CONFIG_SIZE['src']
        assert CONFIG_SIZE['dist'] == 1080
        segm = cv2.resize(segm, dsize=(CONFIG_SIZE['dist'], CONFIG_SIZE['dist']), interpolation=cv2.INTER_CUBIC)
        # print(segm.shape)
    else :
        assert segm.shape[0] == segm.shape[1] and segm.shape[0] == 1080

    LABELS_CLOTHING = {
        'Face': [0, 0, 255],
        'Arms': [51, 170, 221],
        'Legs': [85, 255, 170],
        'Shoes': [255, 255, 0],
        'lxz': [255, 255, 255]
    }
    segm[np.any(segm > 0, axis=2)] = LABELS_CLOTHING['lxz']

    return segm[:, :, ::-1] / 255.

def openpose_from_file(
        file,
        CONFIG_SIZE:Dict[str, int] or None=None,
        resolution=(1080, 1080),
        person=0
    ):
    with open(file) as f:
        data = json.load(f)['people'][person]

        pose = np.array(data['pose_keypoints_2d']).reshape(-1, 3)
        if CONFIG_SIZE is not None :
            assert CONFIG_SIZE['dist'] == 1080
            pose[:, 0:2] = pose[:, 0:2] / CONFIG_SIZE['src'] * CONFIG_SIZE['dist']
        pose[:, 2] /= np.expand_dims(np.mean(pose[:, 2][pose[:, 2] > 0.1]), -1)
        pose = pose * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
        pose[:, 0] *= 1. * resolution[1] / resolution[0]

        face = np.array(data['face_keypoints_2d']).reshape(-1, 3)
        if CONFIG_SIZE is not None :
            assert CONFIG_SIZE['dist'] == 1080
            face[:, 0:2] = face[:, 0:2] / CONFIG_SIZE['src'] * CONFIG_SIZE['dist']
        face = face * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
        face[:, 0] *= 1. * resolution[1] / resolution[0]

        return pose, face









def get_img1(path_list):
    imgs = []
    for p in path_list:
        img=read_segmentation(p)
        imgs.append(img)
    #print("imgs:",imgs)
    return np.array(imgs).reshape(len(path_list), 1080, 1080, 3)

def get_img(path):
    img=read_segmentation(path)
    # print('size:',type(img))
    return img
    # imgs.append(img)
    #print("imgs:",imgs)
    # return np.array(imgs).reshape(len(path_list), 1080, 1080, 3)



def get_rendered(path):
    img=read_segmentation(path)
    # print('size:',type(img))
    return np.expand_dims(
                    np.any(np.float32(img.reshape((1080, 1080, -1)) > 0), axis=-1),
                    -1)
    
def get_2dKeypoints1(path_list):
    joints=[]
    faces=[]
    for p in path_list:
        j, f=openpose_from_file(p)
        joints.append(j)
        faces.append(f)
    return np.array(joints), np.array(faces)

def get_2dKeypoints(path):
    joints, faces=openpose_from_file(path)
    return joints,faces

def get_poses(path):
    with open(path,"r") as f:
        data = f.read()  
        pose=np.array(data.split()[3:]).reshape(-1,3).astype(np.float32)
        # pose = tf.reshape(batch_rodrigues(pose.reshape(-1, 3).astype(np.float32)), (24, 3, 3 ))
        # pose = K.eval(pose)
        return pose
                    
                    # trans=np.array(a[:3])
                    # poses=np.array(a[3:])
                    # print(trans.shape)
                    # print(poses.shape)

def get_trans(path):
    with open(path,'r') as f:
        data = f.read()
        a = np.array(data.split()).astype(np.float32)
        return a[:3]

def write_mesh(filename, v, f):
    with open(filename, 'w') as fp:
        fp.write(('v {:f} {:f} {:f}\n' * len(v)).format(*v.reshape(-1)))
        fp.write(('f {:d} {:d} {:d}\n' * len(f)).format(*(f.reshape(-1) + 1)))
