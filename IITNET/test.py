import os
import argparse
import tensorflow as tf
import keras.backend as K

from glob import glob
import numpy as np

from lib.iox import openpose_from_file, read_segmentation, write_mesh
from model.octopus_modi import Octopus
from smpl.batch_smpl_np import BATCHSMPLModel,load_joints_pkl
import cv2
import pickle




def main(input_, output_, female):
    if female :
        option_dict = {'data_path': input_,
                       'out_path':  output_,
                       'pkl': 'female_smpl.pkl', 
                       'weights': 'female_x_double.h5'}
    else:
        option_dict = {'data_path':input_,
                       'out_path': output_,
                       'pkl': 'male_smpl.pkl', 
                       'weights': 'male_x_double.hdf5'}

    K.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))
    smpl_gender_pkl_path = 'assets/%s' % option_dict['pkl']
    model = Octopus(batch_size=1, smpl_gender_pkl_path=smpl_gender_pkl_path)
    num_images = 8
    
    path = option_dict['data_path']
    outpath = option_dict['out_path']
    CONFIG_SIZE = {
        'src': 512,
        'dist': 1080
    }
    
    nameList = os.listdir(path)
    nameList.sort()
    for name in nameList:
        print(name)
        segmPath = path+  name+ '/inner_seg/'
        posePath = path+name+ '/heatmap/'
        framePath = path + name + '/cloth/'
        
        segm_files = sorted(glob(os.path.join(segmPath, '*.png')))
        pose_files = sorted(glob(os.path.join(posePath, '*.json')))

        segmentations = [read_segmentation(f, CONFIG_SIZE) for f in segm_files]
        joints_2d, face_2d = [], []
        for f in pose_files:
            # j, f = openpose_from_file(f)
            j, f = openpose_from_file(f, CONFIG_SIZE)

            assert (len(j) == 25)
            assert (len(f) == 70)

            joints_2d.append(j)
            face_2d.append(f)
        outfolder = outpath + name + '/'
        if not os.path.exists(outfolder):
            os.makedirs(outfolder)
        
        type = option_dict['weights']
        model.inference_model.load_weights('weights/' + type, by_name=True)
        pred = model.predict(segmentations, joints_2d)


        print('Optimizing for pose...')
        model.opt_pose(segmentations, joints_2d, name , opt_steps=15)
        print('Optimizing for shape...')
        model.opt_shape(segmentations, joints_2d, face_2d,name, opt_steps=20)
        pred = model.predict(segmentations, joints_2d)



        smpl = BATCHSMPLModel(smpl_gender_pkl_path, num_images)
        beta = np.expand_dims(pred['betas'], axis=0).repeat(num_images, axis=0)
        pose = np.reshape(pred['poses'], [pred['poses'].shape[0], 24, -1])
        trans = pred['trans']
        displacement = np.expand_dims(np.zeros_like(pred['offsets']), axis=0).repeat(num_images, axis=0)
        smpl.set_params(beta=beta,
                        pose=pose,
                        trans=trans,
                        Displayment=displacement)

        if not os.path.exists(outfolder ):
            os.mkdir(outfolder )
        for l in range(num_images):
            write_mesh(outfolder + '/pred_innerbody_{}.obj'.format(l), pred['vertices'][l],
                       pred['faces'])
    print('Done.')


if __name__ == '__main__':

    main('../sample/' ,'../result/', female=True)
