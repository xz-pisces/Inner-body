import glob
import os
import cv2
import numpy as np
import torch
from PIL import Image
# from torch import optim
import scipy.io as scio
from torch import optim, nn
from torchvision import transforms
from tqdm import tqdm
from network import Unet_2
import make_heatmap

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--input_dir',default='../sample/4001_00', type=str)
parser.add_argument('--model',default='./weights/sunet.pkl', type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
data_path = args.input_dir                            
savevis_path = os.path.join(data_path, 'inner_seg')
savemask_path = os.path.join(data_path, 'inner')
transform = transforms.Compose([transforms.ToTensor()])
model = Unet_2(input_dim=48).cuda()
model = nn.DataParallel(model, device_ids=range(1))
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
optimizer = optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
current_epoch = checkpoint['epoch']
sum_step = checkpoint['sum_step']
torch.set_grad_enabled(False)
model.eval()

piclist = list(x for x in glob.glob(os.path.join(data_path , 'cloth/*.png')))
piclist.sort()
for index in tqdm(range(len(piclist))):
    cloth_image = Image.open(piclist[index])
    cloth = transform(cloth_image).type(torch.FloatTensor).unsqueeze(0).cuda()
    name = piclist[index].split('/')[-1].split('.')[0]

    seg_image = np.load(os.path.join(data_path , 'seg_lip/' , name + '.npy'))
    seg = torch.tensor(seg_image).type(torch.FloatTensor).unsqueeze(0).cuda()

    joint = make_heatmap.make_heatmap(os.path.join(data_path , 'heatmap/' , name + '_keypoints.json'), [512, 512]).unsqueeze(
        0).cuda()

    guide = torch.cat((seg, joint), 1)

    output = model(cloth, guide)
    
    if os.path.exists(savevis_path) == False:
        os.makedirs(savevis_path)
    if os.path.exists(savemask_path) == False:
        os.makedirs(savemask_path)

    mask = np.array(output.detach().cpu() * 255)
    # mask = np.round(mask)
    mask = mask.squeeze()
    mask = cv2.cvtColor(np.asarray(mask),cv2.COLOR_RGB2BGR) 
    a,mask = cv2.threshold(mask, 128, 255,0)
    # print(mask.shape)
    cv2.imwrite(os.path.join(savemask_path, name+'.png'),mask)
    visualize_lip_path = os.path.join(data_path , 'visualize_lip/',name+'.png')
    visualize_lip = cv2.imread(visualize_lip_path)
    # print(visualize_lip.shape)
    cv2.imwrite(os.path.join(savevis_path, name+'.png'),mask/255*visualize_lip) 
