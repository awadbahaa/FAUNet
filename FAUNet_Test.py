import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ctypes
from torch.autograd import Variable
import json
import cv2
from torch.utils import data
from pycocotools.coco import COCO
import math
from torch.utils.data import DataLoader
from unet_utils.dice_score import dice_loss


from FAUNet_Train import channel_normalize
from FAUNet_Train import myDataset
from FAUNet_Train import DoubleConv,Down,Up,OutConv
from FAUNet_Train import FAUNet
from FAUNet_Train import mynormalize

import os

Device = 'cuda:0'    

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


main_path = './data_set/'

epoch = 96

### set paths
weights_folder = "./FAUNET_weights/"
PATH = weights_folder + "/weights_" + str(epoch) + "epoch_final.pt"
#print('loading model: ', PATH)
# resfolder = './unet_res_lowres_full/'

#####
# faunet = FAUNet(3,2,Device)
faunet = torch.load(PATH)
# faunet.load_state_dict(PATH)
faunet.eval()
faunet.to(torch.device(Device))



pth_images_test = "./new_test_denmark_inv/images/val2023/"
resfolder = './new_test_denmark_inv/test/val/'
os.makedirs(resfolder,exist_ok= True)


file_list = os.listdir(pth_images_test)

for ii,file_ in enumerate(file_list):
    print(ii,file_)
    with torch.no_grad():
        print(ii + 1,'out of: ', 100)
        im_name = str(ii) 
        img_dir = file_ 
        im = cv2.imread(pth_images_test + '/' + img_dir)
        print(im.max())
        
        im_h,im_w,im_c = im.shape
        if im_h == 256:
            if im_h == 256:
                im_norm = mynormalize(im)
                im_transposed = np.transpose(im_norm,(2,0,1))
                im_tensor = torch.tensor(im_transposed).float()     
                x =  im_tensor.to(torch.device(Device)).unsqueeze(0)
                print(x.size())
                c,e = faunet(x)
                seg = c.squeeze(0).cpu().numpy()
                edges = e.squeeze(0).cpu().numpy()       
                
                final_res =   1.0*(seg[1,:,:]>0)* (1 - 1.0*(edges[1,:,:]>0))
                
                # edges = 1.0*(edges[1,:,:]>0)
                cv2.imwrite(resfolder + im_name+ '.png',255*im_norm)
                # cv2.imwrite(resfolder + sim_name + '_gt_edge.png',sample[1]['edgemasks'].squeeze(0).cpu().numpy()*255)
                # cv2.imwrite(resfolder + im_name + '_gt_mask.png',sample[1]['masks'].squeeze(0).cpu().numpy()*255)
                # cv2.imwrite(resfolder + im_name + '_gt.png',GROUNTRUTH*255)
                cv2.imwrite(resfolder + im_name + '_res.png',255.0*(seg[1,:,:]>0))
                cv2.imwrite(resfolder + im_name+ '_resedge.png',255.0*(edges[1,:,:]>0))
                final_res = final_res.astype('uint8')
                cv2.imwrite(resfolder + im_name+ '_zfinal_res.png',255.0*final_res)
                if ii>100:
                    break