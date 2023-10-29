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


# this is the folder where your test images are 
pth_images_test = "./data_set/images/val2016/"

# this is the folder where the results will be written to
resfolder = './data_set/test_results/'
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

                # The resulting extent mask
                cv2.imwrite(resfolder + im_name + '_resExtent.png',255.0*(seg[1,:,:]>0))

                # The resulting edge mask
                cv2.imwrite(resfolder + im_name+ '_resEdge.png',255.0*(edges[1,:,:]>0))
                final_res = final_res.astype('uint8')
                # this is the postprocessed result (close parcels)
                cv2.imwrite(resfolder + im_name+ '_final_res.png',255.0*final_res)
                if ii>5:
                    break