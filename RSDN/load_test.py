import os
import torch.utils.data as data
import torch
import numpy as np
from PIL import Image, ImageOps
import random
from gaussian_downsample import gaussian_downsample
from bicubic import imresize
from utils import *
from torch.nn import functional as F
def load_img(image_path, scale, L, image_pad):
    char_len = len(image_path)
    HR = []
    for img_num in range(L):
        index = int(image_path[char_len-7:char_len-4]) + img_num
        image = image_path[0:char_len-7]+'{0:03d}'.format(index)+'.png'
        GT_temp = modcrop(Image.open(image).convert('RGB'), scale)
        HR.append(GT_temp)
    return HR, len(HR)



class DataloadFromFolderTest(data.Dataset): # load test dataset vid4
    def __init__(self, image_dir, scale, test_name, transform):
        super(DataloadFromFolderTest, self).__init__()
        alist = [line.rstrip() for line in open(os.path.join("./test_list/vid4",test_name+'.txt'))]
        #[walk,calendar,raw,bla]
        self.image_filenames = [os.path.join(image_dir,x) for x in alist] #4
        L = os.listdir(os.path.join(image_dir, test_name.split('_')[0],"original"))
        self.L = len(L)#L是一个视频中帧的数量
        self.scale = scale
        self.transform = transform # To_tensor
    def __getitem__(self, index):
        GT, L = load_img(self.image_filenames[index], self.scale, self.L, image_pad=True) 
        GT = [np.asarray(HR) for HR in GT] 
        GT = np.asarray(GT)
        if self.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0,0), (2*4,2*4), (2*4,2*4), (0,0)), mode='reflect')
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        #print("saveGTimage")
        #Image.fromarray(np.uint8(GT[0]*255)).convert('RGB').save("./down/GT{}.png".format(index))
        GT = GT.transpose(1,2,3,0).reshape(h,w,-1) # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT) # Tensor, [CT',H',W']
        GT = GT.view(c,t,h,w) # Tensor, [C,T,H,W]

        LR = gaussian_downsample(GT, self.scale,index)
        #print("testLRsize",LR.permute(1, 2, 3, 0)[0].size())
        #tmp=np.ascontiguousarray(LR.permute(1, 2, 3, 0)[0])
        #print("saveLRimage")
        #Image.fromarray(np.uint8(tmp*255)).convert('RGB').save("./down/LR{}.png".format(index))
        LR = LR.permute(1,0,2,3)
        GT = GT.permute(1,0,2,3) # [T,C,H,W]
        LR_S = F.interpolate(LR, scale_factor=0.5, mode='bilinear', align_corners=False)
        LR_S = F.interpolate(LR_S, scale_factor=2, mode='bilinear', align_corners=False)
        LR_D = LR - LR_S

        GT_S = F.interpolate(GT, scale_factor=0.5, mode='bilinear', align_corners=False)
        GT_S = F.interpolate(GT_S, scale_factor=2, mode='bilinear', align_corners=False)
        GT_D = GT - GT_S

        return LR, LR_D, LR_S, GT, L
        
    def __len__(self):
        return len(self.image_filenames) # total video number. not image number


