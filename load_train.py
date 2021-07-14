import torch.utils.data as data
import os.path as osp
import os
from load_test import modcrop
from PIL import Image
import numpy as np
from gaussian_downsample import gaussian_downsample
from torch.nn import functional as F
#dataset for VIMEO90k
class VIMEO90KDataset(data.Dataset):
    def __init__(self,opt,transform):
        """Vimeo90K dataset for training.

            The keys are generated from a meta info txt file.
            meta_info_Vimeo90K_train_GT.txt这样就减少了对目录的重复读取

            Each line contains:
            1. clip name; 2. frame number; 3. image shape, seperated by a white space.
            Examples:
                00001/0001 7 (256,448,3)
                00001/0002 7 (256,448,3)

            Key examples: "00001/0001"
            GT (gt): Ground-Truth;
            LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

            The neighboring frame list for different num_frame:
            num_frame | frame list
                     1 | 4
                     3 | 3,4,5
                     5 | 2,3,4,5,6
                     7 | 1,2,3,4,5,6,7

            """
        super(VIMEO90KDataset, self).__init__()
        self.transform=transform
        self.opt=opt
        self.train_root=osp.join(self.opt.train_dir+self.opt.train_name)
        with open(opt.data_info_path, 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]#00001/0001
            # indices of input images
        self.neighbor_list = [i + 1 for i in range(opt.num_frame)]#[1,2,3,4,5,6,7]

    def __getitem__(self, index):
        key=self.keys[index]
        video_path=osp.join(self.train_root,key)
        GT=[]
        for image_num in range(self.opt.num_frame):
            #print(image_num)
            image=osp.join(video_path,"im{}.png".format(self.neighbor_list[image_num]))
            #print(image)
            GT_temp = modcrop(Image.open(image).convert('RGB'), self.opt.scale)
            GT.append(GT_temp)
        GT = [np.asarray(HR) for HR in GT]
        GT = np.asarray(GT)
        if self.opt.scale == 4:
            GT = np.lib.pad(GT, pad_width=((0, 0), (2 * 4, 2 * 4), (2 * 4, 2 * 4), (0, 0)), mode='reflect')
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]
        LR = gaussian_downsample(GT, self.opt.scale)
        LR = LR.permute(1, 0, 2, 3)
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]
        #print("GTshape:",GT.shape)
        LR_S = F.interpolate(LR, scale_factor=0.5, mode='bilinear', align_corners=False)
        LR_S = F.interpolate(LR_S, scale_factor=2, mode='bilinear', align_corners=False)
        LR_D = LR - LR_S

        GT_S = F.interpolate(GT, scale_factor=0.5, mode='bilinear', align_corners=False)
        GT_S = F.interpolate(GT_S, scale_factor=2, mode='bilinear', align_corners=False)
        GT_D = GT - GT_S

        return LR, LR_D, LR_S, GT, GT_D,GT_S
    def __len__(self):
        return len(open(self.opt.data_info_path,'rU').readlines())