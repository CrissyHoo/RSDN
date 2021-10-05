import torch.utils.data as data
import os.path as osp
import os
from load_test import modcrop
from PIL import Image
import numpy as np
from gaussian_downsample import gaussian_downsample
from torch.nn import functional as F
import random
#dataset for VIMEO90k
class VIMEO90KDataset(data.Dataset):
    def __init__(self,opt,transform,mode):
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
        self.mode=mode
        if mode=="train":
            self.train_root=osp.join(self.opt.train_dir+self.opt.train_name)
            with open(opt.data_info_path, 'r') as fin:
                self.keys = [line.split(' ')[0] for line in fin]  # 00001/0001
        elif mode=="val":
            self.train_root=self.opt.val_dir
            with open(opt.data_info_path, 'r') as fin:
                keys_t=[line.split(' ')[0] for line in fin]
                val_list=random.sample(range(0, 64000), 900)
                self.keys=[keys_t[val] for val in val_list]

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
        if self.mode=="train":
            return len(open(self.opt.data_info_path,'rU').readlines())
        elif self.mode=="val":
            return len(self.keys)

def load_frames(path,num_frame,scale,size):
    #从视频中随机crop出小块进行处理
    lr_path=osp.join(path,"input4")
    hr_path=osp.join(path,"truth")
    frames=os.listdir(hr_path)
    max_frame=len(frames)
    #print(path)
    #print(max_frame)
    #print(num_frame)
    st=np.random.randint(0,max_frame-num_frame+1)#随机选择视频的帧数
    LR=[]
    HR=[]
    for i in range(num_frame):
        lrpath=osp.join(lr_path,frames[st+i])
        hrpath=osp.join(hr_path,frames[st+i])
        #GT_temp=modcrop(Image.open(imagepath).convert('RGB'),scale)
        lr=np.array(Image.open(lrpath).convert('RGB'))
        hr=np.array(Image.open(hrpath).convert('RGB'))
        print(lrpath,hrpath)
        LR.append(lr)
        HR.append(hr)
    LR = [np.asarray(lr) for lr in LR]
    LR = np.asarray(LR)
    HR=[np.asarray(hr) for hr in HR]
    HR=np.asarray(HR)
    n, h, w, c = LR.shape
    w0 = np.random.randint(0, w - size + 1)
    h0 = np.random.randint(0, h - size + 1)
    LR = LR[:, h0:h0 + size, w0:w0 + size, :]
    HR = HR[:, h0 * scale:(h0 + size) * scale, w0 * scale:(w0 + size) * scale, :]
    #print(HR.shape)
    return LR,HR



#dataset for MM522
class MM522Dataset(data.Dataset):
    def __init__(self,opt,transform,mode):
        super(MM522Dataset, self).__init__()
        self.transform=transform
        self.opt=opt
        self.mode=mode
        self.opt.train_name="MM522"

        self.val_path=os.path.join(self.opt.data_dir,"eval")
        self.train_list = ["HP", "LCF", "PE_S2E1", "PE_S2E2", "PE_S2E3", "WBC_E1", "WBC_E2", "WBC_E3", "WBC_E4"]
        train_path =[]
        for pa in self.train_list:
            train_path.append(os.path.join(self.opt.data_dir, self.opt.train_name,pa))
        self.train_path=train_path
        train_paths=[]
        eval_paths=[]
        #print("trainpath",self.train_path)
        for path in self.train_path:
            for dir in os.listdir(path):

                train_paths.append(os.path.join(path,dir))
        for dir in os.listdir(self.val_path):
            eval_paths.append(os.path.join(self.val_path,dir))

        if self.mode=="train":
            self.path=train_paths
        elif self.mode=="val":
            self.path=eval_paths
    def __getitem__(self, index):
        #print(self.hr_path[index])
        LR,GT=load_frames(self.path[index],num_frame=self.opt.num_frame,scale=self.opt.scale,size=self.opt.crop_size)
        #if self.opt.scale == 4:#对图像进行padding
        #    GT = np.lib.pad(GT, pad_width=((0, 0), (2 * 4, 2 * 4), (2 * 4, 2 * 4), (0, 0)), mode='reflect')

        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]
        tl = LR.shape[0]
        hl = LR.shape[1]
        wl = LR.shape[2]
        cl = LR.shape[3]
        LR=LR.transpose(1,2,3,0).reshape(hl,wl,-1)
        if self.transform:
            LR=self.transform(LR)
        LR=LR.view(cl,tl,hl,wl)

        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]
        LR = LR.permute(1, 0, 2, 3)
        #print("LR",LR)
        #print("LRshape:",LR.shape)
        #print("GTshape",GT.shape)
        LR_S = F.interpolate(LR, scale_factor=0.5, mode='bilinear', align_corners=False)
        LR_S = F.interpolate(LR_S, scale_factor=2, mode='bilinear', align_corners=False)
        LR_D = LR - LR_S

        GT_S = F.interpolate(GT, scale_factor=0.5, mode='bilinear', align_corners=False)
        GT_S = F.interpolate(GT_S, scale_factor=2, mode='bilinear', align_corners=False)
        GT_D = GT - GT_S

        return LR, LR_D, LR_S, GT, GT_D, GT_S
    def __len__(self):
        return len(self.path)





