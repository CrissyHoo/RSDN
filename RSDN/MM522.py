import os.path

import torch.utils.data as data
import numpy as np
from torch.nn import functional as F
from utils import *
from option import opt
import torchvision
import random
from PIL import Image, ImageOps
from gaussian_downsample import *
def modcrop(img,scale):
    crop=opt.crop_size*scale
    (iw, ih) = img.size
    #print("iw,ih",iw,ih)
    #print("crop",crop)
    rh = random.randint(0,ih-crop)
    rw = random.randint(0,iw-crop)
    #print("rh,rw",rh,rw)
    img = img.crop((rw,rh,rw+crop,rh+crop))
    #print(img.size)
    return img

def load_img(path,scale,image_pad,index):
    #需要return hr和帧的数量
    HR=[]
    frames=os.listdir(os.path.join(path,"truth"))
    le=len(frames)
    #print("framesnumber",le)
    st=random.randint(0,le-opt.frame_num)#generate random int between a and b(including a and b)
    for i in range(opt.frame_num):
        #print(os.path.join(path,"%03d.png" % (st+i)))
        GT_temp = modcrop(Image.open(os.path.join(path,"truth","%03d.png" % (st+i))).convert('RGB'), scale)
        HR.append(GT_temp)
    #HR[0].convert('RGB').save("./down/hr2rgb{}.png".format(index))#正常
    #HR[0].save("./down/hr2{}.png".format(index))#正常
    #testshape = [np.asarray(H) for H in HR]
    #testshape = np.asarray(testshape)
    #testshape=np.asarray(testshape)
    #print(testshape.shape)
    return HR,opt.frame_num


#MM522数据集有些特殊，应该每个子文件夹内的数据大小不一样，所以要分开处理，或者说直接处理成小patch，建议后一种办法
class MM522(data.Dataset):  # load train dataset
    def __init__(self, image_dir, scale, train_name, transform):
        super(MM522, self).__init__()
        self.train_dir=os.path.join(image_dir,train_name)#这个是到mm522，下面还有子目录
        train_list = ["HP", "LCF", "PE_S2E1", "PE_S2E2", "PE_S2E3", "WBC_E1", "WBC_E2", "WBC_E3", "WBC_E4"]
        video_dir=[]
        self.image_filenames=[]
        for item in train_list:
            video_dir.append(os.path.join(self.train_dir,item))
        for video in video_dir:
            videos=os.listdir(video)
            for video_name in videos:
                self.image_filenames.append(os.path.join(video,video_name))
        #L = os.listdir(self.image_filenames)
        #self.L = len(L)
        self.scale = scale
        self.transform = transform  # To_tensor
        #load_img(self.image_filenames[0], self.scale, image_pad=True)

    def __getitem__(self, index):
        GT, L = load_img(self.image_filenames[index], self.scale, image_pad=True,index=index)
        GT = [np.asarray(HR) for HR in GT]
        GT = np.asarray(GT)
        #print("GTshape",GT.shape)
        #if self.scale == 4:
         #   GT = np.lib.pad(GT, pad_width=((0, 0), (2 * 4, 2 * 4), (2 * 4, 2 * 4), (0, 0)), mode='reflect')
        t = GT.shape[0]
        h = GT.shape[1]
        w = GT.shape[2]
        c = GT.shape[3]
        #print("thwc",t,h,w,c)
        #print("saveHRimage")
        #Image.fromarray(GT[0]).convert('RGB').save("./down/HR{}.png".format(index))#这个也是正常的
        #cv2.imwrite("./down/MMCV2{}.png".format(index), GT[0], [cv2.IMWRITE_PNG_COMPRESSION, 0])
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]
        LR = gaussian_downsample(GT, self.scale,index)
        #print("LRsize********************",LR.size())[3,14,72,72]
        #print("saving down image",LR[0].permute(1,2,3,0).size())
        #cv2.imwrite(os.path.join("./down{}.png".format(index)), np.ascontiguousarray(LR.permute(1,2,3,0)[0]), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        #tmp = np.ascontiguousarray(LR.permute(1, 2, 3, 0)[0])
        #print("saveLRimage")
        #Image.fromarray(np.uint8(tmp * 255)).convert('RGB').save("./down/LR{}.png".format(index))
        LR = LR.permute(1, 0, 2, 3)
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]
        LR_S = F.interpolate(LR, scale_factor=0.5, mode='bilinear', align_corners=False)
        LR_S = F.interpolate(LR_S, scale_factor=2, mode='bilinear', align_corners=False)
        LR_D = LR - LR_S

        GT_S = F.interpolate(GT, scale_factor=0.5, mode='bilinear', align_corners=False)
        GT_S = F.interpolate(GT_S, scale_factor=2, mode='bilinear', align_corners=False)
        GT_D = GT - GT_S

        return LR, LR_D, LR_S, GT, GT_D, GT_S ,L

    def __len__(self):
        return len(self.image_filenames)  # total video number. not image number

