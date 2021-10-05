import torch.utils.data as data
import numpy as np
from torch.nn import functional as F
from utils import *
from option import opt
import random
from PIL import Image, ImageOps
from gaussian_downsample import *
def modcrop(img,scale):
    (iw, ih) = img.size
    ih = ih - (ih % scale)
    iw = iw - (iw % scale)
    img = img.crop((0,0,iw,ih))
    #print(img.size)
    return img

def load_img(path,scale,image_pad):
    #需要return hr和帧的数量
    HR=[]
    frames=os.listdir(os.path.join(path,"truth"))
    le=len(frames)
    print("framesnumber",le)
    #st=random.randint(0,le-opt.frame_num)#generate random int between a and b(including a and b)
    for i in range(le):
        #print(os.path.join(path,"%03d.png" % (st+i)))
        GT_temp = modcrop(Image.open(os.path.join(path,"truth","%03d.png" % (i))).convert('RGB'), scale)
        HR.append(GT_temp)
    #testshape = [np.asarray(H) for H in HR]
    #testshape = np.asarray(testshape)
    #testshape=np.asarray(testshape)
    #print(testshape.shape)
    return HR,le



class EVAL(data.Dataset):  # load train dataset
    def __init__(self, image_dir, scale, eval_name, transform):
        super(EVAL, self).__init__()
        self.eval_dir=os.path.join(image_dir,eval_name)#这个是到eval，下面就是所有的video
        val_lists=[]
        for item in os.listdir(self.eval_dir):
            val_lists.append(os.path.join(self.eval_dir,item))
        self.image_filenames=val_lists
        #L = os.listdir(self.image_filenames)
        #self.L = len(L)
        self.scale = scale
        self.transform = transform  # To_tensor
        #load_img(self.image_filenames[0], self.scale, image_pad=True)

    def __getitem__(self, index):
        GT, le = load_img(self.image_filenames[index], self.scale, image_pad=True)
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
        GT = GT.transpose(1, 2, 3, 0).reshape(h, w, -1)  # numpy, [H',W',CT]
        if self.transform:
            GT = self.transform(GT)  # Tensor, [CT',H',W']
        GT = GT.view(c, t, h, w)  # Tensor, [C,T,H,W]
        LR = gaussian_downsample(GT, self.scale)
        LR = LR.permute(1, 0, 2, 3)
        GT = GT.permute(1, 0, 2, 3)  # [T,C,H,W]
        LR_S = F.interpolate(LR, scale_factor=0.5, mode='bilinear', align_corners=False)
        LR_S = F.interpolate(LR_S, scale_factor=2, mode='bilinear', align_corners=False)
        LR_D = LR - LR_S

        GT_S = F.interpolate(GT, scale_factor=0.5, mode='bilinear', align_corners=False)
        GT_S = F.interpolate(GT_S, scale_factor=2, mode='bilinear', align_corners=False)
        GT_D = GT - GT_S

        return LR, LR_D, LR_S, GT, GT_D, GT_S ,le

    def __len__(self):
        return len(self.image_filenames)  # total video number. not image number

