from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_test_set #, get_eval_set
import time
import torch.backends.cudnn as cudnn
import cv2
import math
import sys
import datetime
from utils import Logger
import numpy as np
import torchvision.utils as vutils

#from tensorboardX import SummaryWriter
import time
from torch.nn import functional as F
import utils
import matplotlib.pyplot as plt
import data
import torch.optim.lr_scheduler as lrs
import torch.optim as optim
import decimal
from loss import L1_Charbonnier_loss
from test import *
import warnings
from numpy import *
import gc
from option import opt,systime,gpus_list
import model

class Trainer:
    def __init__(self,opt,model,train_loader):
        self.opt=opt
        self.device = torch.device('cuda')
        self.train_loader = train_loader
        self.model=model
        self.optimizer = self.make_optimizer()
        self.scheduler=self.make_scheduler()
        self.loss= L1_Charbonnier_loss()
        self.max_psnr=0



    def make_optimizer(self):
        #bata值的设置和默认的一样
        return optim.Adam(self.model.parameters(),lr=self.opt.initial_lr)

    def make_scheduler(self):
        '''
        The learning rate is initially set to 1 × 10−4
        and is later down-scaled by a factor of 0.1 every
        60 epoch till 70 epochs.
        '''
        return lrs.MultiStepLR(self.optimizer,[60],0.9,last_epoch=-1)

    def train(self):
        batch_out=[[],[],[]]
        self.scheduler.step()
        l_r = self.scheduler.get_lr()[0]
        print('Epoch {:3d} with Lr {:.2e}'.format(self.scheduler.last_epoch, decimal.Decimal(l_r)))
        self.model.train()
        for batch, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            #for name, parameters in self.model.named_parameters():  # 打印出每一层的参数的大小
             #   print(name, ':', parameters)
              #  break
            # print("LR",LR.shape)#[B,T,C,H,W][16,7,3,68,116]每次返回了一整个batch，
            # print("L",L)#注意这里的L也是有batch个
            LR, LR_D, LR_S, GT, GT_D, GT_S=data[0],data[1],data[2],data[3],data[4],data[5]
            
            loop =LR.shape[0]
            #print("loop:",loop)#应该是16
            psnr_sum = 0
            ssim_sum = 0

            GT=GT.to(self.device)
            GT_D=GT_D.to(self.device)
            GT_S=GT_S.to(self.device)
            LR=LR.to(self.device)
            LR_D = LR_D.to(self.device)
            LR_S = LR_S.to(self.device)
            SR,SR_D,SR_S=self.model(LR,LR_D,LR_S)

            #print("data grad",SR.grad)
            loss=self.loss(SR,GT)+self.loss(SR_D,GT_D)+self.loss(SR_S,GT_S)
            #loss.retain_grad()
            #print("loss",loss)
            #print(SR.shape,GT.shape)#[8,7,3,h,w]
            loss.backward()
            #print("lossgrad", loss.grad)
            self.optimizer.step()
            for batch_num in range(loop):
                SR_t=SR[batch_num].permute(0, 2, 3, 1)
                GT_t=GT[batch_num].permute(0,2,3,1)
                SR_t=SR_t.detach().cpu().numpy()[:,:,:,::-1]# tensor rgb -> bgr
                GT_t=GT_t.detach().cpu().numpy()[:,:,:,::-1]
                for frame_num in range(SR.shape[1]):
                    #应该就是L的值，但这样写也可以
                    SR_frame=bgr2ycbcr(SR_t[frame_num])*255
                    GT_frame=bgr2ycbcr(GT_t[frame_num])*255
                    #preparation well, then we calculate psnr and ssim
                    psnr_sum+=calculate_psnr(SR_frame,GT_frame)
                    ssim_sum+=calculate_ssim(SR_frame,GT_frame)
            #这样就计算完了8个video的总psnr和总ssim,我们需要得到平均的值
            psnr_sum/=(self.opt.trainbatchsize*self.opt.num_frame)
            ssim_sum/=(self.opt.trainbatchsize*self.opt.num_frame)
            batch_out[0].append(psnr_sum)
            batch_out[1].append(ssim_sum)
            batch_out[2].append(loss.item())
            if batch%100==0:
                print("batch", batch)
                print("in every 100 batch: psnr:", mean(batch_out[0]), "ssim:", mean(batch_out[1]), "loss:", mean(batch_out[2]))
            print("in one batch: psnr:", mean(batch_out[0]), "ssim:", mean(batch_out[1]), "loss:", mean(batch_out[2]))

        psnr=mean(batch_out[0])#epoch内的均值
        ssim=mean(batch_out[1])
        print("in one epoch：psnr:",psnr,"ssim:",ssim,"loss:",mean(batch_out[2]))
        return psnr,ssim,batch_out

    def plot_graph(self,name,data):
        axis=np.linspace(1,self.opt.epoch,self.opt.epoch)
        fig=plt.figure()
        plt.title("{} graph".format(name))
        plt.plot(axis,data)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(name)
        plt.grid(True)
        plt.savefig(os.path.join(self.opt.image_out, '{}.pdf'.format(name)))
        plt.close(fig)

    def validate(self):
        print("evaluation")
        self.model.eval()
        psnr_sum=0
        with torch.no_grad():
            for batch, data in enumerate(self.loader_val):
                LR, LR_D, LR_S, GT, GT_D, GT_S = data[0], data[1], data[2], data[3], data[4], data[5]
                GT=GT.to(self.device)
                LR = LR.to(self.device)
                LR_D = LR_D.to(self.device)
                LR_S = LR_S.to(self.device)
                SR, SR_D, SR_S = self.model(LR, LR_D, LR_S)
                PSNR=calculate_psnr(SR,GT)
                psnr_sum+=PSNR
            psnr_sum /= (self.opt.trainbatchsize * self.opt.num_frame)
            print("in evaluation, the psnr is:",psnr_sum)
            if psnr_sum>self.max_psnr:
                self.max_psnr=psnr_sum
                self.model.save(self,os.path.join(self.opt.image_out,"model"),filename="rsdn")







def main():
    sys.stdout = Logger(os.path.join(opt.save_train_log, 'train_' + systime + '.txt'))
    if not torch.cuda.is_available():
        raise Exception('No Gpu found, please run with gpu')
    else:
        use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = False
        torch.cuda.manual_seed(opt.seed)
    pin_memory = True if use_gpu else False
    print('===> Loading train Datasets')
    train_set = data.get_train_set(opt)  # 一下子获取到了所有的set
    train_loader = DataLoader(train_set, batch_size=opt.trainbatchsize, shuffle=True, pin_memory=pin_memory,
                              num_workers=opt.threads)
    print('===> Loading train Datasets finished')
    #rsdn = RSDN9_128(opt)  # initial filter generate network 加载好了最初的模型
    #rsdn = torch.nn.DataParallel(rsdn, device_ids=gpus_list)

    #if use_gpu:
     #   rsdn = rsdn.to(torch.device('cuda'))
    rsdn=model.Model(opt)

    #在训练过程中需要记录这些参数
    PSNR = []
    SSIM = []
    LOSS=[]


    t = Trainer(opt, rsdn,train_loader)
    #开始训练，进行epoch代训练
    while t.scheduler.last_epoch<opt.epoch:
        psnr, ssim, out = t.train()
        t.validate()
        PSNR.append(psnr)
        SSIM.append(ssim)
        LOSS.append(mean(out[2]))
    t.plot_graph("psnr",PSNR)










if __name__=='__main__':
    #utils.get_lr_frames()
    warnings.filterwarnings('ignore')
    main()