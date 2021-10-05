from __future__ import print_function
import argparse
from math import log10
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import * #, get_eval_set
import time
import torch.backends.cudnn as cudnn
import cv2
import math
import sys
import datetime
from utils import Logger
import numpy as np
import torchvision.utils as vutils
from arch import RSDN9_128
from test import *
from train import *
import time
from torch.nn import functional as F
from option import *
import torch.multiprocessing as mp


def main():
    #writer = SummaryWriter()
    #sys.stdout = Logger(os.path.join(opt.save_test_log,'test_'+systime+'.txt'))
    #if not torch.cuda.is_available():
     #   raise Exception('No Gpu found, please run with gpu')
    #else:
     #   use_gpu = torch.cuda.is_available()
    #if use_gpu:
     #   cudnn.benchmark = False
      #  torch.cuda.manual_seed(opt.seed)


    #pin_memory = True if use_gpu else False
    if opt.seed is not None:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        cudnn.deterministic=True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    # Selecting network
    rsdn = RSDN9_128(4) # initial filter generate network
    #print(rsdn)
    #print("Model size: {:.5f}M".format(sum(p.numel() for p in rsdn.parameters())*4/1048576))
    #rsdn = torch.nn.DataParallel(rsdn, device_ids=gpus_list)
    print('===> load pretrained model')
    if opt.load_pretrain and os.path.isfile(opt.pretrain):
        rsdn.load_state_dict(torch.load(opt.pretrain, map_location=lambda storage, loc: storage),False)
        #rsdn.load_state_dict(torch.load(opt.pretrain))
        print('===> pretrained model is load')
    torch.cuda.set_device('cuda:{}'.format(gpus_list[0]))
    rsdn.cuda()
    rsdn=torch.nn.DataParallel(rsdn,device_ids=gpus_list,output_device=gpus_list[0])
    #if use_gpu:
     #   rsdn = rsdn.cuda(gpus_list[0])
    if opt.mode=="test":
        print('===> Loading test Datasets')
        PSNR_avg = 0
        SSIM_avg = 0
        count = 0
        out = []
        test_list = ['foliage_r.txt','walk_r.txt','city_r.txt','calendar_r.txt']
        for test_name in test_list:
            test_set = get_test_set(opt.test_dir, opt.scale, test_name.split('.')[0])
            test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testbatchsize, shuffle=False, pin_memory=True, drop_last=False)
            print('===> DataLoading Finished')
            PSNR, SSIM, out = test(test_loader, rsdn, test_name.split('.')[0], out)
            PSNR_avg += PSNR
            SSIM_avg += SSIM
            count += 1


        PSNR_avg = PSNR_avg/len(test_list)
        SSIM_avg = SSIM_avg/len(test_list)
        print('==> Average PSNR = {:.6f}'.format(PSNR_avg))
        print('==> Average SSIM = {:.6f}'.format(SSIM_avg))
    elif opt.mode=="train":
        #PSNR_avg = 0
        #SSIM_avg = 0
        print('===> Loading train Datasets')
        train_set = get_train_set(opt)  # 一下子获取到了所有的set
        train_loader = DataLoader(train_set, batch_size=opt.trainbatchsize, shuffle=True, pin_memory=pin_memory,
                                  num_workers=opt.threads)
        print('===> train Dataloading finished')
        PSNR, SSIM = train(train_loader, rsdn)
        #PSNR_avg += PSNR
        #SSIM_avg += SSIM



if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()