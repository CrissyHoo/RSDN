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
from arch import RSDN9_128
import utils
import time
from torch.nn import functional as F



def test(test_loader, filter_net, test_name, out):
    filter_net.eval()
    count = 0
    PSNR = 0
    SSIM = 0
    PSNR_ = 0
    SSIM_ = 0
    for image_num, data in enumerate(test_loader):
        #LR, LR_d, LR_s, target, L = data[0],data[1], data[2], data[3], data[4]
        LR, LR_d, LR_s, target, L=data[0],data[1], data[2], data[3], data[-1]
        with torch.no_grad():
            LR = Variable(LR).cuda()
            LR_d = Variable(LR_d).cuda()
            LR_s = Variable(LR_s).cuda()
            target = Variable(target).cuda()
            prediction, out_d, out_s = filter_net(LR, LR_d, LR_s)
        count += 1
        prediction = prediction.squeeze(0).permute(0,2,3,1) # [T,H,W,C]
        prediction = prediction.cpu().numpy()[:,:,:,::-1] # tensor -> numpy, rgb -> bgr

        L = L.numpy()
        L = int(L)
        target = target.squeeze(0).permute(0,2,3,1) # [T,H,W,C]
        target = target.cpu().numpy()[:,:,:,::-1] # tensor -> numpy, rgb -> bgr
        target = utils.crop_border_RGB(target, 8)
        prediction = utils.crop_border_RGB(prediction, 8)
        for i in range(L):
            utils.save_img(prediction[i], test_name, i, False)
            # test_Y______________________
            prediction_Y = utils.bgr2ycbcr(prediction[i])
            target_Y = utils.bgr2ycbcr(target[i])
            prediction_Y = prediction_Y * 255
            target_Y = target_Y * 255
            # _______________________________
            #prediction_Y = prediction[i] * 255
            #target_Y = target[i] * 255
            # ________________________________
            # calculate PSNR and SSIM
            print('PSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(utils.calculate_psnr(prediction_Y, target_Y), utils.calculate_ssim(prediction_Y, target_Y)))
            PSNR += utils.calculate_psnr(prediction_Y, target_Y)
            SSIM += utils.calculate_ssim(prediction_Y, target_Y)
            out.append(utils.calculate_psnr(prediction_Y, target_Y))
        print('===>{} PSNR = {}'.format(test_name, PSNR/(L)))
        print('===>{} SSIM = {}'.format(test_name, SSIM/(L)))
        PSNR_ += PSNR/(L)
        SSIM_ += SSIM/(L)
    return PSNR_, SSIM_, out





