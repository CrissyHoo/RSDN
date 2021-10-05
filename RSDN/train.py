import utils
from utils import *
import torch.optim as optim
from option import opt
from loss import *
import torch.optim.lr_scheduler as lrs
import decimal
import torch
import numpy as np
from data import *
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
def train(train_loader,model):
    #设置optimizer
    optimizer=optim.Adam(model.parameters(), lr=opt.initial_lr)
    sche=lrs.MultiStepLR(optimizer,[60],0.9,last_epoch=-1)
    lossfunc=L1_Charbonnier_loss()
    cudnn.benchmark=True
    print("begintrain")
    PSNR=[]
    SSIM=[]
    LOSS=[]
    max_psnr=0
    while sche.last_epoch < opt.epoch:
        print("intrain")
        batch_out = [[], [], []]
        sche.step()
        l_r = sche.get_lr()[0]
        print('Epoch {:3d} with Lr {:.2e}'.format(sche.last_epoch, decimal.Decimal(l_r)))
        model.train()
        for batch, data in enumerate(train_loader):
            optimizer.zero_grad()
            # for name, parameters in self.model.named_parameters():  # 打印出每一层的参数的大小
            #   print(name, ':', parameters)
            #  break
            # print("LR",LR.shape)#[B,T,C,H,W][16,7,3,68,116]每次返回了一整个batch，
            # print("L",L)#注意这里的L也是有batch个
            LR, LR_D, LR_S, GT, GT_D, GT_S ,L= data[0], data[1], data[2], data[3], data[4], data[5],data[6]
            #print("Ltype",type(L))
            # data_r=data.flip(dims=[2])
            # LRr, LR_Dr, LR_Sr, GTr, GT_Dr, GT_Sr=data_r[0].to(self.device),data_r[1].to(self.device),data_r[2].to(self.device),data_r[3].to(self.device),data_r[4].to(self.device),data_r[5].to(self.device)

            loop = LR.shape[0]
            # print("loop:",loop)#应该是16
            psnr_sum = 0
            ssim_sum = 0

            GT = GT.cuda(non_blocking=True)
            GT_D = GT_D.cuda(non_blocking=True)
            GT_S = GT_S.cuda(non_blocking=True)
            LR = LR.cuda(non_blocking=True)
            LR_D = LR_D.cuda(non_blocking=True)
            LR_S = LR_S.cuda(non_blocking=True)
            SR, SR_D, SR_S = model(LR, LR_D, LR_S)
            # SRr,SR_Dr,SR_Sr=self.model(LRr,LR_Dr,LR_Sr)

            # print("data grad",SR.grad)
            # print("SRshape",SR.shape,SR_S.shape,SR_D.shape)
            # print("GTshape",GT.shape,GT_S.shape,GT_D.shape)
            loss = lossfunc(SR, GT) + lossfunc(SR_D, GT_D) + lossfunc(SR_S, GT_S)  # GT的尺寸不对
            #loss.retain_grad()
            #print("loss",loss)
            # print(SR.shape,GT.shape)#[8,7,3,h,w]
            loss.backward()
            #print("lossgrad", loss.grad)
            optimizer.step()

            #calculate psnr and ssim
            for batch_num in range(loop):
                SR_t = SR[batch_num].permute(0, 2, 3, 1)#tchw->thwc
                GT_t = GT[batch_num].permute(0, 2, 3, 1)
                SR_t = SR_t.detach().cpu().numpy()[:, :, :, ::-1]  # tensor rgb -> bgr
                GT_t = GT_t.detach().cpu().numpy()[:, :, :, ::-1]
                SR_t=crop_border_RGB(SR_t,8)
                GT_t=crop_border_RGB(GT_t,8)
                for frame_num in range(SR.shape[1]):
                    # 应该就是L的值，但这样写也可以
                    SR_frame = bgr2ycbcr(SR_t[frame_num])
                    SR_frame=SR_frame*255
                    GT_frame = bgr2ycbcr(GT_t[frame_num])
                    GT_frame=GT_frame*255
                    # preparation well, then we calculate psnr and ssim
                    psnr_sum += calculate_psnr(SR_frame, GT_frame)
                    ssim_sum += calculate_ssim(SR_frame, GT_frame)
            # 这样就计算完了8个video的总psnr和总ssim,我们需要得到平均的值
            psnr_sum /= (loop * opt.frame_num)
            ssim_sum /= (loop * opt.frame_num)
            #print(psnr_sum)
            #print(type(psnr_sum))
            #print(ssim_sum)
            #print(type(ssim_sum))
            batch_out[0].append(psnr_sum)
            batch_out[1].append(ssim_sum)
            batch_out[2].append(loss.item())
            if batch % 50 == 0:
                print("batch", batch)
                print("in every 50 batch: psnr:", np.mean(batch_out[0]), "ssim:", np.mean(batch_out[1]), "loss:",np.mean(batch_out[2]))
            print("in one batch: psnr:", np.mean(batch_out[0]), "ssim:", np.mean(batch_out[1]), "loss:", np.mean(batch_out[2]))

        psnr = np.mean(batch_out[0])  # epoch内的均值
        ssim = np.mean(batch_out[1])
        print("in one epoch：psnr:", psnr, "ssim:", ssim, "loss:", np.mean(batch_out[2]))
        LOSS.append(np.mean(batch_out[2]))


        print("in eval")

        print('===> Loading eval Datasets')
        eval_set = get_eval_set(opt)  # 一下子获取到了所有的set
        eval_loader = DataLoader(eval_set, batch_size=opt.evalbatchsize, shuffle=True, pin_memory=True,
                                  num_workers=opt.threads)
        print('===> eval Dataloading finished')
        if((sche.last_epoch+1)%10==0):
            save(model,os.path.join(opt.image_out, "model"), filename="rsdn{}".format(sche.last_epoch+1))
        batch_out = [[], []]#psnr ssim
        model.eval()
        psnr_sum = 0
        ssim_sum = 0
        with torch.no_grad():
            for batch, data in enumerate(eval_loader):
                LR, LR_D, LR_S, GT, GT_D, GT_S ,le= data[0], data[1], data[2], data[3], data[4], data[5],data[6]
                loop = LR.shape[0]
                GT = GT.cuda(non_blocking=True)
                LR = LR.cuda(non_blocking=True)
                LR_D = LR_D.cuda(non_blocking=True)
                LR_S = LR_S.cuda(non_blocking=True)
                SR, SR_D, SR_S = model(LR, LR_D, LR_S)
                for batch_num in range(loop):
                    SR_t = SR[batch_num].permute(0, 2, 3, 1)
                    GT_t = GT[batch_num].permute(0, 2, 3, 1)
                    SR_t = SR_t.cpu().numpy()[:, :, :, ::-1]  # tensor rgb -> bgr
                    GT_t = GT_t.cpu().numpy()[:, :, :, ::-1]
                    SR_t = utils.crop_border_RGB(SR_t, 8)
                    GT_t = utils.crop_border_RGB(GT_t, 8)
                    for frame_num in range(SR.shape[1]):
                        utils.save_img(SR_t[frame_num],"eval",frame_num,False)
                        # 应该就是L的值，但这样写也可以
                        SR_frame = bgr2ycbcr(SR_t[frame_num])
                        SR_frame=SR_frame*255
                        GT_frame = bgr2ycbcr(GT_t[frame_num])
                        GT_frame=GT_frame*255
                        # preparation well, then we calculate psnr and ssim
                        psnr_sum += calculate_psnr(SR_frame, GT_frame)
                        ssim_sum += calculate_ssim(SR_frame, GT_frame)
                # 这样就计算完了8个video的总psnr和总ssim,我们需要得到平均的值
                psnr_sum /= (loop * le)
                ssim_sum /= (loop * le)
            batch_out[0].append(psnr_sum)
            batch_out[1].append(ssim_sum)
            psnr_sum = np.mean(batch_out[0])
            print("in evaluation, the psnr is:", psnr_sum, "the ssim is:", np.mean(batch_out[1]))

            if psnr_sum > max_psnr:
                max_psnr = psnr_sum

        PSNR.append(psnr_sum)
        SSIM.append(ssim_sum)

    plot_graph("psnr in eval", PSNR)
    return PSNR,SSIM
