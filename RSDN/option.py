import argparse
import datetime
import time
parser = argparse.ArgumentParser(description='PyTorch RSDN Example')
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testbatchsize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=0, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=3, type=int, help='number of gpu')
parser.add_argument('--cuda',default=True, type=bool)
parser.add_argument('--test_dir',type=str,default='/home/crissyhu/data/vid4')
parser.add_argument('--file_test_list',type=str, default ='',help='where record all of image name in dataset.')
parser.add_argument('--save_test_log', type=str,default='./log/test')
parser.add_argument('--pretrain', type=str, default='RSDN.pth')
parser.add_argument('--image_out', type=str, default='./out/')
parser.add_argument('--trainbatchsize', type=int, default=8, help='training batch size')
parser.add_argument('--train_dir',type=str,default='/home/chloe/yipeng/data/vsr')
parser.add_argument('--train_name',type=str,default='/vimeo90k/vimeo_septuplet/origin')
parser.add_argument('--save_train_log', type=str,default='./log/train')
parser.add_argument('--data_info_path',type=str,default='./meta_info_Vimeo90K_train_GT.txt')
parser.add_argument('--num_frame', type=int, default=7,help="the number of input frames")
parser.add_argument('--epoch', type=int, default=70)
parser.add_argument('--initial_lr', type=float, default=1e-4)
parser.add_argument('--loss_alpha', type=float, default=1.0)
parser.add_argument('--loss_beta', type=float, default=1.0)
parser.add_argument('--loss_gamma', type=float, default=1.0)
opt = parser.parse_args()
gpus_list = range(opt.gpus)
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
print(opt)

opt = parser.parse_args()
gpus_list = [0]
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
