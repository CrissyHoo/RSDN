import datetime
import argparse

parser = argparse.ArgumentParser(description='PyTorch RSDN Example')
#model parameter
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--scale', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--upscale_factor', default=4, type=int, help='scale value')
parser.add_argument('--epoch', default=1, type=int, help='epoch number')
parser.add_argument('--frame_num', default=7, type=int, help='frame number dealing once')
parser.add_argument('--initial_lr', type=float, default=1e-4)
parser.add_argument('--crop_size', default=72, type=int, help='lr patch size')
parser.add_argument('--loss_alpha', type=float, default=1.0)
parser.add_argument('--loss_beta', type=float, default=1.0)
parser.add_argument('--loss_gamma', type=float, default=1.0)

#device
parser.add_argument('--threads', type=int, default=32, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=None, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=3, type=int, help='number of gpu')
parser.add_argument('--cuda',default=True, type=bool)

#train
parser.add_argument('--pretrain', type=str, default='./out/model/model_rsdnmm522best.pth')
parser.add_argument('--load_pretrain',type=bool,default=True)
parser.add_argument('--train_dir', type=str, default='/home/zywang4/dat01/hsc/data')
#parser.add_argument('--train_name', type=str, default='MM522')
parser.add_argument('--train_name', type=str, default='vimeo90k/vimeo_septuplet/sequences')
parser.add_argument('--data_info_path', type=str, default='/home/zywang4/dat01/hsc/data/vimeo90k/meta_info_Vimeo90K_train_GT.txt')
parser.add_argument('--trainbatchsize', default=12, type=int, help='trainbatchsize')

#eval
parser.add_argument('--eval_dir', type=str, default='/home/zywang4/dat01/hsc/data')
parser.add_argument('--eval_name', type=str, default='eval')
parser.add_argument('--evalbatchsize', default=1, type=int, help='evalbatchsize')

#test
parser.add_argument('--testbatchsize', type=int, default=1, help='testing batch size')
parser.add_argument('--test_dir',type=str,default='/home/zywang4/dat01/hsc/data/vid4')
parser.add_argument('--file_test_list',type=str, default ='',help='where record all of image name in dataset.')
parser.add_argument('--save_test_log', type=str,default='./log/test')

#logger
parser.add_argument('--image_out', type=str, default='./out/')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
print(opt)