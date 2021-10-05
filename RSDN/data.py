#from load_duf import DataloadFromFolder
from load_test import *
from MM522 import MM522
from EVAL import EVAL
from VIMEO90k import VIMEO90K

from torchvision.transforms import Compose, ToTensor

def transform():
    return Compose([
             ToTensor(),
            ])

def get_test_set(data_dir, upscale_factor, test_name):
    return DataloadFromFolderTest(data_dir, upscale_factor, test_name,transform=transform())
def get_train_set(opt):
    #return MM522(opt.train_dir,opt.upscale_factor,opt.train_name,transform=transform())
    return VIMEO90K(opt,transform=transform())
def get_eval_set(opt):
    return EVAL(opt.eval_dir,opt.upscale_factor,opt.eval_name,transform=transform())

