import os
from importlib import import_module
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()
        self.opt=opt
        self.device=torch.device('cuda')
        module = import_module('model.rsdn')
        self.model = module.make_model(opt).to(self.device)
        #print(self.get_model())

    def forward(self, *args):
        return self.model(*args)

    def get_model(self):
        return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self,apath,filename=''):
        target=self.get_model()
        filename='model_{}'.format(filename)
        torch.save(
            target.state_dict(),
            os.path.join(apath, 'model', '{}best.pt'.format(filename))
        )
