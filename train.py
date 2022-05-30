import os
import torch
print(torch.cuda.is_available())#是否有可用的gpu
print(torch.cuda.device_count())#有几个可用的gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#声明gpu
dev=torch.device('cuda:0')#调用哪个gpu
a=torch.rand(100,100).to(dev)

print(torch.cuda.current_device())#可用gpu编号
print( torch.cuda.get_device_capability(device=None),  torch.cuda.get_device_name(device=None))#可用gpu内存大小，可用gpu的名字
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#声明gpu
dev=torch.device('cuda:0')#调用哪个gpu
a=torch.rand(100,100).to(dev)

import torch
import torch.nn as nn
import torchvision
import torch.quantization

import torch.backends.cudnn as cudnn
import torch.optim
import random
import os
import sys
import argparse
import time
import dataloader
from model import MSPEC_Net,Discriminator
from Myloss import My_loss,D_loss
import numpy as np
from torchvision import transforms

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
import torch, gc

