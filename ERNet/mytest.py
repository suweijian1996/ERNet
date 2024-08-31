import os
import numpy as np

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms

from PIL import Image


# def get_vgg_feature(input):
#     vgg = models.vgg16(pretrained=True).features.eval()
#     out = []
#     for i in range(len(vgg)):
#         input = vgg[i](input)
#         if i in [3, 8, 15, 22]:
#             out.append(input)
#     return out
model_28 = models.vgg16(pretrained=True).features[2:22]  # 其实就是定位到第28层，对照着上面的key看就可以理解
model_28 = model_28.eval()
print(model_28)
a = torch.randn([8,64,256,256])
b = model_28(a)
print(b.shape)
