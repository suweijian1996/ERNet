import torch

def gray(tensor):
    R = tensor[:,[0],:,:]
    G = tensor[:,[1],:,:]
    B = tensor[:,[2],:,:]
    t = 0.299 * R + 0.587 * G + 0.114 * B
    return t
# a = torch.randn(8,3,128,128)
# b = gray(a)
# print(b.shape)


