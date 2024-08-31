import torchvision.transforms as transforms
# from FusionNet.gennet import Gen
import torch
import time
import cv2
import os
import numpy as np
from FusionNet.gennet import *
import skimage.measure
# from FusionNet.dis_net import COM
from utils.utils import gen_label, Gradient, GaussianBlur2d
from utils.image import *
import torchvision.models as models
from utils.utils import *
from torch.utils.data import DataLoader
# from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.TestDataSet import TestDataset, yolo_dataset_collate

# from utils.GenDataset import GDataset,yolo_dataset_collate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 测试模型
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def savenp(tensor, path):
    y = tensor
    h = y.shape[2]
    w = y.shape[3]
    y = y * 255.0
    img_copy = y.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    img_copy = np.clip(img_copy, 0, 255)
    img_copy = img_copy.astype(np.uint8)[0, :, :, :]
    cv2.imwrite(path, img_copy)
    sd = np.std(img_copy)
    en = skimage.measure.shannon_entropy(img_copy)
    return sd, en


def testfusion(path, fusionnet, feature):
    print('Loading weights into state dict...')
    fusion_model = torch.nn.DataParallel(fusionnet)
    fusion_model.load_state_dict(torch.load(path))
    fusionnet.eval()
    fake_ab = fusionnet(feature)
    return fake_ab


def testdense(path, densenet, inputir, inputvis):
    print('feature extract')
    densenet.load_state_dict(torch.load(path))
    densenet.eval()
    feature = densenet.encoder(torch.cat((inputir, inputvis), 1))
    fake_l = densenet.decoder(feature)
    return fake_l, feature


def vgg_feature(tensor):
    input = torch.cat((tensor, tensor, tensor), 1)
    vgg = models.vgg16_bn(pretrained=True).features[:28].to(device)
    vgg.eval()
    out = vgg(input)
    return out


def load_model(path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    gen_model = Gen().to(device)
    gen_model_path = path
    print('Loading weights into state dict...')
    gen_model_dict = gen_model.state_dict()
    gen_pretrained_dict = torch.load(gen_model_path, map_location=device)
    gen_pretrained_dict = {k: v for k, v in gen_pretrained_dict.items() if np.shape(gen_model_dict[k]) == np.shape(v)}
    gen_model_dict.update(gen_pretrained_dict)
    gen_model.load_state_dict(gen_model_dict)
    print('Finished!')
    gennet = gen_model.eval()
    for param in gennet.parameters():
        param.requires_grad = False
    return gen_model


if __name__ == "__main__":
    if True:
        val_split = 0
        num = 0
        input_shape = (256, 256)
        # 0.197~94985
        annotation_irpath = '../path/test_TNO21_ir.txt'
        with open(annotation_irpath) as f:
            lines = f.readlines()
        # np.random.seed(10101)
        # np.random.shuffle(lines)
        # np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val
        ir_dataset = TestDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        genir = DataLoader(ir_dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True,
                           drop_last=False, collate_fn=yolo_dataset_collate)
        annotation_vispath = '../path/test_TNO21_vis.txt'
        with open(annotation_vispath) as f:
            lines = f.readlines()
        # np.random.seed(10101)
        # np.random.shuffle(lines)
        # np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines)
        vis_dataset = TestDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        genvis = DataLoader(vis_dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=True,
                            drop_last=False, collate_fn=yolo_dataset_collate)
        print(len(genvis))
        std = 0
        encropy = 0
        # print(torch.max(lir),torch.min(lvis))
        # --------------------#
        test_f = False
        test_base = False
        com = False
        dark_net = False
        endecoder = True
        tt = 0
        number = 0

        if endecoder == True:
            for num in range(5, 6):
                for iteration, batch in enumerate(zip(genir, genvis)):
                    ir_imgs = batch[0]
                    vis_imgs = batch[1]
                    with torch.no_grad():
                        # images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                        ir_imgs = torch.from_numpy(ir_imgs).type(torch.FloatTensor).cuda()
                        vis_imgs = torch.from_numpy(vis_imgs).type(torch.FloatTensor).cuda()
                    # print(lab.shape)
                    irlab = rgb2lab(ir_imgs)
                    vislab = rgb2lab(vis_imgs)
                    lvis = vislab[:, [0], :, :]
                    lir = irlab[:, [0], :, :]
                    # ------#
                    encoder = Encoder().to(device)
                    decoder = Decoder().to(device)

                    encoder = torch.nn.DataParallel(encoder)
                    decoder = torch.nn.DataParallel(decoder)
                    # print("finish")
                    encoder.load_state_dict(
                        torch.load('./logs/encoder' + str(num) + '-Total_Loss0.0000-Val_Loss0.0000.pth'))
                    encoder.eval()
                    decoder.load_state_dict(
                        torch.load('./logs/decoder' + str(num) + '-Total_Loss0.0000-Val_Loss0.0000.pth'))
                    decoder.eval()
                    etotal_params = sum(p.numel() for p in encoder.parameters())
                    dtotal_params = sum(p.numel() for p in decoder.parameters())
                    with torch.no_grad():
                        begin = time.time()
                        real_feature, x1, x2 = encoder(torch.cat((lir, lvis), 1))
                        fake_ir, fake_vis, fusion = decoder(real_feature, x1, x2)
                        fake_feature, _, _ = encoder(torch.cat((fake_ir, fake_vis), 1))
                        torch.cuda.synchronize()
                        end = time.time()
                        temp = (end - begin)
                        tt = tt + temp

                    dirpath = '../result/fused_result/epoch_' + str(num)
                    mkdir(dirpath)
                    a, _ = savenp((fake_ir) * 0.5 + 0.5, "../result/fake_ir/" + str(iteration) + ".png")
                    b, _ = savenp((fake_vis) * 0.5 + 0.5, "../result/fake_vis/" + str(iteration) + ".png")
                    c, en = savenp(fusion * 0.5 + 0.5, dirpath + '/' + str(iteration + 1) + ".png")
                    if number == 1:
                        for fea in range(1, 257):
                            mean, _ = savenp((fake_feature[:, fea - 1:fea, :, :]) * 0.5 + 0.5,
                                             "../result/feature/" + str(fea) + ".png")
                    std = c + std
                    encropy = encropy + en
                    number = number + 1

        print('sd:' + str(std / len(genvis)))
        print('en:' + str(encropy / len(genvis)))
        print("时间：", (tt / 32) * 1000)
        print(f'{etotal_params / 1000 / 1000:,} encoder total parameters.')
        print(f'{dtotal_params / 1000 / 1000:,} decoder total parameters.')
        # print('total:',num/52.)
