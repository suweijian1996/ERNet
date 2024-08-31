from FusionNet.gennet import *
from nets.yolo_training import weights_init,LossHistory
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.optim as optim
from utils.imagepool import ImagePool
from torch.utils.data import DataLoader
from FusionNet.dis_net import DisVIS_net,DisIR_net,VGG
from FusionNet.fusionnet import Perceptual
# from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.GenDataset import GDataset,yolo_dataset_collate
from utils.fusionloss import GANLoss,msssim,GenGANLoss
import matplotlib.pyplot as plt
from utils.data_iv import ir,vis
from utils.image import *
from tqdm import tqdm
from utils.utils import gen_label,Gradient,GaussianBlur2d,gen_int_label,Sobel,rgb2lab,lab2rgb
import torchvision.transforms as transforms
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def fit_one_epoch(encoder,decoder,dirnet,epoch, epoch_size, epoch_size_val, genir, genvis,gen, Epoch, batch_size,cuda):
    tloss_idt = 0
    tloss_base = 0
    val_loss = 0
    total_loss = 0
    print('Start Train')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(zip(gen,genir,genvis,genthree)):
            if iteration >= epoch_size:
                break
            images = batch[0]
            ir = batch[1]
            vis = batch[2]
            three = batch[3]
            # ir_label = gen_label(batch_size,'ir')
            # vis_label = gen_label(batch_size, 'vis')
            # natural_label = gen_label(batch_size, 'natural')
            # ir_int_label = gen_int_label(batch_size,'ir')
            # vis_int_label = gen_int_label(batch_size, 'vis')
            # fake_natural_label = gen_label(batch_size, 'fake_natural')
            # # print(ir_label.shape)
            # img = images[1]  # plt.imshow()只能接受3-D Tensor，所以也要用image[0]消去batch那一维
            # img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
            # # 显示图片
            # plt.imshow(img)
            # plt.show()
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    ir = torch.from_numpy(ir).type(torch.FloatTensor).cuda()
                    vis = torch.from_numpy(vis).type(torch.FloatTensor).cuda()
                    three = torch.from_numpy(three).type(torch.FloatTensor).cuda()
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    ir = torch.from_numpy(ir).type(torch.FloatTensor)
                    vis = torch.from_numpy(vis).type(torch.FloatTensor)
                    three = torch.from_numpy(three).type(torch.FloatTensor)
            # print(images.shape)
            lab_ir = rgb2lab(ir)
            lab_vis = rgb2lab(vis)
            lab_three = rgb2lab(three)
            lab = rgb2lab(images)
            l = lab[:,[0],:,:]
            lir = lab_ir[:,[0],:,:]
            lvis = lab_vis[:,[0],:,:]
            lthree = lab_three[:,[0],:,:]


            # -----------------------#
            #  判别器vis
            # ------------------------#
            # dvis_optimizer.zero_grad()
            # fake_ir, fake_vis, _ = net(lir, lvis)
            # loss_B = backward_B(dvisnet, lvis, fake_vis)
            # loss_B.backward()
            # dvis_optimizer.step()


            for temp in range(1):

                # -----------------------#
                #  判别器ir
                # -----------------------#
                dir_optimizer.zero_grad()
                real_feature, x1,x2= encoder(torch.cat((lir, lvis,lthree),1))
                fake_ir,fake_vis,fake_three,_ = decoder(real_feature,x1,x2)
                fake_feature,_,_ = encoder(torch.cat((fake_ir,fake_vis,fake_three),1))
                loss_A = backward_A(dirnet, real_feature, fake_feature)
                loss_A.backward()
                dir_optimizer.step()
                # for parm in disvis_net.parameters():
                #     parm.data.clamp_(-0.01, 0.01)
            # for parm in dirnet.parameters():
            #     parm.data.clamp_(-0.01, 0.01)
            #-----------------------#
            #    更新生成器
            #    encoder
            #-----------------------#
            en_optimizer.zero_grad()
            real_feature, x1,x2 = encoder(torch.cat((lir, lvis,lthree), 1))
            fake_ir, fake_vis,fake_three,_ = decoder(real_feature, x1,x2)
            fake_feature, _, _ = encoder(torch.cat((fake_ir, fake_vis,fake_three), 1))
            loss_Gir =0.5*criterionGAN(dirnet(fake_feature),True)
            loss_idt = l1loss(fake_ir,lir) + l1loss(fake_vis,lvis) + l1loss(fake_three,lthree)
            loss_encoder = loss_Gir+loss_idt
            loss_encoder.backward()
            en_optimizer.step()
            # -----------------------#
            #    更新生成器
            #    decoder
            # -----------------------#
            # loss_per = 300*(l1loss(feature_fakeir,feature_ir) + l1loss(feature_fakevis,feature_fakevis))
            de_optimizer.zero_grad()
            l_feature, x1, x2 = encoder(torch.cat((l, l,l), 1))
            _, _,_, output = decoder(l_feature.detach(), x1, x2)
            loss_base = 100*l1loss(gradient(output), gradient(l)) + mseloss(output, l)
            loss_base.backward()
            de_optimizer.step()
            # loss_idt = -(l1smoothloss(outputvis,output) + l1smoothloss(outputir,output))

            # ----------------------#
            #   生成损失
            # ----------------------#
            # print(output1.shape,output2.shape,images.shape)
            tloss_idt += loss_idt.item()
            tloss_base += loss_base.item()
            #-----------------------#
            #    更新判别器
            # -----------------------#
            # d_optimizer.zero_grad()
            # output_i, output_v = gnet(images)
            # loss_D = backward_D(dnet, ir, vis,output_i,output_v)
            # dfake_i = dnet(output_i)
            # dfake_v = dnet(output_v)
            # loss_di = mseloss(dfake_i,natural_label)#001
            # loss_dv = mseloss(dfake_v,natural_label)#001
            # loss_dgan = loss_di
            # #-------------------------#
            # #     真实图像
            # # ------------------------#
            # irl = dnet(ir)
            # visl = dnet(vis)
            # naturall = dnet(gray(images))
            # loss_realiv = mseloss(irl,ir_label)
            # d_loss = loss_dgan + loss_realiv
            # #-------------------------#
            # #     反向传播
            # #-------------------------#
            # loss_D.backward()
            # d_optimizer.step()
            # dtotal_loss += loss_D.item()
            pbar.set_postfix(**{'total_loss': tloss_base / (iteration + 1),
                                'loss_idt':tloss_idt / (iteration + 1),
                                'ga_loss':loss_Gir.item(),
                                'A_loss':loss_A.item(),
                                'lr': get_lr(de_optimizer)})
            pbar.update(1)

    print('Finish Train')
    print('Start Validation')
    # with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
    #     for iteration, batch in enumerate(genval):
    #         if iteration >= epoch_size_val:
    #             break
    #         images_val,_ = batch[0], batch[1]
    #         with torch.no_grad():
    #             if cuda:
    #                 images_val = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
    #                 # targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
    #             else:
    #                 images_val = torch.from_numpy(images_val).type(torch.FloatTensor)
    #                 # targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
    #             optimizer.zero_grad()
    #             output1, output2 = gnet(l)
    #             # ----------------------#
    #             #   计算损失
    #             # ----------------------#
    #             loss_pir = mseloss(output1, l)
    #             loss_pvis = mseloss(output2, l)
    #             loss = loss_pir + loss_pvis
    #             val_loss += loss.item()
    #         pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
    #         pbar.update(1)

    # loss_history.append_loss(loss_base / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    print('Saving state, iter:', str(epoch + 1))
    torch.save(encoder.state_dict(), 'logs/encoder%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    torch.save(decoder.state_dict(), 'logs/decoder%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

def backward_B_basic(netD, real, fake):
    pred_real = netD(real.detach())
    loss_D_real = criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake)
    return loss_D

def backward_B(D, real_B, fake_B):
    """Calculate GAN loss for discriminator D_A"""
    #fake_B = fake_B_pool.query(fake_B)
    loss_D = backward_B_basic(D, real_B, fake_B)
    return loss_D

def backward_A_basic(netD, real, fake):
    pred_real = netD(real.detach())
    loss_D_real = criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach())
    loss_D_fake = criterionGAN(pred_fake, False)
    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake)
    return loss_D

def backward_A(D, real_B, fake_B):
    """Calculate GAN loss for discriminator D_A"""
    #fake_B = fake_A_pool.query(fake_B)
    loss_D = backward_A_basic(D, real_B, fake_B)
    return loss_D
if __name__ == "__main__":
    encoder = Three_Encoder()
    disir_net = DisIR_net()
    decoder = Three_Decoder()
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # ------------------------------------------------------#
    #   输入的shape大小
    # ------------------------------------------------------#
    input_shape = (256, 256)
    # ------------------------------------------------------#
    #   创建yolo模型
    #   训练前一定要修改Config里面的classes参数
    # ------------------------------------------------------#
    fake_A_pool = ImagePool(50)
    fake_B_pool = ImagePool(50)
    weights_init(encoder)
    weights_init(disir_net)
    weights_init(decoder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # weights_init(model1)
    if Cuda:
        cudnn.benchmark = True
        encoder = torch.nn.DataParallel(encoder)
        disir_net = torch.nn.DataParallel(disir_net)
        decoder = torch.nn.DataParallel(decoder)
        disir_net = disir_net.cuda()
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    mseloss = nn.MSELoss().to(device)
    l1loss = nn.L1Loss().to(device)
    ssim_loss = msssim
    l1smoothloss = nn.SmoothL1Loss().to(device)
    criterionGAN = GANLoss().to(device)
    # criterionGenGAN = GenGANLoss().to(device)
    gradient = Sobel().to(device)
    gauss = GaussianBlur2d((3, 3), (2, 2))
    loss_history = LossHistory("logs/")

    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    # ----------------------------------------------------------------------#
    #   验证集的划分在train.py代码里面进行
    #   2007_test.txt和2007_val.txt里面没有内容是正常的。训练不会使用到。
    #   当前划分方式下，验证集和训练集的比例为1:9
    # ----------------------------------------------------------------------#
    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = 1e-4
        Batch_size =4
        Init_Epoch = 0
        Freeze_Epoch = 50
        gamma = 0.92
        num_workers = 0

        en_optimizer = optim.Adam(encoder.parameters(), lr=lr)
        de_optimizer = optim.Adam(decoder.parameters(), lr=lr)
        dir_optimizer = optim.Adam(disir_net.parameters(), lr=lr)
        enlr_scheduler = optim.lr_scheduler.StepLR(en_optimizer, step_size=1, gamma=gamma)
        delr_scheduler = optim.lr_scheduler.StepLR(de_optimizer, step_size=1, gamma=gamma)
        dlr_scheduler = optim.lr_scheduler.StepLR(dir_optimizer, step_size=1, gamma=gamma)

        # dvis_optimizer = optim.SGD(disvis_net.parameters(), lr=lr)
        # irloader = torch.utils.data.DataLoader(ir, batch_size=Batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,drop_last=True)
        # visloader = torch.utils.data.DataLoader(vis, batch_size=Batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,drop_last=True)
        val_split = 0
        #0.101~16000
        # 0.197~94985
        val_natural = 0
        annotation_path = '../2017_train.txt'
        with open(annotation_path) as f:
            lines = f.readlines()
        # np.random.seed(10101)
        # np.random.shuffle(lines)
        # np.random.seed(None)
        num_val = int(len(lines) * val_natural)
        num_train = len(lines) - num_val
        train_dataset = GDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        annotation_path = '../path/mf/train_patch1.txt'
        with open(annotation_path) as f:
            lines = f.readlines()
        # np.random.seed(10101)
        # np.random.shuffle(lines)
        # np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val
        ir_dataset = GDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        genir = DataLoader(ir_dataset, shuffle=False, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                           drop_last=False, collate_fn=yolo_dataset_collate)
        annotation_path = '../path/mf/train_patch2.txt'
        with open(annotation_path) as f:
            lines = f.readlines()
        # np.random.seed(10101)
        # np.random.shuffle(lines)
        # np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val
        vis_dataset = GDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        genvis = DataLoader(vis_dataset, shuffle=False, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                            drop_last=False, collate_fn=yolo_dataset_collate)

        annotation_path = '../path/mf/train_patch3.txt'
        with open(annotation_path) as f:
            lines = f.readlines()
        # np.random.seed(10101)
        # np.random.shuffle(lines)
        # np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val
        vis_dataset = GDataset(lines[:num_train], (input_shape[0], input_shape[1]), True)
        genthree = DataLoader(vis_dataset, shuffle=False, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                            drop_last=False, collate_fn=yolo_dataset_collate)
        print(len(genvis),len(genir))
        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size

        if epoch_size == 0 :
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(encoder,decoder,disir_net, epoch, epoch_size, epoch_size_val, genir, genvis,gen, Freeze_Epoch,Batch_size,Cuda)
            enlr_scheduler.step()
            delr_scheduler.step()
            dlr_scheduler.step()
