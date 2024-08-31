from __future__ import division
import random
import os
from typing import Tuple
import kornia
from kornia.filters.kernels import get_gaussian_kernel2d
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        #-----------------------------------------------------------#
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #-----------------------------------------------------------#
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 255, 13, 13
        #   batch_size, 255, 26, 26
        #   batch_size, 255, 52, 52
        #-----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        #-----------------------------------------------#
        #   输入为416x416时
        #   stride_h = stride_w = 32、16、8
        #-----------------------------------------------#
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width
        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

        #-----------------------------------------------#
        #   输入的input一共有三个，他们的shape分别是
        #   batch_size, 3, 13, 13, 85
        #   batch_size, 3, 26, 26, 85
        #   batch_size, 3, 52, 52, 85
        #-----------------------------------------------#
        prediction = input.view(batch_size, self.num_anchors,
                                self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]
        h = prediction[..., 3]
        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        #----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角 
        #   batch_size,3,13,13
        #----------------------------------------------------------#
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        #----------------------------------------------------------#
        #   按照网格格式生成先验框的宽高
        #   batch_size,3,13,13
        #----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        #----------------------------------------------------------#
        #   将输出结果调整成相对于输入图像大小
        #----------------------------------------------------------#
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, self.num_classes)), -1)
        return output.data
        
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)/input_shape
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   将预测结果的格式转换成左上角右下角的格式。
    #   prediction  [batch_size, num_anchors, 85]
    #----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选
        #----------------------------------------------------------#
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()

        #----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf, class_conf, class_pred
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            #------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]
            
            # # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # 进行非极大抑制
            # max_detections = []
            # while detections_class.size(0):
            #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # 堆叠
            # max_detections = torch.cat(max_detections).data
            
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output
def gen_int_label(batchsize,key):
    labels = torch.Size([])
    if key == 'ir':#10
        for i in range(batchsize):
            x = (torch.FloatTensor([1])).to(device)
            y = (torch.FloatTensor([0])).to(device)
            z = (torch.FloatTensor([0])).to(device)
            l = torch.cat((x, y,z), 0)
            l = l.unsqueeze(0)
            label = l
            if (labels == torch.Size([])):
                labels = label
            else:
                labels = torch.cat((labels, label), 0)
    elif key == 'vis':#01
        for i in range(batchsize):
            x = (torch.FloatTensor([0])).to(device)
            y = (torch.FloatTensor([1])).to(device)
            z = (torch.FloatTensor([0])).to(device)
            l = torch.cat((x, y,z), 0)
            l = l.unsqueeze(0)
            label = l
            if (labels == torch.Size([])):
                labels = label
            else:
                labels = torch.cat((labels, label), 0)
    else:
        print("label index is wrong")
        labels = 0
    return labels.unsqueeze(2).unsqueeze(2)
def gen_label(batchsize,key):
    labels = torch.Size([])
    if key == 'ir':#10
        for i in range(batchsize):
            x = (torch.FloatTensor([random.random()*0.5+0.7])).to(device)
            y = (torch.FloatTensor([random.random()*0.3])).to(device)
            z = (torch.FloatTensor([random.random()*0.3])).to(device)
            l = torch.cat((x, y), 0)
            l = l.unsqueeze(0)
            label = l
            if (labels == torch.Size([])):
                labels = label
            else:
                labels = torch.cat((labels, label), 0)
    elif key == 'vis':#01
        for i in range(batchsize):
            x = (torch.FloatTensor([random.random()*0.3])).to(device)
            y = (torch.FloatTensor([random.random()*0.5+0.7])).to(device)
            z = (torch.FloatTensor([random.random()*0.3])).to(device)
            l = torch.cat((x, y), 0)
            l = l.unsqueeze(0)
            label = l
            if (labels == torch.Size([])):
                labels = label
            else:
                labels = torch.cat((labels, label), 0)
    elif key =='natural':#11
        for i in range(batchsize):
            x = (torch.FloatTensor([random.random()*0.3])).to(device)
            y = (torch.FloatTensor([random.random()*0.5+0.7])).to(device)
            z = (torch.FloatTensor([random.random()*0.5+0.7])).to(device)
            l = torch.cat((x, y), 0)
            l = l.unsqueeze(0)
            label = l
            if (labels == torch.Size([])):
                labels = label
            else:
                labels = torch.cat((labels, label), 0)
    elif key == 'fake_natural':
        for i in range(batchsize):
            x = (torch.FloatTensor([random.random()*0.5+0.7])).to(device)
            y = (torch.FloatTensor([random.random()*0.5+0.7])).to(device)
            z = (torch.FloatTensor([random.random()*0.3])).to(device)
            l = torch.cat((x, y, z), 0)
            l = l.unsqueeze(0)
            label = l
            if (labels == torch.Size([])):
                labels = label
            else:
                labels = torch.cat((labels, label), 0)
    else:
        print("label index is wrong")
        labels = 0
    return labels.unsqueeze(2).unsqueeze(2)

class Gradient(nn.Module):
    def __init__(self, channels=1):
        super(Gradient, self).__init__()
        self.channels = channels
        kernel = [[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels)
        return x



def gaussian_blur2d(
    input: torch.Tensor, kernel_size: Tuple[int, int], sigma: Tuple[float, float], border_type: str = 'reflect'
) -> torch.Tensor:
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    .. image:: _static/img/gaussian_blur2d.png

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        input: the input tensor with shape :math:`(B,C,H,W)`.
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        the blurred tensor with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       gaussian_blur.html>`__.

    Examples:
        >>> input = torch.rand(2, 4, 5, 5)
        >>> output = gaussian_blur2d(input, (3, 3), (1.5, 1.5))
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """
    kernel: torch.Tensor = torch.unsqueeze(get_gaussian_kernel2d(kernel_size, sigma), dim=0)

    return kornia.filter2d(input, kernel, border_type)



class GaussianBlur2d(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It supports batched operation.

    Arguments:
        kernel_size: the size of the kernel.
        sigma: the standard deviation of the kernel.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.

    Returns:
        the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = GaussianBlur2d((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
        >>> output.shape
        torch.Size([2, 4, 5, 5])
    """

    def __init__(self, kernel_size: Tuple[int, int], sigma: Tuple[float, float], border_type: str = 'reflect') -> None:
        super(GaussianBlur2d, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self.border_type = border_type

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(kernel_size='
            + str(self.kernel_size)
            + ', '
            + 'sigma='
            + str(self.sigma)
            + ', '
            + 'border_type='
            + self.border_type
            + ')'
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return gaussian_blur2d(input, self.kernel_size, self.sigma, self.border_type)

class Sobel(nn.Module):
    def __init__(self, channels=1):
        super(Sobel, self).__init__()
        self.channels = channels
        kernel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
        kernel_x= torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_x = np.repeat(kernel_x, self.channels, axis=0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        kernel_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        kernel_y = np.repeat(kernel_y, self.channels, axis=0)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def __call__(self, x):
        x_x = F.conv2d(x, self.weight_x, padding=1, groups=self.channels)
        x_x = torch.abs(x_x)
        x_y = F.conv2d(x, self.weight_y, padding=1, groups=self.channels)
        x_y = torch.abs(x_y)
        x = torch.add(0.5*x_x,0.5*x_y)
        return x



# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    # if(torch.sum(torch.isnan(rgb))>0):
        # print('xyz2rgb')
        # embed()
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('xyz2lab')
        # embed()

    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc

    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2xyz')
        # embed()

    return out
#输入RGB：0-1 输出lab:-1-1
def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    # print(lab[0, 0, 0, 0])
    l_rs = (lab[:,[0],:,:]*2.-100.)/100.
    # print(l_rs[0, 0, 0, 0])
    ab_rs = lab[:,1:,:,:]/110.
    out = torch.cat((l_rs,ab_rs),dim=1)
    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2lab')
        # embed()
    return out

#输入lab:-1-1 输出rgb：0-1
def lab2rgb(lab_rs):
    l = lab_rs[:,[0],:,:]*100./2. + 50.
    ab = lab_rs[:,1:,:,:]*110.
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    # out = (out-0.5)*2.
    #-1-1
    # if(torch.sum(torch.isnan(out))>0):
        # print('lab2rgb')
        # embed()
    return out

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


class YCbCrToRGB(object):
    def __call__(self, img):
        return torch.stack((img[:, 0, :, :] + (img[:, 2, :, :] - 128 / 256.) * 1.402,
                            img[:, 0, :, :] - (img[:, 1, :, :] - 128 / 256.) * 0.344136 - (img[:, 2, :, :] - 128 / 256.) * 0.714136,
                            img[:, 0, :, :] + (img[:, 1, :, :] - 128 / 256.) * 1.772),
                            dim=1)

class RGBToYCbCr(object):
    def __call__(self, img):
        return torch.stack((0. / 256. + img[:, 0, :, :] * 0.299000 + img[:, 1, :, :] * 0.587000 + img[:, 2, :, :] * 0.114000,
                           128. / 256. - img[:, 0, :, :] * 0.168736 - img[:, 1, :, :] * 0.331264 + img[:, 2, :, :] * 0.500000,
                           128. / 256. + img[:, 0, :, :] * 0.500000 - img[:, 1, :, :] * 0.418688 - img[:, 2, :, :] * 0.081312),
                          dim=1)

def SF(tensor):
    RF = torch.diff(tensor.dim)
# a = torch.FloatTensor([8,3,256,256])
# b = rgb2lab(a)
# print(b.shape)

