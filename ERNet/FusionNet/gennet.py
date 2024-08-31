import torch.nn as nn
import torch
from torch.nn import functional as F
from deform.deform_conv_v2 import *
#
class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,True)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,True)
        )
        self.C4 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,True)
        )
        self.C5 = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1),
            nn.Tanh()
        )

        # self.B1 = nn.Sequential(
        #     nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2,True)
        # )
        self.B2 = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,True)
        )
        self.B3 = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,True)
        )
        self.B4 = nn.Sequential(
            # DeformConv2d(16,16,3,padding=1,bias=False,modulation=0.2,True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,True)
        )
        self.B5 = nn.Sequential(
            nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1),
            nn.Tanh()

        )
        self.Fusion = nn.Sequential(
            nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1)
        )

        # self.tanh=nn.Tanh()

    def forward(self, inputir,inputvis):
        ir = self.C1(inputir)
        vis = self.C1(inputvis)
        ir = self.C2(ir)
        vis = self.B2(vis)
        ir = self.C3(ir)
        vis = self.B3(vis)
        ir = self.C4(ir)
        vis = self.B4(vis)
        ir = self.C5(ir)
        vis = self.B5(vis)
        # fusion = self.Fusion(ir+vis)
        return ir,vis,ir+vis



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.C4 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.C5 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True)
        )

        self.C6 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True)
        )


    def forward(self, input):
        x1 = self.C1(input)
        x1 = self.C2(x1)
        x2 = self.C3(x1)
        x2 = self.C4(x2)
        out = self.C5(x2)
        out = self.C6(out)

        return out,x1,x2

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.C2_1 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.C3_1 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )

        self.C4 = nn.Sequential(
            nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
        )

        self.B1 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.B2 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.B2_1 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.B3 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.B3_1 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.B4 = nn.Sequential(
            nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1))

        self.Fusion = nn.Sequential(
            # nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )



    def forward(self, input,x1,x2):
        x = self.C1(input)
        x = F.interpolate(x,size=(x2.shape[2],x2.shape[3]),mode='bilinear',align_corners=True)
        x = self.C2(torch.cat((x,x2),1))
        x = self.C2_1(x)
        x = F.interpolate(x, size=(x1.shape[2], x1.shape[3]),mode='bilinear',align_corners=True)
        x = self.C3(torch.cat((x,x1),1))
        x = self.C3_1(x)
        out = self.C4(x)

        y = self.B1(input)
        y = F.interpolate(y,size=(x2.shape[2],x2.shape[3]),mode='bilinear',align_corners=True)
        y = self.B2(torch.cat((y,x2),1))
        y = self.B2_1(y)
        y = F.interpolate(y, size=(x1.shape[2], x1.shape[3]),mode='bilinear',align_corners=True)
        y = self.B3(torch.cat((y,x1),1))
        y = self.B3_1(y)
        out1 = self.B4(y)
        # fusion = (out+out1)
        fusion = self.Fusion(out+out1)
        return out,out1,fusion


class Three_Encoder(nn.Module):
    def __init__(self):
        super(Three_Encoder, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.C4 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.C5 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True)
        )

        self.C6 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True)
        )


    def forward(self, input):
        x1 = self.C1(input)
        x1 = self.C2(x1)
        x2 = self.C3(x1)
        x2 = self.C4(x2)
        out = self.C5(x2)
        out = self.C6(out)

        return out,x1,x2

class Three_Decoder(nn.Module):
    def __init__(self):
        super(Three_Decoder, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.C2_1 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.C3 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.C3_1 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )

        self.C4 = nn.Sequential(
            nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)
        )
#-------------------------------------------------------------------
        self.B1 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.B2 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.B2_1 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.B3 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.B3_1 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.B4 = nn.Sequential(
            nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)

        )
        # -------------------------------------------------------------------

        self.A1 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.A2 = nn.Sequential(
            nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
        )
        self.A2_1 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.A3 = nn.Sequential(
            nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.A3_1 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.A4 = nn.Sequential(
            nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1)

        )

        self.Fusion = nn.Sequential(
            nn.Tanh()
        )




    def forward(self, input,x1,x2):
        x = self.C1(input)
        x = F.interpolate(x,size=(x2.shape[2],x2.shape[3]),mode='bilinear',align_corners=True)
        x = self.C2(torch.cat((x,x2),1))
        x = self.C2_1(x)
        x = F.interpolate(x, size=(x1.shape[2], x1.shape[3]),mode='bilinear',align_corners=True)
        x = self.C3(torch.cat((x,x1),1))
        x = self.C3_1(x)
        out = self.C4(x)

        y = self.B1(input)
        y = F.interpolate(y,size=(x2.shape[2],x2.shape[3]),mode='bilinear',align_corners=True)
        y = self.B2(torch.cat((y,x2),1))
        y = self.B2_1(y)
        y = F.interpolate(y, size=(x1.shape[2], x1.shape[3]),mode='bilinear',align_corners=True)
        y = self.B3(torch.cat((y,x1),1))
        y = self.B3_1(y)
        out1 = self.B4(y)

        z = self.A1(input)
        z = F.interpolate(z,size=(x2.shape[2],x2.shape[3]),mode='bilinear',align_corners=True)
        z = self.A2(torch.cat((z,x2),1))
        z = self.A2_1(z)
        z = F.interpolate(z, size=(x1.shape[2], x1.shape[3]),mode='bilinear',align_corners=True)
        z = self.A3(torch.cat((z,x1),1))
        z = self.A3_1(z)
        out2= self.A4(z)
        fusion = self.Fusion(out + out1+out2)
        return out,out1,out2,fusion

class E(nn.Module):
    def __init__(self):
        super(E, self).__init__()
        self.C1 = nn.Sequential(
            nn.Tanh()
        )


    def forward(self, input1,input2):
        return self.C1(input1+input2)
