""" Full assembly of the parts to form the complete network """
import math

import torch

from .layers import *
from ..utils import utils
from ..utils.utils import dwt2
from .unet_parts import *


class Teacher(nn.Module):
    def __init__(self, n_channels, n_classes, geo_channels, bilinear=False):
        super(Teacher, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.IR_inc = (DoubleConv(8, 64))
        self.PMW_inc = (DoubleConv(13, 64))
        self.PR_inc = (DoubleConv(176, 64))
        self.geo_inc = (DoubleConv(geo_channels, 64))
        self.x_geo_inc = (DoubleConv(128, 64))
        self.x_geo_inc1 = (DoubleConv(64, 32))
        # self.inc = (DoubleConv(64, 64))

        incha = 64
        self.down1 = (Down(incha, incha*2))
        self.down2 = (Down(incha*2, incha*4))
        self.down3 = (Down(incha*4, incha*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(incha*8, incha*16 // factor))
        self.up1 = (Up(incha*16, incha*16, incha*8 // factor, bilinear))
        self.up2 = (Up(incha*8, incha*8, incha*4 // factor, bilinear))
        self.up3 = (Up(incha*4, incha*4, incha*2 // factor, bilinear))
        self.up4 = (Up(incha*2, incha*2, incha, bilinear))
        self.outc = (OutConv(32, n_classes))

    def forward(self, x, geo):
        IR_x = self.IR_inc(x[:, :8])
        PMW_x = self.PMW_inc(x[:, 8:21])
        PR_x = self.PR_inc(x[:, 21:])

        x_cat = IR_x + PMW_x + PR_x
        # x1 = self.inc(x_cat)
        x2 = self.down1(x_cat)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x_cat)

        geo1 = self.geo_inc(geo)
        # cat
        x_geo_cat = torch.cat((x, geo1), dim=1)
        x_geo_cat = self.x_geo_inc(x_geo_cat)
        x_geo_cat = self.x_geo_inc1(x_geo_cat)
        logits = self.outc(x_geo_cat)
        return logits, [x2, x3, x4]

class PRE_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, fe_block=None):
        super(PRE_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fe_block = fe_block

        factor = 2 if bilinear else 1
        self.up1 = (Up(1024, 1536, 512 // factor, bilinear))
        self.up2 = (Up(512, 768, 256 // factor, bilinear))
        self.up3 = (Up(256, 384, 128 // factor, bilinear))
        self.up4 = (Up(128, 128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))



    def forward(self, x, mask):
        # [x1, x_down4_fuse, x_down8_fuse, x_down16_fuse, x_down32], [x_down4, x_down8, x_down16, x_down4_mae, x_down8_mae, x_down16_mae] = self.fe_block(x, mask)
        [x1, x_down4, x_down8, x_down16, x_down32], [x_down4_mae, x_down8_mae, x_down16_mae], sc_loss = self.fe_block(x, mask)
        x_up16_fuse = self.up1(x_down32, self.cat(x_down16_mae, x_down16))
        x_up8_fuse = self.up2(x_up16_fuse, self.cat(x_down8_mae, x_down8))
        x_up4_fuse = self.up3(x_up8_fuse, self.cat(x_down4_mae, x_down4))
        x = self.up4(x_up4_fuse, x1)

        logits = self.outc(x)

        feat = [x_down4, x_down8, x_down16, x_down4_mae, x_down8_mae, x_down16_mae]
        return logits, feat, sc_loss

    def cat(self, x1, x2):
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x2], dim=1)
        return x

class fe_block_mae_v3(nn.Module):
    def __init__(self, n_channels, bilinear=False, mae_block=None):
        super(fe_block_mae_v3, self).__init__()
        self.n_channels = n_channels
        self.mae_block = mae_block
        if mae_block is not None:
            for i, block in enumerate(mae_block):
                setattr(self, f'mae_block_{i}', block)

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down_mask(64, 128))
        self.down2 = (Down_mask(128, 256))
        self.down3 = (Down_mask(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        self.sa = SpatialAttention(kernel_size=3)
        self.conv1 = nn.Conv2d(192, 128, 1)
        self.conv2 = nn.Conv2d(384, 256, 1)
        self.conv3 = nn.Conv2d(768, 512, 1)


    def forward(self, x, mask):
        x1 = self.inc(x)

        x_down4 = self.down1(x1, F.interpolate(mask, scale_factor=8, mode='nearest'))
        cA, (cH, cV, cD) = dwt2(x1, wavelet='haar')
        HF = torch.cat([cH, cV, cD], dim=1)
        hf_feat4 = self.sa(self.conv1(torch.tensor(HF)))
        x_down4_mae, loss1 = getattr(self, 'mae_block_0')(x_down4, F.interpolate(mask, scale_factor=4, mode='nearest'), hf_feat4)

        x_down8 = self.down2(x_down4, F.interpolate(mask, scale_factor=4, mode='nearest'))
        cA, (cH, cV, cD) = dwt2(x_down4, wavelet='haar')
        HF = torch.cat([cH, cV, cD], dim=1)
        hf_feat8 = self.sa(self.conv2(torch.tensor(HF)))
        x_down8_mae, loss2 = getattr(self, 'mae_block_1')(x_down8, F.interpolate(mask, scale_factor=2, mode='nearest'), hf_feat8)

        x_down16 = self.down3(x_down8, F.interpolate(mask, scale_factor=2, mode='nearest'))
        cA, (cH, cV, cD) = dwt2(x_down8, wavelet='haar')
        HF = torch.cat([cH, cV, cD], dim=1)
        hf_feat16 = self.sa(self.conv3(torch.tensor(HF)))
        x_down16_mae, loss3 = getattr(self, 'mae_block_2')(x_down16, mask, hf_feat16)

        x_down32 = self.down4(x_down16)

        return [x1, x_down4, x_down8, x_down16, x_down32], [x_down4_mae, x_down8_mae, x_down16_mae], loss1 + loss2 + loss3


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, fe_block=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.fe_block = fe_block
        if fe_block is not None:
            self.feature_extraction = fe_block

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 512, 256 // factor, bilinear))
        self.up3 = (Up(256, 256, 128 // factor, bilinear))
        self.up4 = (Up(128, 128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))


    def forward(self, x):
        if self.fe_block is None:
            x1 = self.inc(x)
            x_down4 = self.down1(x1)
            x_down8 = self.down2(x_down4)
            x_down16 = self.down3(x_down8)
            x_down32 = self.down4(x_down16)
        else:
            x1, x_down4, x_down8, x_down16, x_down32 = self.feature_extraction(x)

        x_down16_fuse = self.up1(x_down32, x_down16)
        x_down8_fuse = self.up2(x_down16_fuse, x_down8)
        x_down4_fuse = self.up3(x_down8_fuse, x_down4)
        x = self.up4(x_down4_fuse, x1)

        logits = self.outc(x)

        return logits, [x_down4, x_down8, x_down16]