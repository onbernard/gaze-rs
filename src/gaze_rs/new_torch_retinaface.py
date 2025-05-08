import torch
import torch.nn as nn
import numpy as np
import h5py

from .utils import center_crop


BN_EPS = 1.9999999494757503e-05


class RetinafaceModel(nn.Module):
    def __init__(self):
        super(RetinafaceModel, self).__init__()
        # stage 0
        self.bn_data = nn.BatchNorm2d(3, eps=BN_EPS)
        self.conv0_pad = nn.ZeroPad2d(3)
        self.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=0,
            bias=False
        )
        self.bn0 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.relu0 = nn.ReLU()
        self.pooling0_pad = nn.ZeroPad2d(1)
        self.pooling0 =  nn.MaxPool2d(kernel_size=3, stride=2, padding=0) # [1, 64, 56, 56]
        # stage 1
        ## unit 1
        self.stage1_unit1_bn1 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.stage1_unit1_relu1 = nn.ReLU()
        self.stage1_unit1_conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage1_unit1_bn2 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.stage1_unit1_relu2 = nn.ReLU()
        self.stage1_unit1_conv2_pad = nn.ZeroPad2d(1)
        self.stage1_unit1_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage1_unit1_bn3 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.stage1_unit1_relu3 = nn.ReLU()
        self.stage1_unit1_conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage1_unit1_sc = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        # plus0_v1 : [1, 256, 56, 56]
        ## unit 1 stage 2
        self.stage1_unit2_bn1 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage1_unit2_relu1 = nn.ReLU()
        self.stage1_unit2_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage1_unit2_bn2 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.stage1_unit2_relu2 = nn.ReLU()
        self.stage1_unit2_conv2_pad = nn.ZeroPad2d(1)
        self.stage1_unit2_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage1_unit2_bn3 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.stage1_unit2_relu3 = nn.ReLU()
        self.stage1_unit2_conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        # plus1_v2: 1, 256, 56, 56]
        ## stage1_unit3
        self.stage1_unit3_bn1 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage1_unit3_relu1 = nn.ReLU()
        self.stage1_unit3_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage1_unit3_bn2 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.stage1_unit3_relu2 = nn.ReLU()
        self.stage1_unit3_conv2_pad = nn.ZeroPad2d(1)
        self.stage1_unit3_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage1_unit3_bn3 = nn.BatchNorm2d(64, eps=BN_EPS)
        self.stage1_unit3_relu3 = nn.ReLU()
        self.stage1_unit3_conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        # plus2: [1, 256, 56, 56]
        # stage2
        ## stage2_unit1
        self.stage2_unit1_bn1 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage2_unit1_relu1 = nn.ReLU()
        self.stage2_unit1_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage2_unit1_bn2 = nn.BatchNorm2d(128, eps=BN_EPS)
        self.stage2_unit1_relu2 = nn.ReLU()
        self.stage2_unit1_conv2_pad = nn.ZeroPad2d(1)
        self.stage2_unit1_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(2,2),
            padding=0,
            bias=False,
        )
        self.stage2_unit1_bn3 = nn.BatchNorm2d(128, eps=BN_EPS)
        self.stage2_unit1_relu3 = nn.ReLU()
        self.stage2_unit1_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage2_unit1_sc = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(1,1),
            stride=(2,2),
            padding=0,
            bias=False,
        )
        # plus3: [1, 512, 28, 28]
        ## stage2_unit2
        self.stage2_unit2_bn1 = nn.BatchNorm2d(512, eps=BN_EPS)
        self.stage2_unit2_relu1 = nn.ReLU()
        self.stage2_unit2_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage2_unit2_bn2 = nn.BatchNorm2d(128, eps=BN_EPS)
        self.stage2_unit2_relu2 = nn.ReLU()
        self.stage2_unit2_conv2_pad = nn.ZeroPad2d(1)
        self.stage2_unit2_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        self.stage2_unit2_bn3 = nn.BatchNorm2d(128, eps=BN_EPS)
        self.stage2_unit2_relu3 = nn.ReLU()
        self.stage2_unit2_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding=0,
            bias=False,
        )
        # plus4: [1, 512, 80, 80]
        ## stage2_unit3
        self.stage2_unit3_bn1 = nn.BatchNorm2d(512, eps=BN_EPS)
        self.stage2_unit3_relu1 = nn.ReLU()
        self.stage2_unit3_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=(1,1),
            stride=(1,1),
        )
        self.stage2_unit3_bn2 = nn.BatchNorm2d(128, eps=BN_EPS)
        self.stage2_unit3_relu2 = nn.ReLU()
        self.stage2_unit3_conv2_pad = nn.ZeroPad2d(1)
        self.stage2_unit3_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
        )
        self.stage2_unit3_bn3 = nn.BatchNorm2d(128, eps=BN_EPS)
        self.stage2_unit3_relu3 = nn.ReLU()
        self.stage2_unit3_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
        )
        # plus5: [1, 512, 28, 28]
        ## stage2_unit4
        self.stage2_unit4_bn1 = nn.BatchNorm2d(512, eps=BN_EPS)
        self.stage2_unit4_relu1 = nn.ReLU()
        self.stage2_unit4_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=(1,1),
            stride=(1,1),
        )
        self.stage2_unit4_bn2 = nn.BatchNorm2d(128, eps=BN_EPS)
        self.stage2_unit4_relu2 = nn.ReLU()
        self.stage2_unit4_conv2_pad = nn.ZeroPad2d(1)
        self.stage2_unit4_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
        )
        self.stage2_unit4_bn3 = nn.BatchNorm2d(128, eps=BN_EPS)
        self.stage2_unit4_relu3 = nn.ReLU()
        self.stage2_unit4_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
        )
        # plus6: [1, 512, 28, 28]
        self.stage3_unit1_bn1 = nn.BatchNorm2d(512, eps=BN_EPS)
        self.stage3_unit1_relu1 = nn.ReLU()
        self.stage3_unit1_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
        )
        self.stage3_unit1_bn2 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit1_relu2 = nn.ReLU()
        self.stage3_unit1_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit1_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(2,2),
        )
        self.stage3_unit1_bn3 = nn.BatchNorm2d(256, eps=BN_EPS )
        self.stage3_unit1_relu3 = nn.ReLU()
        self.stage3_unit1_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1)
        )
        self.stage3_unit1_sc = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(2,2)
        )
        # plus7: [1, 1024, 14, 14]
        self.stage3_unit2_bn1 = nn.BatchNorm2d(1024, eps=BN_EPS)
        self.stage3_unit2_relu1 = nn.ReLU()
        self.stage3_unit2_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1)
        )
        self.stage3_unit2_bn2 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit2_relu2 = nn.ReLU()
        self.stage3_unit2_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit2_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1)
        )
        self.stage3_unit2_bn3 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit2_relu3 = nn.ReLU()
        self.stage3_unit2_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1)
        )
        # plus8: [1, 1024, 14, 14]
        ## stage3_unit3
        self.stage3_unit3_bn1 = nn.BatchNorm2d(1024, eps=BN_EPS)
        self.stage3_unit3_relu1 = nn.ReLU()
        self.stage3_unit3_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1)
        )
        self.stage3_unit3_bn2 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit3_relu2 = nn.ReLU()
        self.stage3_unit3_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit3_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1)
        )
        self.stage3_unit3_bn3 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit3_relu3 = nn.ReLU()
        self.stage3_unit3_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1)
        )
        # plus9: [1, 1024, 14, 14]
        ## stage3_unit4
        self.stage3_unit4_bn1 = nn.BatchNorm2d(1024, eps=BN_EPS)
        self.stage3_unit4_relu1 = nn.ReLU()
        self.stage3_unit4_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1)
        )
        self.stage3_unit4_bn2 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit4_relu2 = nn.ReLU()
        self.stage3_unit4_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit4_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1)
        )
        self.stage3_unit4_bn3 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit4_relu3 = nn.ReLU()
        self.stage3_unit4_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1)
        )
        # plus10: [1, 1024, 14, 14]
        ## stage3_unit5
        self.stage3_unit5_bn1 = nn.BatchNorm2d(1024, eps=BN_EPS)
        self.stage3_unit5_relu1 = nn.ReLU()
        self.stage3_unit5_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1)
        )
        self.stage3_unit5_bn2 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit5_relu2 = nn.ReLU()
        self.stage3_unit5_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit5_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1)
        )
        self.stage3_unit5_bn3 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit5_relu3 = nn.ReLU()
        self.stage3_unit5_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1)
        )
        # plus11: [1, 1024, 14, 14]
        ## stage3_unit6
        self.stage3_unit6_bn1 = nn.BatchNorm2d(1024, eps=BN_EPS)
        self.stage3_unit6_relu1 = nn.ReLU()
        self.stage3_unit6_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1)
        )
        self.stage3_unit6_bn2 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit6_relu2 = nn.ReLU()
        self.stage3_unit6_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit6_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1)
        )
        self.stage3_unit6_bn3 = nn.BatchNorm2d(256, eps=BN_EPS)
        self.stage3_unit6_relu3 = nn.ReLU()
        self.stage3_unit6_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1)
        )
        # plus12: [1, 1024, 14, 14]
        # stage4
        ## stage4_unit1
        self.stage4_unit1_bn1 = nn.BatchNorm2d(1024, eps=BN_EPS)
        self.stage4_unit1_relu1 = nn.ReLU()
        self.stage4_unit1_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
        )
        self.stage4_unit1_bn2 = nn.BatchNorm2d(512, eps=BN_EPS)
        self.stage4_unit1_relu2 = nn.ReLU()
        self.stage4_unit1_conv2_pad = nn.ZeroPad2d(1)
        self.stage4_unit1_conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3,3),
            stride=(2,2),
        )
        self.stage4_unit1_bn3 = nn.BatchNorm2d(512, eps=BN_EPS )
        self.stage4_unit1_relu3 = nn.ReLU()
        self.stage4_unit1_conv3 = nn.Conv2d(
            in_channels=512,
            out_channels=2048,
            kernel_size=(1,1),
            stride=(1,1)
        )
        self.stage4_unit1_sc = nn.Conv2d(
            in_channels=1024,
            out_channels=2048,
            kernel_size=(1,1),
            stride=(2,2)
        )
        # plus13: [1, 2048, 7, 7]
        ## stage4_unit2
        self.stage4_unit2_bn1 = nn.BatchNorm2d(2048, eps=BN_EPS)
        self.stage4_unit2_relu1 = nn.ReLU()
        self.stage4_unit2_conv1 = nn.Conv2d(
            in_channels=2048,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1)
        )
        self.stage4_unit2_bn2 = nn.BatchNorm2d(512, eps=BN_EPS)
        self.stage4_unit2_relu2 = nn.ReLU()
        self.stage4_unit2_conv2_pad = nn.ZeroPad2d(1)
        self.stage4_unit2_conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3,3),
            stride=(1,1)
        )
        self.stage4_unit2_bn3 = nn.BatchNorm2d(512, eps=BN_EPS)
        self.stage4_unit2_relu3 = nn.ReLU()
        self.stage4_unit2_conv3 = nn.Conv2d(
            in_channels=512,
            out_channels=2048,
            kernel_size=(1,1),
            stride=(1,1)
        )
        # plus14: [1, 2048, 7, 7]
        ## stage4_unit3
        self.stage4_unit3_bn1 = nn.BatchNorm2d(2048, eps=BN_EPS)
        self.stage4_unit3_relu1 = nn.ReLU()
        self.stage4_unit3_conv1 = nn.Conv2d(
            in_channels=2048,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1)
        )
        self.stage4_unit3_bn2 = nn.BatchNorm2d(512, eps=BN_EPS)
        self.stage4_unit3_relu2 = nn.ReLU()
        self.stage4_unit3_conv2_pad = nn.ZeroPad2d(1)
        self.stage4_unit3_conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3,3),
            stride=(1,1)
        )
        self.stage4_unit3_bn3 = nn.BatchNorm2d(512, eps=BN_EPS)
        self.stage4_unit3_relu3 = nn.ReLU()
        self.stage4_unit3_conv3 = nn.Conv2d(
            in_channels=512,
            out_channels=2048,
            kernel_size=(1,1),
            stride=(1,1)
        )
        # plus15: [1, 2048, 7, 7]
        self.bn1 = nn.BatchNorm2d(2048, eps=BN_EPS)
        self.relu1 = nn.ReLU()
        # ssh_m3
        self.ssh_c3_lateral = nn.Conv2d(
            in_channels=2048,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        self.ssh_c3_lateral_bn = nn.BatchNorm2d(256, eps=BN_EPS)
        self.ssh_c3_lateral_relu = nn.ReLU()
        #
        self.ssh_c3_up = nn.Upsample(scale_factor=2, mode="nearest")
        # ssh_c3_up: [1, 256, 14, 14]
        self.ssh_m3_det_context_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_context_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m3_det_context_conv1_bn = nn.BatchNorm2d(128, eps=BN_EPS)
        self.ssh_m3_det_context_conv1_relu = nn.ReLU() # [1, 128, 7, 7]
        #
        self.ssh_m3_det_context_conv3_1_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_context_conv3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m3_det_context_conv3_1_bn = nn.BatchNorm2d(128, eps=BN_EPS)
        self.ssh_m3_det_context_conv3_1_relu = nn.ReLU()
        self.ssh_m3_det_context_conv3_2_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_context_conv3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m3_det_context_conv3_2_bn = nn.BatchNorm2d(128, eps=BN_EPS) # [1, 128, 7, 7]
        #
        self.ssh_m3_det_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m3_det_conv1_bn = nn.BatchNorm2d(256, eps=BN_EPS) # [1, 256, 7, 7]
        #
        self.ssh_m3_det_context_conv2_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_context_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m3_det_context_conv2_bn = nn.BatchNorm2d(128, eps=BN_EPS) # [1, 128, 7, 7]
        # ssh_m2
        self.ssh_c2_lateral = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        self.ssh_c2_lateral_bn = nn.BatchNorm2d(256, eps=BN_EPS)
        self.ssh_c2_lateral_relu = nn.ReLU()
        # plus0_v2: [1, 256, 14, 14]
        self.ssh_c2_aggr_pad = nn.ZeroPad2d(1)
        self.ssh_c2_aggr = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_c2_aggr_bn = nn.BatchNorm2d(256, eps=BN_EPS)
        self.ssh_c2_aggr_relu = nn.ReLU() # [1, 256, 14, 14]
        #
        self.ssh_m2_red_up = nn.Upsample(scale_factor=2, mode="nearest")
        #
        self.ssh_m2_det_context_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m2_det_context_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m2_det_context_conv1_bn = nn.BatchNorm2d(128, eps=BN_EPS)
        self.ssh_m2_det_context_conv1_relu = nn.ReLU() # [1, 128, 14, 14]
        #
        self.ssh_m2_det_context_conv3_1_pad = nn.ZeroPad2d(1)
        self.ssh_m2_det_context_conv3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m2_det_context_conv3_1_bn = nn.BatchNorm2d(128, eps=BN_EPS)
        self.ssh_m2_det_context_conv3_1_relu = nn.ReLU()
        self.ssh_m2_det_context_conv3_2_pad = nn.ZeroPad2d(1)
        self.ssh_m2_det_context_conv3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m2_det_context_conv3_2_bn = nn.BatchNorm2d(128, eps=BN_EPS) # [1, 128, 14, 14]
        #
        self.ssh_m2_det_context_conv2_pad = nn.ZeroPad2d(1)
        self.ssh_m2_det_context_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m2_det_context_conv2_bn = nn.BatchNorm2d(128, eps=BN_EPS) # [1, 128, 14, 14]
        #
        self.ssh_m2_det_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m2_det_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m2_det_conv1_bn = nn.BatchNorm2d(256, eps=BN_EPS) # [1, 256, 14, 14]
        # ssh_m1
        self.ssh_m1_red_conv = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m1_red_conv_bn = nn.BatchNorm2d(256, eps=BN_EPS)
        self.ssh_m1_red_conv_relu = nn.ReLU()
        # plus1_v1: [1, 256, 28, 28]
        self.ssh_c1_aggr_pad = nn.ZeroPad2d(1)
        self.ssh_c1_aggr = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_c1_aggr_bn = nn.BatchNorm2d(256, eps=BN_EPS)
        self.ssh_c1_aggr_relu = nn.ReLU() # [1, 256, 28, 28]
        #
        self.ssh_m1_det_context_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_context_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m1_det_context_conv1_bn = nn.BatchNorm2d(128, eps=BN_EPS)
        self.ssh_m1_det_context_conv1_relu = nn.ReLU()
        #
        self.ssh_m1_det_context_conv3_1_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_context_conv3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m1_det_context_conv3_1_bn = nn.BatchNorm2d(128, eps=BN_EPS)
        self.ssh_m1_det_context_conv3_1_relu = nn.ReLU()
        self.ssh_m1_det_context_conv3_2_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_context_conv3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m1_det_context_conv3_2_bn = nn.BatchNorm2d(128, eps=BN_EPS)
        #
        self.ssh_m1_det_context_conv2_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_context_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m1_det_context_conv2_bn = nn.BatchNorm2d(128, eps=BN_EPS) # [1, 128, 28, 28]
        #
        self.ssh_m1_det_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_m1_det_conv1_bn = nn.BatchNorm2d(256, eps=BN_EPS)
        # ssh_m1_det_concat: [1, 512, 28, 28]
        self.ssh_m1_det_concat_relu = nn.ReLU()
        #
        self.face_rpn_bbox_pred_stride8 = nn.Conv2d(
            in_channels=512,
            out_channels=8,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        self.face_rpn_cls_score_stride8 = nn.Conv2d(
            in_channels=512,
            out_channels=4,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        # face_rpn_cls_score_reshape_stride8: [1, 2, 28, 2, 28]
        self.face_rpn_cls_prob_stride8 = nn.Softmax(dim=3)
        #
        self.face_rpn_landmark_pred_stride8 = nn.Conv2d(
            in_channels=512,
            out_channels=20,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        # stride16
        self.ssh_m2_det_concat_relu = nn.ReLU()
        self.face_rpn_bbox_pred_stride16 = nn.Conv2d(
            in_channels=512,
            out_channels=8,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        self.face_rpn_cls_score_stride16 = nn.Conv2d(
            in_channels=512,
            out_channels=4,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        # face_rpn_cls_score_reshape_stride8: [1, 2, 28, 2, 28]
        self.face_rpn_cls_prob_stride16 = nn.Softmax(dim=3)
        #
        self.face_rpn_landmark_pred_stride16 = nn.Conv2d(
            in_channels=512,
            out_channels=20,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        # stride32
        self.ssh_m3_det_concat_relu = nn.ReLU()
        self.face_rpn_bbox_pred_stride32 = nn.Conv2d(
            in_channels=512,
            out_channels=8,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        self.face_rpn_cls_score_stride32 = nn.Conv2d(
            in_channels=512,
            out_channels=4,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )
        # face_rpn_cls_score_reshape_stride8: [1, 2, 28, 2, 28]
        self.face_rpn_cls_prob_stride32 = nn.Softmax(dim=1)
        #
        self.face_rpn_landmark_pred_stride32 = nn.Conv2d(
            in_channels=512,
            out_channels=20,
            kernel_size=(1,1),
            stride=(1,1),
            bias=True,
        )


    def forward(self, x):
        # stage 0
        bn_data = self.bn_data(x)
        conv0_pad = self.conv0_pad(bn_data)
        conv0 = self.conv0(conv0_pad)
        bn0 = self.bn0(conv0)
        relu0 = self.relu0(bn0)
        pooling0_pad = self.pooling0_pad(relu0)
        pooling0 = self.pooling0(pooling0_pad)
        # stage1
        ## stage1_unit1
        stage1_unit1_bn1 = self.stage1_unit1_bn1(pooling0)
        stage1_unit1_relu1 = self.stage1_unit1_relu1(stage1_unit1_bn1)
        stage1_unit1_conv1 = self.stage1_unit1_conv1(stage1_unit1_relu1)
        stage1_unit1_bn2 = self.stage1_unit1_bn2(stage1_unit1_conv1)
        stage1_unit1_relu2 = self.stage1_unit1_relu2(stage1_unit1_bn2)
        stage1_unit1_conv2_pad = self.stage1_unit1_conv2_pad(stage1_unit1_relu2)
        stage1_unit1_conv2 = self.stage1_unit1_conv2(stage1_unit1_conv2_pad)
        stage1_unit1_bn3 = self.stage1_unit1_bn3(stage1_unit1_conv2)
        stage1_unit1_relu3 = self.stage1_unit1_relu3(stage1_unit1_bn3)
        stage1_unit1_conv3 = self.stage1_unit1_conv3(stage1_unit1_relu3)
        stage1_unit1_sc = self.stage1_unit1_sc(stage1_unit1_relu1)
        plus0_v1 = stage1_unit1_conv3 + stage1_unit1_sc
        ## stage1_unit2
        stage1_unit2_bn1 = self.stage1_unit2_bn1(plus0_v1)
        stage1_unit2_relu1 = self.stage1_unit2_relu1(stage1_unit2_bn1)
        stage1_unit2_conv1 = self.stage1_unit2_conv1(stage1_unit2_relu1)
        stage1_unit2_bn2 = self.stage1_unit2_bn2(stage1_unit2_conv1)
        stage1_unit2_relu2 = self.stage1_unit2_relu2(stage1_unit2_bn2)
        stage1_unit2_conv2_pad = self.stage1_unit2_conv2_pad(stage1_unit2_relu2)
        stage1_unit2_conv2 = self.stage1_unit2_conv2(stage1_unit2_conv2_pad)
        stage1_unit2_bn3 = self.stage1_unit2_bn3(stage1_unit2_conv2)
        stage1_unit2_relu3 = self.stage1_unit2_relu3(stage1_unit2_bn3)
        stage1_unit2_conv3 = self.stage1_unit2_conv3(stage1_unit2_relu3)
        plus1_v2 = plus0_v1 + stage1_unit2_conv3
        ## stage1_unit3
        stage1_unit3_bn1 = self.stage1_unit3_bn1(plus1_v2)
        stage1_unit3_relu1 = self.stage1_unit3_relu1(stage1_unit3_bn1)
        stage1_unit3_conv1 = self.stage1_unit3_conv1(stage1_unit3_relu1)
        stage1_unit3_bn2 = self.stage1_unit3_bn2(stage1_unit3_conv1)
        stage1_unit3_relu2 = self.stage1_unit3_relu2(stage1_unit3_bn2)
        stage1_unit3_conv2_pad = self.stage1_unit3_conv2_pad(stage1_unit3_relu2)
        stage1_unit3_conv2 = self.stage1_unit3_conv2(stage1_unit3_conv2_pad)
        stage1_unit3_bn3 = self.stage1_unit3_bn3(stage1_unit3_conv2)
        stage1_unit3_relu3 = self.stage1_unit3_relu3(stage1_unit3_bn3)
        stage1_unit3_conv3 = self.stage1_unit3_conv3(stage1_unit3_relu3)
        plus2 = plus1_v2 + stage1_unit3_conv3
        # stage2
        ## stage2_unit1
        stage2_unit1_bn1 = self.stage2_unit1_bn1(plus2)
        stage2_unit1_relu1 = self.stage2_unit1_relu1(stage2_unit1_bn1)
        stage2_unit1_conv1 = self.stage2_unit1_conv1(stage2_unit1_relu1)
        stage2_unit1_bn2 = self.stage2_unit1_bn2(stage2_unit1_conv1)
        stage2_unit1_relu2 = self.stage2_unit1_relu2(stage2_unit1_bn2)
        stage2_unit1_conv2_pad = self.stage2_unit1_conv2_pad(stage2_unit1_relu2)
        stage2_unit1_conv2 = self.stage2_unit1_conv2(stage2_unit1_conv2_pad)
        stage2_unit1_bn3 = self.stage2_unit1_bn3(stage2_unit1_conv2)
        stage2_unit1_relu3 = self.stage2_unit1_relu3(stage2_unit1_bn3)
        stage2_unit1_conv3 = self.stage2_unit1_conv3(stage2_unit1_relu3)
        stage2_unit1_sc = self.stage2_unit1_sc(stage2_unit1_relu1)
        plus3 = stage2_unit1_conv3 + stage2_unit1_sc
        ## stage2_unit2
        stage2_unit2_bn1 = self.stage2_unit2_bn1(plus3)
        stage2_unit2_relu1 = self.stage2_unit2_relu1(stage2_unit2_bn1)
        stage2_unit2_conv1 = self.stage2_unit2_conv1(stage2_unit2_relu1)
        stage2_unit2_bn2 = self.stage2_unit2_bn2(stage2_unit2_conv1)
        stage2_unit2_relu2 = self.stage2_unit2_relu2(stage2_unit2_bn2)
        stage2_unit2_conv2_pad = self.stage2_unit2_conv2_pad(stage2_unit2_relu2)
        stage2_unit2_conv2 = self.stage2_unit2_conv2(stage2_unit2_conv2_pad)
        stage2_unit2_bn3 = self.stage2_unit2_bn3(stage2_unit2_conv2)
        stage2_unit2_relu3 = self.stage2_unit2_relu3(stage2_unit2_bn3)
        stage2_unit2_conv3 = self.stage2_unit2_conv3(stage2_unit2_relu3)
        plus4 = stage2_unit2_conv3 + plus3
        ## stage2_unit3
        stage2_unit3_bn1 = self.stage2_unit3_bn1(plus4)
        stage2_unit3_relu1 = self.stage2_unit3_relu1(stage2_unit3_bn1)
        stage2_unit3_conv1 = self.stage2_unit3_conv1(stage2_unit3_relu1)
        stage2_unit3_bn2 = self.stage2_unit3_bn2(stage2_unit3_conv1)
        stage2_unit3_relu2 = self.stage2_unit3_relu2(stage2_unit3_bn2)
        stage2_unit3_conv2_pad = self.stage2_unit3_conv2_pad(stage2_unit3_relu2)
        stage2_unit3_conv2 = self.stage2_unit3_conv2(stage2_unit3_conv2_pad)
        stage2_unit3_bn3 = self.stage2_unit3_bn3(stage2_unit3_conv2)
        stage2_unit3_relu3 = self.stage2_unit3_relu3(stage2_unit3_bn3)
        stage2_unit3_conv3 = self.stage2_unit3_conv3(stage2_unit3_relu3)
        plus5 = plus4 + stage2_unit3_conv3
        ## stage2_unit4
        stage2_unit4_bn1 = self.stage2_unit4_bn1(plus5)
        stage2_unit4_relu1 = self.stage2_unit4_relu1(stage2_unit4_bn1)
        stage2_unit4_conv1 = self.stage2_unit4_conv1(stage2_unit4_relu1)
        stage2_unit4_bn2 = self.stage2_unit4_bn2(stage2_unit4_conv1)
        stage2_unit4_relu2 = self.stage2_unit4_relu2(stage2_unit4_bn2)
        stage2_unit4_conv2_pad = self.stage2_unit4_conv2_pad(stage2_unit4_relu2)
        stage2_unit4_conv2 = self.stage2_unit4_conv2(stage2_unit4_conv2_pad)
        stage2_unit4_bn3 = self.stage2_unit4_bn3(stage2_unit4_conv2)
        stage2_unit4_relu3 = self.stage2_unit4_relu3(stage2_unit4_bn3)
        stage2_unit4_conv3 = self.stage2_unit4_conv3(stage2_unit4_relu3)
        plus6 = plus5 + stage2_unit4_conv3
        # stage3
        ## stage3_unit1
        stage3_unit1_bn1 = self.stage3_unit1_bn1(plus6)
        stage3_unit1_relu1 = self.stage3_unit1_relu1(stage3_unit1_bn1)
        stage3_unit1_conv1 = self.stage3_unit1_conv1(stage3_unit1_relu1)
        stage3_unit1_bn2 = self.stage3_unit1_bn2(stage3_unit1_conv1)
        stage3_unit1_relu2 = self.stage3_unit1_relu2(stage3_unit1_bn2)
        stage3_unit1_conv2_pad = self.stage3_unit1_conv2_pad(stage3_unit1_relu2)
        stage3_unit1_conv2 = self.stage3_unit1_conv2(stage3_unit1_conv2_pad)
        stage3_unit1_bn3 = self.stage3_unit1_bn3(stage3_unit1_conv2)
        stage3_unit1_relu3 = self.stage3_unit1_relu3(stage3_unit1_bn3)
        stage3_unit1_conv3 = self.stage3_unit1_conv3(stage3_unit1_relu3)
        stage3_unit1_sc = self.stage3_unit1_sc(stage3_unit1_relu1)
        plus7 = stage3_unit1_conv3 + stage3_unit1_sc
        ## stage3_unit2
        stage3_unit2_bn1 = self.stage3_unit2_bn1(plus7)
        stage3_unit2_relu1 = self.stage3_unit2_relu1(stage3_unit2_bn1)
        stage3_unit2_conv1 = self.stage3_unit2_conv1(stage3_unit2_relu1)
        stage3_unit2_bn2 = self.stage3_unit2_bn2(stage3_unit2_conv1)
        stage3_unit2_relu2 = self.stage3_unit2_relu2(stage3_unit2_bn2)
        stage3_unit2_conv2_pad = self.stage3_unit2_conv2_pad(stage3_unit2_relu2)
        stage3_unit2_conv2 = self.stage3_unit2_conv2(stage3_unit2_conv2_pad)
        stage3_unit2_bn3 = self.stage3_unit2_bn3(stage3_unit2_conv2)
        stage3_unit2_relu3 = self.stage3_unit2_relu3(stage3_unit2_bn3)
        stage3_unit2_conv3 = self.stage3_unit2_conv3(stage3_unit2_relu3)
        plus8 = plus7 + stage3_unit2_conv3
        ## stage3_unit3
        stage3_unit3_bn1 = self.stage3_unit3_bn1(plus8)
        stage3_unit3_relu1 = self.stage3_unit3_relu1(stage3_unit3_bn1)
        stage3_unit3_conv1 = self.stage3_unit3_conv1(stage3_unit3_relu1)
        stage3_unit3_bn2 = self.stage3_unit3_bn2(stage3_unit3_conv1)
        stage3_unit3_relu2 = self.stage3_unit3_relu2(stage3_unit3_bn2)
        stage3_unit3_conv2_pad = self.stage3_unit3_conv2_pad(stage3_unit3_relu2)
        stage3_unit3_conv2 = self.stage3_unit3_conv2(stage3_unit3_conv2_pad)
        stage3_unit3_bn3 = self.stage3_unit3_bn3(stage3_unit3_conv2)
        stage3_unit3_relu3 = self.stage3_unit3_relu3(stage3_unit3_bn3)
        stage3_unit3_conv3 = self.stage3_unit3_conv3(stage3_unit3_relu3)
        plus9 = plus8 + stage3_unit3_conv3
        ## stage3_unit4
        stage3_unit4_bn1 = self.stage3_unit4_bn1(plus9)
        stage3_unit4_relu1 = self.stage3_unit4_relu1(stage3_unit4_bn1)
        stage3_unit4_conv1 = self.stage3_unit4_conv1(stage3_unit4_relu1)
        stage3_unit4_bn2 = self.stage3_unit4_bn2(stage3_unit4_conv1)
        stage3_unit4_relu2 = self.stage3_unit4_relu2(stage3_unit4_bn2)
        stage3_unit4_conv2_pad = self.stage3_unit4_conv2_pad(stage3_unit4_relu2)
        stage3_unit4_conv2 = self.stage3_unit4_conv2(stage3_unit4_conv2_pad)
        stage3_unit4_bn3 = self.stage3_unit4_bn3(stage3_unit4_conv2)
        stage3_unit4_relu3 = self.stage3_unit4_relu3(stage3_unit4_bn3)
        stage3_unit4_conv3 = self.stage3_unit4_conv3(stage3_unit4_relu3)
        plus10 = plus9 + stage3_unit4_conv3
        ## stage3_unit5
        stage3_unit5_bn1 = self.stage3_unit5_bn1(plus10)
        stage3_unit5_relu1 = self.stage3_unit5_relu1(stage3_unit5_bn1)
        stage3_unit5_conv1 = self.stage3_unit5_conv1(stage3_unit5_relu1)
        stage3_unit5_bn2 = self.stage3_unit5_bn2(stage3_unit5_conv1)
        stage3_unit5_relu2 = self.stage3_unit5_relu2(stage3_unit5_bn2)
        stage3_unit5_conv2_pad = self.stage3_unit5_conv2_pad(stage3_unit5_relu2)
        stage3_unit5_conv2 = self.stage3_unit5_conv2(stage3_unit5_conv2_pad)
        stage3_unit5_bn3 = self.stage3_unit5_bn3(stage3_unit5_conv2)
        stage3_unit5_relu3 = self.stage3_unit5_relu3(stage3_unit5_bn3)
        stage3_unit5_conv3 = self.stage3_unit5_conv3(stage3_unit5_relu3)
        plus11 = plus10 + stage3_unit5_conv3
        ## stage3_unit6
        stage3_unit6_bn1 = self.stage3_unit6_bn1(plus11)
        stage3_unit6_relu1 = self.stage3_unit6_relu1(stage3_unit6_bn1)
        stage3_unit6_conv1 = self.stage3_unit6_conv1(stage3_unit6_relu1)
        stage3_unit6_bn2 = self.stage3_unit6_bn2(stage3_unit6_conv1)
        stage3_unit6_relu2 = self.stage3_unit6_relu2(stage3_unit6_bn2)
        stage3_unit6_conv2_pad = self.stage3_unit6_conv2_pad(stage3_unit6_relu2)
        stage3_unit6_conv2 = self.stage3_unit6_conv2(stage3_unit6_conv2_pad)
        stage3_unit6_bn3 = self.stage3_unit6_bn3(stage3_unit6_conv2)
        stage3_unit6_relu3 = self.stage3_unit6_relu3(stage3_unit6_bn3)
        stage3_unit6_conv3 = self.stage3_unit6_conv3(stage3_unit6_relu3)
        plus12 = plus11 + stage3_unit6_conv3
        # stage4
        ## stage4_unit1
        stage4_unit1_bn1 = self.stage4_unit1_bn1(plus12)
        stage4_unit1_relu1 = self.stage4_unit1_relu1(stage4_unit1_bn1)
        stage4_unit1_conv1 = self.stage4_unit1_conv1(stage4_unit1_relu1)
        stage4_unit1_bn2 = self.stage4_unit1_bn2(stage4_unit1_conv1)
        stage4_unit1_relu2 = self.stage4_unit1_relu2(stage4_unit1_bn2)
        stage4_unit1_conv2_pad = self.stage4_unit1_conv2_pad(stage4_unit1_relu2)
        stage4_unit1_conv2 = self.stage4_unit1_conv2(stage4_unit1_conv2_pad)
        stage4_unit1_bn3 = self.stage4_unit1_bn3(stage4_unit1_conv2)
        stage4_unit1_relu3 = self.stage4_unit1_relu3(stage4_unit1_bn3)
        stage4_unit1_conv3 = self.stage4_unit1_conv3(stage4_unit1_relu3)
        stage4_unit1_sc = self.stage4_unit1_sc(stage4_unit1_relu1)
        plus13 = stage4_unit1_conv3 + stage4_unit1_sc
        ## stage4_unit2
        stage4_unit2_bn1 = self.stage4_unit2_bn1(plus13)
        stage4_unit2_relu1 = self.stage4_unit2_relu1(stage4_unit2_bn1)
        stage4_unit2_conv1 = self.stage4_unit2_conv1(stage4_unit2_relu1)
        stage4_unit2_bn2 = self.stage4_unit2_bn2(stage4_unit2_conv1)
        stage4_unit2_relu2 = self.stage4_unit2_relu2(stage4_unit2_bn2)
        stage4_unit2_conv2_pad = self.stage4_unit2_conv2_pad(stage4_unit2_relu2)
        stage4_unit2_conv2 = self.stage4_unit2_conv2(stage4_unit2_conv2_pad)
        stage4_unit2_bn3 = self.stage4_unit2_bn3(stage4_unit2_conv2)
        stage4_unit2_relu3 = self.stage4_unit2_relu3(stage4_unit2_bn3)
        stage4_unit2_conv3 = self.stage4_unit2_conv3(stage4_unit2_relu3)
        plus14 = plus13 + stage4_unit2_conv3
        ## stage4_unit3
        stage4_unit3_bn1 = self.stage4_unit3_bn1(plus14)
        stage4_unit3_relu1 = self.stage4_unit3_relu1(stage4_unit3_bn1)
        stage4_unit3_conv1 = self.stage4_unit3_conv1(stage4_unit3_relu1)
        stage4_unit3_bn2 = self.stage4_unit3_bn2(stage4_unit3_conv1)
        stage4_unit3_relu2 = self.stage4_unit3_relu2(stage4_unit3_bn2)
        stage4_unit3_conv2_pad = self.stage4_unit3_conv2_pad(stage4_unit3_relu2)
        stage4_unit3_conv2 = self.stage4_unit3_conv2(stage4_unit3_conv2_pad)
        stage4_unit3_bn3 = self.stage4_unit3_bn3(stage4_unit3_conv2)
        stage4_unit3_relu3 = self.stage4_unit3_relu3(stage4_unit3_bn3)
        stage4_unit3_conv3 = self.stage4_unit3_conv3(stage4_unit3_relu3)
        plus15 = plus14 + stage4_unit3_conv3
        #
        bn1 = self.bn1(plus15)
        relu1 = self.relu1(bn1)
        # ssh_m3
        ssh_c3_lateral = self.ssh_c3_lateral(relu1)
        ssh_c3_lateral_bn = self.ssh_c3_lateral_bn(ssh_c3_lateral)
        ssh_c3_lateral_relu = self.ssh_c3_lateral_relu(ssh_c3_lateral_bn)
        #
        ssh_c3_up = self.ssh_c3_up(ssh_c3_lateral_relu)
        #
        ssh_m3_det_context_conv1_pad = self.ssh_m3_det_context_conv1_pad(ssh_c3_lateral_relu)
        ssh_m3_det_context_conv1 = self.ssh_m3_det_context_conv1(ssh_m3_det_context_conv1_pad)
        ssh_m3_det_context_conv1_bn = self.ssh_m3_det_context_conv1_bn(ssh_m3_det_context_conv1)
        ssh_m3_det_context_conv1_relu = self.ssh_m3_det_context_conv1_relu(ssh_m3_det_context_conv1_bn)
        #
        ssh_m3_det_context_conv3_1_pad = self.ssh_m3_det_context_conv3_1_pad(ssh_m3_det_context_conv1_relu)
        ssh_m3_det_context_conv3_1 = self.ssh_m3_det_context_conv3_1(ssh_m3_det_context_conv3_1_pad)
        ssh_m3_det_context_conv3_1_bn = self.ssh_m3_det_context_conv3_1_bn(ssh_m3_det_context_conv3_1)
        ssh_m3_det_context_conv3_1_relu = self.ssh_m3_det_context_conv3_1_relu(ssh_m3_det_context_conv3_1_bn)
        ssh_m3_det_context_conv3_2_pad = self.ssh_m3_det_context_conv3_2_pad(ssh_m3_det_context_conv3_1_relu)
        ssh_m3_det_context_conv3_2 = self.ssh_m3_det_context_conv3_2(ssh_m3_det_context_conv3_2_pad)
        ssh_m3_det_context_conv3_2_bn = self.ssh_m3_det_context_conv3_2_bn(ssh_m3_det_context_conv3_2)
        #
        ssh_m3_det_conv1_pad = self.ssh_m3_det_conv1_pad(ssh_c3_lateral_relu)
        ssh_m3_det_conv1 = self.ssh_m3_det_conv1(ssh_m3_det_conv1_pad)
        ssh_m3_det_conv1_bn = self.ssh_m3_det_conv1_bn(ssh_m3_det_conv1)
        #
        ssh_m3_det_context_conv2_pad = self.ssh_m3_det_context_conv2_pad(ssh_m3_det_context_conv1_relu)
        ssh_m3_det_context_conv2 = self.ssh_m3_det_context_conv2(ssh_m3_det_context_conv2_pad)
        ssh_m3_det_context_conv2_bn = self.ssh_m3_det_context_conv2_bn(ssh_m3_det_context_conv2)
        # ssh_m2
        ssh_c2_lateral = self.ssh_c2_lateral(stage4_unit1_relu2)
        ssh_c2_lateral_bn = self.ssh_c2_lateral_bn(ssh_c2_lateral)
        ssh_c2_lateral_relu = self.ssh_c2_lateral_relu(ssh_c2_lateral_bn)
        crop0 = center_crop(ssh_c3_up, ssh_c2_lateral_relu)
        plus0_v2 = crop0 + ssh_c2_lateral_relu
        #
        ssh_c2_aggr_pad = self.ssh_c2_aggr_pad(plus0_v2)
        ssh_c2_aggr = self.ssh_c2_aggr(ssh_c2_aggr_pad)
        ssh_c2_aggr_bn = self.ssh_c2_aggr_bn(ssh_c2_aggr)
        ssh_c2_aggr_relu = self.ssh_c2_aggr_relu(ssh_c2_aggr_bn)
        #
        ssh_m2_red_up = self.ssh_m2_red_up(ssh_c2_aggr_relu)
        #
        ssh_m2_det_context_conv1_pad = self.ssh_m2_det_context_conv1_pad(ssh_c2_aggr_relu)
        ssh_m2_det_context_conv1 = self.ssh_m2_det_context_conv1(ssh_m2_det_context_conv1_pad)
        ssh_m2_det_context_conv1_bn = self.ssh_m2_det_context_conv1_bn(ssh_m2_det_context_conv1)
        ssh_m2_det_context_conv1_relu = self.ssh_m2_det_context_conv1_relu(ssh_m2_det_context_conv1_bn)
        #
        ssh_m2_det_context_conv3_1_pad = self.ssh_m2_det_context_conv3_1_pad(ssh_m2_det_context_conv1_relu)
        ssh_m2_det_context_conv3_1 = self.ssh_m2_det_context_conv3_1(ssh_m2_det_context_conv3_1_pad)
        ssh_m2_det_context_conv3_1_bn = self.ssh_m2_det_context_conv3_1_bn(ssh_m2_det_context_conv3_1)
        ssh_m2_det_context_conv3_1_relu = self.ssh_m2_det_context_conv3_1_relu(ssh_m2_det_context_conv3_1_bn)
        ssh_m2_det_context_conv3_2_pad = self.ssh_m2_det_context_conv3_2_pad(ssh_m2_det_context_conv3_1_relu)
        ssh_m2_det_context_conv3_2 = self.ssh_m2_det_context_conv3_2(ssh_m2_det_context_conv3_2_pad)
        ssh_m2_det_context_conv3_2_bn = self.ssh_m2_det_context_conv3_2_bn(ssh_m2_det_context_conv3_2)
        #
        ssh_m2_det_context_conv2_pad = self.ssh_m2_det_context_conv2_pad(ssh_m2_det_context_conv1_relu)
        ssh_m2_det_context_conv2 = self.ssh_m2_det_context_conv2(ssh_m2_det_context_conv2_pad)
        ssh_m2_det_context_conv2_bn = self.ssh_m2_det_context_conv2_bn(ssh_m2_det_context_conv2)
        #
        ssh_m2_det_conv1_pad = self.ssh_m2_det_conv1_pad(ssh_c2_aggr_relu)
        ssh_m2_det_conv1 = self.ssh_m2_det_conv1(ssh_m2_det_conv1_pad)
        ssh_m2_det_conv1_bn = self.ssh_m2_det_conv1_bn(ssh_m2_det_conv1)
        # ssh_m1
        ssh_m1_red_conv = self.ssh_m1_red_conv(stage3_unit1_relu2)
        ssh_m1_red_conv_bn = self.ssh_m1_red_conv_bn(ssh_m1_red_conv)
        ssh_m1_red_conv_relu = self.ssh_m1_red_conv_relu(ssh_m1_red_conv_bn)
        crop1 = center_crop(ssh_m2_red_up, ssh_m1_red_conv_relu)
        plus1_v1 = ssh_m1_red_conv_relu + crop1
        #
        ssh_c1_aggr_pad = self.ssh_c1_aggr_pad(plus1_v1)
        ssh_c1_aggr = self.ssh_c1_aggr(ssh_c1_aggr_pad)
        ssh_c1_aggr_bn = self.ssh_c1_aggr_bn(ssh_c1_aggr)
        ssh_c1_aggr_relu = self.ssh_c1_aggr_relu(ssh_c1_aggr_bn)
        #
        ssh_m1_det_context_conv1_pad = self.ssh_m1_det_context_conv1_pad(ssh_c1_aggr_relu)
        ssh_m1_det_context_conv1 = self.ssh_m1_det_context_conv1(ssh_m1_det_context_conv1_pad)
        ssh_m1_det_context_conv1_bn = self.ssh_m1_det_context_conv1_bn(ssh_m1_det_context_conv1)
        ssh_m1_det_context_conv1_relu = self.ssh_m1_det_context_conv1_relu(ssh_m1_det_context_conv1_bn)
        #
        ssh_m1_det_context_conv3_1_pad = self.ssh_m1_det_context_conv3_1_pad(ssh_m1_det_context_conv1_relu)
        ssh_m1_det_context_conv3_1 = self.ssh_m1_det_context_conv3_1(ssh_m1_det_context_conv3_1_pad)
        ssh_m1_det_context_conv3_1_bn = self.ssh_m1_det_context_conv3_1_bn(ssh_m1_det_context_conv3_1)
        ssh_m1_det_context_conv3_1_relu = self.ssh_m1_det_context_conv3_1_relu(ssh_m1_det_context_conv3_1_bn)
        ssh_m1_det_context_conv3_2_pad = self.ssh_m1_det_context_conv3_2_pad(ssh_m1_det_context_conv3_1_relu)
        ssh_m1_det_context_conv3_2 = self.ssh_m1_det_context_conv3_2(ssh_m1_det_context_conv3_2_pad)
        ssh_m1_det_context_conv3_2_bn = self.ssh_m1_det_context_conv3_2_bn(ssh_m1_det_context_conv3_2)
        #
        ssh_m1_det_context_conv2_pad = self.ssh_m1_det_context_conv2_pad(ssh_m1_det_context_conv1_relu)
        ssh_m1_det_context_conv2 = self.ssh_m1_det_context_conv2(ssh_m1_det_context_conv2_pad)
        ssh_m1_det_context_conv2_bn = self.ssh_m1_det_context_conv2_bn(ssh_m1_det_context_conv2)
        #
        ssh_m1_det_conv1_pad = self.ssh_m1_det_conv1_pad(ssh_c1_aggr_relu)
        ssh_m1_det_conv1 = self.ssh_m1_det_conv1(ssh_m1_det_conv1_pad)
        ssh_m1_det_conv1_bn = self.ssh_m1_det_conv1_bn(ssh_m1_det_conv1)
        # stride8
        ssh_m1_det_concat = torch.cat(
            [ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, ssh_m1_det_context_conv3_2_bn],
            dim=1  # channel axis in PyTorch
        )
        ssh_m1_det_concat_relu = self.ssh_m1_det_concat_relu(ssh_m1_det_concat)
        ##
        face_rpn_bbox_pred_stride8 = self.face_rpn_bbox_pred_stride8(ssh_m1_det_concat_relu)
        ##
        face_rpn_cls_score_stride8 = self.face_rpn_cls_score_stride16(ssh_m1_det_concat_relu)
        B, A2, H, W = face_rpn_cls_score_stride8.shape  # A2 = 2 * num_anchors
        face_rpn_cls_prob_stride8 = face_rpn_cls_score_stride8.view(B, 2, A2 // 2, H, W)
        face_rpn_cls_prob_stride8 = self.face_rpn_cls_prob_stride8(face_rpn_cls_prob_stride8)
        face_rpn_cls_prob_stride8 = face_rpn_cls_prob_stride8.view(B, A2, H, W)
        #
        face_rpn_landmark_pred_stride8 = self.face_rpn_landmark_pred_stride8(ssh_m1_det_concat_relu) # [1, 20, 28, 28]
        # stride16
        ssh_m2_det_concat = torch.cat(
            [ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, ssh_m2_det_context_conv3_2_bn],
            dim=1  # channel axis in PyTorch
        )
        ssh_m2_det_concat_relu = self.ssh_m2_det_concat_relu(ssh_m2_det_concat) # [1, 512, 14, 14]
        ## bbox
        face_rpn_bbox_pred_stride16 = self.face_rpn_bbox_pred_stride32(ssh_m2_det_concat_relu)
        ## prob
        face_rpn_cls_score_stride16 = self.face_rpn_cls_score_stride16(ssh_m2_det_concat_relu)
        B, A2, H, W = face_rpn_cls_score_stride16.shape  # A2 = 2 * num_anchors
        face_rpn_cls_prob_stride16 = face_rpn_cls_score_stride16.view(B, 2, A2 // 2, H, W)
        face_rpn_cls_prob_stride16 = self.face_rpn_cls_prob_stride16(face_rpn_cls_prob_stride16)
        face_rpn_cls_prob_stride16 = face_rpn_cls_prob_stride16.view(B, A2, H, W)
        ## landmark
        face_rpn_landmark_pred_stride16 = self.face_rpn_landmark_pred_stride16(ssh_m2_det_concat_relu)
        # stride32
        ssh_m3_det_concat = torch.cat(
            [ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn],
            dim=1  # channel axis in PyTorch
        )
        ##
        ssh_m3_det_concat_relu = self.ssh_m3_det_concat_relu(ssh_m3_det_concat)
        ## bbox
        face_rpn_bbox_pred_stride32 = self.face_rpn_bbox_pred_stride32(ssh_m3_det_concat_relu)
        ## prob
        face_rpn_cls_score_stride32 = self.face_rpn_cls_score_stride32(ssh_m3_det_concat_relu)
        B, A2, H, W = face_rpn_cls_score_stride32.shape  # A2 = 2 * num_anchors
        face_rpn_cls_prob_stride32 = face_rpn_cls_score_stride32.view(B, 2, A2 // 2, H, W)
        face_rpn_cls_prob_stride32 = self.face_rpn_cls_prob_stride32(face_rpn_cls_prob_stride32)
        face_rpn_cls_prob_stride32 = face_rpn_cls_prob_stride32.view(B, A2, H, W)
        ## landmark
        face_rpn_landmark_pred_stride32 = self.face_rpn_landmark_pred_stride32(ssh_m3_det_concat_relu)
        return (
            face_rpn_bbox_pred_stride8,
            face_rpn_cls_score_stride8,
            face_rpn_landmark_pred_stride8,
            #
            face_rpn_bbox_pred_stride16,
            face_rpn_cls_score_stride16,
            face_rpn_landmark_pred_stride16,
            #
            face_rpn_bbox_pred_stride32,
            face_rpn_cls_score_stride32,
            face_rpn_landmark_pred_stride32,
        )



def load_bn(torch_bn: nn.BatchNorm2d, keras_group: h5py.Group) -> None:
    with torch.no_grad():
        torch_bn.weight.copy_(torch.tensor(keras_group["gamma:0"][:])) # type: ignore
        torch_bn.bias.copy_(torch.tensor(keras_group["beta:0"][:])) # type: ignore
        torch_bn.running_mean.copy_(torch.tensor(keras_group["moving_mean:0"][:])) # type: ignore
        torch_bn.running_var.copy_(torch.tensor(keras_group["moving_variance:0"][:])) # type: ignore


def load_conv2d(torch_conv: nn.Conv2d, keras_group: h5py.Group, bias: bool=False) -> None:
    with torch.no_grad():
        keras_weights = keras_group["kernel:0"][:]  # type: ignore
        keras_weights = np.transpose(keras_weights, (3, 2, 0, 1)) # type: ignore
        torch_conv.weight.copy_(torch.tensor(keras_weights))
        if bias:
            torch_conv.bias.copy_(torch.tensor(keras_group["bias:0"][:])) # type: ignore


def load_weights(h5_path: str, model: RetinafaceModel):
    f = h5py.File(h5_path)
    model = RetinafaceModel()
    # stage0
    load_bn(model.bn0, f["bn0/bn0"]) # type: ignore
    load_bn(model.bn_data, f["bn_data/bn_data"]) # type: ignore
    load_conv2d(model.conv0, f["conv0/conv0"]) # type: ignore
    # stage1
    # stage1_unit1
    load_bn(model.stage1_unit1_bn1, f["stage1_unit1_bn1/stage1_unit1_bn1"]) # type: ignore
    load_conv2d(model.stage1_unit1_conv1, f["stage1_unit1_conv1/stage1_unit1_conv1"]) # type: ignore
    load_bn(model.stage1_unit1_bn2, f["stage1_unit1_bn2/stage1_unit1_bn2"]) # type: ignore
    load_conv2d(model.stage1_unit1_conv2, f["stage1_unit1_conv2/stage1_unit1_conv2"]) # type: ignore
    load_bn(model.stage1_unit1_bn3, f["stage1_unit1_bn3/stage1_unit1_bn3"]) # type: ignore
    load_conv2d(model.stage1_unit1_conv3, f["stage1_unit1_conv3/stage1_unit1_conv3"]) # type: ignore
    ## stage1_unit2
    load_bn(model.stage1_unit2_bn1, f["stage1_unit2_bn1/stage1_unit2_bn1"]) # type: ignore
    load_conv2d(model.stage1_unit2_conv1, f["stage1_unit2_conv1/stage1_unit2_conv1"]) # type: ignore
    load_bn(model.stage1_unit2_bn2, f["stage1_unit2_bn2/stage1_unit2_bn2"]) # type: ignore
    load_conv2d(model.stage1_unit2_conv2, f["stage1_unit2_conv2/stage1_unit2_conv2"]) # type: ignore
    load_bn(model.stage1_unit2_bn3, f["stage1_unit2_bn3/stage1_unit2_bn3"]) # type: ignore
    load_conv2d(model.stage1_unit2_conv3, f["stage1_unit2_conv3/stage1_unit2_conv3"]) # type: ignore
    ## stage1_unit3
    load_bn(model.stage1_unit3_bn1, f["stage1_unit3_bn1/stage1_unit3_bn1"]) # type: ignore
    load_conv2d(model.stage1_unit3_conv1, f["stage1_unit3_conv1/stage1_unit3_conv1"]) # type: ignore
    load_bn(model.stage1_unit3_bn2, f["stage1_unit3_bn2/stage1_unit3_bn2"]) # type: ignore
    load_conv2d(model.stage1_unit3_conv2, f["stage1_unit3_conv2/stage1_unit3_conv2"]) # type: ignore
    load_bn(model.stage1_unit3_bn3, f["stage1_unit3_bn3/stage1_unit3_bn3"]) # type: ignore
    load_conv2d(model.stage1_unit3_conv3, f["stage1_unit3_conv3/stage1_unit3_conv3"]) # type: ignore
    # stage2
    ## stage2_unit1
    load_bn(model.stage2_unit1_bn1, f["stage2_unit1_bn1/stage2_unit1_bn1"]) # type: ignore
    load_conv2d(model.stage2_unit1_conv1, f["stage2_unit1_conv1/stage2_unit1_conv1"]) # type: ignore
    load_bn(model.stage2_unit1_bn2, f["stage2_unit1_bn2/stage2_unit1_bn2"]) # type: ignore
    load_conv2d(model.stage2_unit1_conv2, f["stage2_unit1_conv2/stage2_unit1_conv2"]) # type: ignore
    load_bn(model.stage2_unit1_bn3, f["stage2_unit1_bn3/stage2_unit1_bn3"]) # type: ignore
    load_conv2d(model.stage2_unit1_conv3, f["stage2_unit1_conv3/stage2_unit1_conv3"]) # type: ignore
    load_conv2d(model.stage2_unit1_sc, f["stage2_unit1_sc/stage2_unit1_sc"]) # type: ignore
    ## stage2_unit2
    load_bn(model.stage2_unit2_bn1, f["stage2_unit2_bn1/stage2_unit2_bn1"]) # type: ignore
    load_conv2d(model.stage2_unit2_conv1, f["stage2_unit2_conv1/stage2_unit2_conv1"]) # type: ignore
    load_bn(model.stage2_unit2_bn2, f["stage2_unit2_bn2/stage2_unit2_bn2"]) # type: ignore
    load_conv2d(model.stage2_unit2_conv2, f["stage2_unit2_conv2/stage2_unit2_conv2"]) # type: ignore
    load_bn(model.stage2_unit2_bn3, f["stage2_unit2_bn3/stage2_unit2_bn3"]) # type: ignore
    load_conv2d(model.stage2_unit2_conv3, f["stage2_unit2_conv3/stage2_unit2_conv3"]) # type: ignore
    ## stage2_unit3
    load_bn(model.stage2_unit3_bn1, f["stage2_unit3_bn1/stage2_unit3_bn1"]) # type: ignore
    load_conv2d(model.stage2_unit3_conv1, f["stage2_unit3_conv1/stage2_unit3_conv1"]) # type: ignore
    load_bn(model.stage2_unit3_bn2, f["stage2_unit3_bn2/stage2_unit3_bn2"]) # type: ignore
    load_conv2d(model.stage2_unit3_conv2, f["stage2_unit3_conv2/stage2_unit3_conv2"]) # type: ignore
    load_bn(model.stage2_unit3_bn3, f["stage2_unit3_bn3/stage2_unit3_bn3"]) # type: ignore
    load_conv2d(model.stage2_unit3_conv3, f["stage2_unit3_conv3/stage2_unit3_conv3"]) # type: ignore
    ## stage2_unit4
    load_bn(model.stage2_unit4_bn1, f["stage2_unit4_bn1/stage2_unit4_bn1"]) # type: ignore
    load_conv2d(model.stage2_unit4_conv1, f["stage2_unit4_conv1/stage2_unit4_conv1"]) # type: ignore
    load_bn(model.stage2_unit4_bn2, f["stage2_unit4_bn2/stage2_unit4_bn2"]) # type: ignore
    load_conv2d(model.stage2_unit4_conv2, f["stage2_unit4_conv2/stage2_unit4_conv2"]) # type: ignore
    load_bn(model.stage2_unit4_bn3, f["stage2_unit4_bn3/stage2_unit4_bn3"]) # type: ignore
    load_conv2d(model.stage2_unit4_conv3, f["stage2_unit4_conv3/stage2_unit4_conv3"]) # type: ignore
    # stage3
    ## stage3_unit1
    load_bn(model.stage3_unit1_bn1, f["stage3_unit1_bn1/stage3_unit1_bn1"]) # type: ignore
    load_conv2d(model.stage3_unit1_conv1, f["stage3_unit1_conv1/stage3_unit1_conv1"]) # type: ignore
    load_bn(model.stage3_unit1_bn2, f["stage3_unit1_bn2/stage3_unit1_bn2"]) # type: ignore
    load_conv2d(model.stage3_unit1_conv2, f["stage3_unit1_conv2/stage3_unit1_conv2"]) # type: ignore
    load_bn(model.stage3_unit1_bn3, f["stage3_unit1_bn3/stage3_unit1_bn3"]) # type: ignore
    load_conv2d(model.stage3_unit1_conv3, f["stage3_unit1_conv3/stage3_unit1_conv3"]) # type: ignore
    load_conv2d(model.stage3_unit1_sc, f["stage3_unit1_sc/stage3_unit1_sc"]) # type: ignore
    ## stage3_unit2
    load_bn(model.stage3_unit2_bn1, f["stage3_unit2_bn1/stage3_unit2_bn1"]) # type: ignore
    load_conv2d(model.stage3_unit2_conv1, f["stage3_unit2_conv1/stage3_unit2_conv1"]) # type: ignore
    load_bn(model.stage3_unit2_bn2, f["stage3_unit2_bn2/stage3_unit2_bn2"]) # type: ignore
    load_conv2d(model.stage3_unit2_conv2, f["stage3_unit2_conv2/stage3_unit2_conv2"]) # type: ignore
    load_bn(model.stage3_unit2_bn3, f["stage3_unit2_bn3/stage3_unit2_bn3"]) # type: ignore
    load_conv2d(model.stage3_unit2_conv3, f["stage3_unit2_conv3/stage3_unit2_conv3"]) # type: ignore
    ## stage3_unit3
    load_bn(model.stage3_unit3_bn1, f["stage3_unit3_bn1/stage3_unit3_bn1"]) # type: ignore
    load_conv2d(model.stage3_unit3_conv1, f["stage3_unit3_conv1/stage3_unit3_conv1"]) # type: ignore
    load_bn(model.stage3_unit3_bn2, f["stage3_unit3_bn2/stage3_unit3_bn2"]) # type: ignore
    load_conv2d(model.stage3_unit3_conv2, f["stage3_unit3_conv2/stage3_unit3_conv2"]) # type: ignore
    load_bn(model.stage3_unit3_bn3, f["stage3_unit3_bn3/stage3_unit3_bn3"]) # type: ignore
    load_conv2d(model.stage3_unit3_conv3, f["stage3_unit3_conv3/stage3_unit3_conv3"]) # type: ignore
    ## stage3_unit4
    load_bn(model.stage3_unit4_bn1, f["stage3_unit4_bn1/stage3_unit4_bn1"]) # type: ignore
    load_conv2d(model.stage3_unit4_conv1, f["stage3_unit4_conv1/stage3_unit4_conv1"]) # type: ignore
    load_bn(model.stage3_unit4_bn2, f["stage3_unit4_bn2/stage3_unit4_bn2"]) # type: ignore
    load_conv2d(model.stage3_unit4_conv2, f["stage3_unit4_conv2/stage3_unit4_conv2"]) # type: ignore
    load_bn(model.stage3_unit4_bn3, f["stage3_unit4_bn3/stage3_unit4_bn3"]) # type: ignore
    load_conv2d(model.stage3_unit4_conv3, f["stage3_unit4_conv3/stage3_unit4_conv3"]) # type: ignore
    ## stage3_unit5
    load_bn(model.stage3_unit5_bn1, f["stage3_unit5_bn1/stage3_unit5_bn1"]) # type: ignore
    load_conv2d(model.stage3_unit5_conv1, f["stage3_unit5_conv1/stage3_unit5_conv1"]) # type: ignore
    load_bn(model.stage3_unit5_bn2, f["stage3_unit5_bn2/stage3_unit5_bn2"]) # type: ignore
    load_conv2d(model.stage3_unit5_conv2, f["stage3_unit5_conv2/stage3_unit5_conv2"]) # type: ignore
    load_bn(model.stage3_unit5_bn3, f["stage3_unit5_bn3/stage3_unit5_bn3"]) # type: ignore
    load_conv2d(model.stage3_unit5_conv3, f["stage3_unit5_conv3/stage3_unit5_conv3"]) # type: ignore
    ## stage3_unit6
    load_bn(model.stage3_unit6_bn1, f["stage3_unit6_bn1/stage3_unit6_bn1"]) # type: ignore
    load_conv2d(model.stage3_unit6_conv1, f["stage3_unit6_conv1/stage3_unit6_conv1"]) # type: ignore
    load_bn(model.stage3_unit6_bn2, f["stage3_unit6_bn2/stage3_unit6_bn2"]) # type: ignore
    load_conv2d(model.stage3_unit6_conv2, f["stage3_unit6_conv2/stage3_unit6_conv2"]) # type: ignore
    load_bn(model.stage3_unit6_bn3, f["stage3_unit6_bn3/stage3_unit6_bn3"]) # type: ignore
    load_conv2d(model.stage3_unit6_conv3, f["stage3_unit6_conv3/stage3_unit6_conv3"]) # type: ignore
    # stage4
    ## stage4_unit1
    load_bn(model.stage4_unit1_bn1, f["stage4_unit1_bn1/stage4_unit1_bn1"]) # type: ignore
    load_conv2d(model.stage4_unit1_conv1, f["stage4_unit1_conv1/stage4_unit1_conv1"]) # type: ignore
    load_bn(model.stage4_unit1_bn2, f["stage4_unit1_bn2/stage4_unit1_bn2"]) # type: ignore
    load_conv2d(model.stage4_unit1_conv2, f["stage4_unit1_conv2/stage4_unit1_conv2"]) # type: ignore
    load_bn(model.stage4_unit1_bn3, f["stage4_unit1_bn3/stage4_unit1_bn3"]) # type: ignore
    load_conv2d(model.stage4_unit1_conv3, f["stage4_unit1_conv3/stage4_unit1_conv3"]) # type: ignore
    load_conv2d(model.stage4_unit1_sc, f["stage4_unit1_sc/stage4_unit1_sc"]) # type: ignore
    ## stage4_unit2
    load_bn(model.stage4_unit2_bn1, f["stage4_unit2_bn1/stage4_unit2_bn1"]) # type: ignore
    load_conv2d(model.stage4_unit2_conv1, f["stage4_unit2_conv1/stage4_unit2_conv1"]) # type: ignore
    load_bn(model.stage4_unit2_bn2, f["stage4_unit2_bn2/stage4_unit2_bn2"]) # type: ignore
    load_conv2d(model.stage4_unit2_conv2, f["stage4_unit2_conv2/stage4_unit2_conv2"]) # type: ignore
    load_bn(model.stage4_unit2_bn3, f["stage4_unit2_bn3/stage4_unit2_bn3"]) # type: ignore
    load_conv2d(model.stage4_unit2_conv3, f["stage4_unit2_conv3/stage4_unit2_conv3"]) # type: ignore
    ## stage4_unit3
    load_bn(model.stage4_unit3_bn1, f["stage4_unit3_bn1/stage4_unit3_bn1"]) # type: ignore
    load_conv2d(model.stage4_unit3_conv1, f["stage4_unit3_conv1/stage4_unit3_conv1"]) # type: ignore
    load_bn(model.stage4_unit3_bn2, f["stage4_unit3_bn2/stage4_unit3_bn2"]) # type: ignore
    load_conv2d(model.stage4_unit3_conv2, f["stage4_unit3_conv2/stage4_unit3_conv2"]) # type: ignore
    load_bn(model.stage4_unit3_bn3, f["stage4_unit3_bn3/stage4_unit3_bn3"]) # type: ignore
    load_conv2d(model.stage4_unit3_conv3, f["stage4_unit3_conv3/stage4_unit3_conv3"]) # type: ignore