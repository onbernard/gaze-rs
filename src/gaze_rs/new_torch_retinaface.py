# import torch
import torch.nn as nn


BN_EPS = 1.9999999494757503e-05


class RetinafaceModel(nn.Module):
    def __init__(self):
        super(RetinafaceModel, self).__init__()
        # stage 0
        self.bn_data = nn.BatchNorm2d(3, eps=BN_EPS, affine=False)
        self.conv0_pad = nn.ZeroPad2d(3)
        self.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding="valid",
            bias=False
        )
        self.bn0 = nn.BatchNorm2d(64, eps=BN_EPS, affine=False)
        self.relu0 = nn.ReLU()
        self.pooling0_pad = nn.ZeroPad2d(1)
        self.pooling0 =  nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0) # [1, 64, 56, 56]
        # stage 1
        ## unit 1
        self.stage1_unit1_bn1 = nn.BatchNorm2d(64, eps=BN_EPS, affine=False)
        self.stage1_unit1_relu1 = nn.ReLU()
        self.stage1_unit1_conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage1_unit1_bn2 = nn.BatchNorm2d(64, eps=BN_EPS, affine=False)
        self.stage1_unit1_relu2 = nn.ReLU()
        self.stage1_unit1_conv2_pad = nn.ZeroPad2d(1)
        self.stage1_unit1_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage1_unit1_bn3 = nn.BatchNorm2d(64, eps=BN_EPS, affine=False)
        self.stage1_unit1_relu3 = nn.ReLU()
        self.stage1_unit1_conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage1_unit1_sc = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        # plus0_v1 : [1, 256, 56, 56]
        ## unit 1 stage 2
        self.stage1_unit2_bn1 = nn.BatchNorm2d(256, eps=BN_EPS, affine=False)
        self.stage1_unit2_relu1 = nn.ReLU()
        self.stage1_unit2_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage1_unit2_bn2 = nn.BatchNorm2d(64, eps=BN_EPS, affine=False)
        self.stage1_unit2_relu2 = nn.ReLU()
        self.stage1_unit2_conv2_pad = nn.ZeroPad2d(1)
        self.stage1_unit2_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage1_unit2_bn3 = nn.BatchNorm2d(64, eps=BN_EPS, affine=False)
        self.stage1_unit2_relu3 = nn.ReLU()
        self.stage1_unit2_conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        # plus1_v2: 1, 256, 56, 56]
        ## stage1_unit3
        self.stage1_unit3_bn1 = nn.BatchNorm2d(256, eps=BN_EPS, affine=False)
        self.stage1_unit3_relu1 = nn.ReLU()
        self.stage1_unit3_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage1_unit3_bn2 = nn.BatchNorm2d(64, eps=BN_EPS, affine=False)
        self.stage1_unit3_relu2 = nn.ReLU()
        self.stage1_unit3_conv2_pad = nn.ZeroPad2d(1)
        self.stage1_unit3_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage1_unit3_bn3 = nn.BatchNorm2d(64, eps=BN_EPS, affine=False)
        self.stage1_unit3_relu3 = nn.ReLU()
        self.stage1_unit3_conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        # plus2: [1, 256, 56, 56]
        # stage2
        ## stage2_unit1
        self.stage2_unit1_bn1 = nn.BatchNorm2d(256, eps=BN_EPS, affine=False)
        self.stage2_unit1_relu1 = nn.ReLU()
        self.stage2_unit1_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage2_unit1_bn2 = nn.BatchNorm2d(128, eps=BN_EPS, affine=False)
        self.stage2_unit1_relu2 = nn.ReLU()
        self.stage2_unit1_conv2_pad = nn.ZeroPad2d(1)
        self.stage2_unit1_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(2,2),
            padding="valid",
            bias=False,
        )
        self.stage2_unit1_bn3 = nn.BatchNorm2d(128, eps=BN_EPS, affine=False)
        self.stage2_unit1_relu3 = nn.ReLU()
        self.stage2_unit1_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage2_unit1_sc = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(1,1),
            stride=(2,2),
            padding="valid",
            bias=False,
        )
        # plus3: [1, 512, 28, 28]
        ...


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
        return plus3