import torch
import torch.nn as nn


class RetinafaceModel(nn.Module):
    def __init__(self):
        super(RetinafaceModel, self).__init__()
        self.bn_data = nn.BatchNorm2d(3, eps=1.9999999494757503e-05, affine=False)
        self.conv0_pad = nn.ConstantPad2d(3, 0)
        self.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding="valid",
            bias=False
        )
        self.bn0 = nn.BatchNorm2d(64, eps=1.9999999494757503e-05, affine=False)
        self.relu0 = nn.ReLU()
        self.pooling0_pad = nn.ConstantPad2d(1, 0)
        self.pooling0 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0)
        # Stage 1
        ## Unit 1
        self.stage1_unit1_bn1 = nn.BatchNorm2d(64, 1.9999999494757503e-05, affine=False)
        self.stage1_unit1_relu1 = nn.ReLU()
        self.stage1_unit1_conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
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
        self.stage1_unit1_bn2 = nn.BatchNorm2d(64, eps=1.9999999494757503e-05, affine=False)
        self.stage1_unit1_relu2 = nn.ReLU()
        self.stage1_unit1_conv2_pad = nn.ZeroPad2d(1)
        self.stage1_unit1_conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False
        )
        self.stage1_unit1_bn3 = nn.BatchNorm2d(64, 1.9999999494757503e-05, affine=False)
        self.stage1_unit1_relu3 = nn.ReLU()
        self.stage1_unit1_conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 2
        self.stage1_unit2_bn1 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage1_unit2_relu1 = nn.ReLU()
        self.stage1_unit2_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage1_unit2_bn2 = nn.BatchNorm2d(64, eps=1.9999999494757503e-05, affine=False)
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
        self.stage1_unit2_bn3 = nn.BatchNorm2d(64, eps=1.9999999494757503e-05, affine=False)
        self.stage1_unit2_relu3 = nn.ReLU()
        self.stage1_unit2_conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 3
        self.stage1_unit3_bn1 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage1_unit3_relu1 = nn.ReLU()
        self.stage1_unit3_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage1_unit3_bn2 = nn.BatchNorm2d(64, eps=1.9999999494757503e-05, affine=False)
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
        self.stage1_unit3_bn3 = nn.BatchNorm2d(64, eps=1.9999999494757503e-05, affine=False)
        self.stage1_unit3_relu3 = nn.ReLU()
        self.stage1_unit3_conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        # Stage 2
        self.stage2_unit1_bn1 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit1_relu1 = nn.ReLU()
        self.stage2_unit1_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
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
        self.stage2_unit1_bn2 = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
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
        self.stage2_unit1_bn3 = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit1_relu3 = nn.ReLU()
        self.stage2_unit1_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 2
        self.stage2_unit2_bn1 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit2_relu1 = nn.ReLU()
        self.stage2_unit2_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage2_unit2_bn2 = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit2_relu2 = nn.ReLU()
        self.stage2_unit2_conv2_pad = nn.ZeroPad2d(1)
        self.stage2_unit2_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage2_unit2_bn3 = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit2_relu3 = nn.ReLU()
        self.stage2_unit2_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 3
        self.stage2_unit3_bn1 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit3_relu1 = nn.ReLU()
        self.stage2_unit3_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage2_unit3_bn2 = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit3_relu2 = nn.ReLU()
        self.stage2_unit3_conv2_pad = nn.ZeroPad2d(1)
        self.stage2_unit3_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage2_unit3_bn3 = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit3_relu3 = nn.ReLU()
        self.stage2_unit3_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        # Unit 4
        self.stage2_unit4_bn1 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit4_relu1 = nn.ReLU()
        self.stage2_unit4_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=128,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage2_unit4_bn2 = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit4_relu2 = nn.ReLU()
        self.stage2_unit4_conv2_pad = nn.ZeroPad2d(1)
        self.stage2_unit4_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage2_unit4_bn3 = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.stage2_unit4_relu3 = nn.ReLU()
        self.stage2_unit4_conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        # Stage 3
        ## Unit 1
        self.stage3_unit1_bn1 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit1_relu1 = nn.ReLU()
        self.stage3_unit1_conv1 = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit1_sc = nn.Conv2d(
            in_channels=512,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(2,2),
            padding="valid",
            bias=False,
        )
        self.stage3_unit1_bn2 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit1_relu2 = nn.ReLU()
        self.stage3_unit1_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit1_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(2,2),
            padding="valid",
            bias=False,
        )
        self.ssh_m1_red_conv = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.stage3_unit1_bn3 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m1_red_conv_bn = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit1_relu3 = nn.ReLU()
        self.ssh_m1_red_conv_relu = nn.ReLU()
        self.stage3_unit1_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 2
        self.stage3_unit2_bn1 = nn.BatchNorm2d(1024, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit2_relu1 = nn.ReLU()
        self.stage3_unit2_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit2_bn2 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit2_relu2 = nn.ReLU()
        self.stage3_unit2_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit2_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit2_bn3 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit2_relu3 = nn.ReLU()
        self.stage3_unit2_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 3
        self.stage3_unit3_bn1 = nn.BatchNorm2d(1024, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit3_relu1 = nn.ReLU()
        self.stage3_unit3_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit3_bn2 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit3_relu2 = nn.ReLU()
        self.stage3_unit3_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit3_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit3_bn3 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit3_relu3 = nn.ReLU()
        self.stage3_unit3_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 4
        self.stage3_unit4_bn1 = nn.BatchNorm2d(1024, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit4_relu1 = nn.ReLU()
        self.stage3_unit4_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit4_bn2 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit4_relu2 = nn.ReLU()
        self.stage3_unit4_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit4_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit4_bn3 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit4_relu3 = nn.ReLU()
        self.stage3_unit4_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        # Unit 5
        self.stage3_unit5_bn1 = nn.BatchNorm2d(1024, eps=1.9999999494757503e-05, affine=False,)
        self.stage3_unit5_relu1 = nn.ReLU()
        self.stage3_unit5_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit5_bn2 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit5_relu2 = nn.ReLU()
        self.stage3_unit5_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit5_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit5_bn3 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit5_relu3 = nn.ReLU()
        self.stage3_unit5_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 6
        self.stage3_unit6_bn1 = nn.BatchNorm2d(1024, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit6_relu1 = nn.ReLU()
        self.stage3_unit6_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False
        )
        self.stage3_unit6_bn2 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit6_relu2 = nn.ReLU()
        self.stage3_unit6_conv2_pad = nn.ZeroPad2d(1)
        self.stage3_unit6_conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage3_unit6_bn3 = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage3_unit6_relu3 = nn.ReLU()
        self.stage3_unit6_conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=1024,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        # Stage 4
        ## Unit 1
        self.stage4_unit1_bn1 = nn.BatchNorm2d(1024, eps=1.9999999494757503e-05, affine=False)
        self.stage4_unit1_relu1 = nn.ReLU()
        self.stage4_unit1_conv1 = nn.Conv2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage4_unit1_sc = nn.Conv2d(
            in_channels=1024,
            out_channels=2048,
            kernel_size=(1,1),
            stride=(2,2),
            padding="valid",
            bias=False,
        )
        self.stage4_unit1_bn2 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.stage4_unit1_relu2 = nn.ReLU()
        self.stage4_unit1_conv2_pad = nn.ZeroPad2d(1)
        self.stage4_unit1_conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3,3),
            stride=(2,2),
            padding="valid",
            bias=False,
        )
        self.ssh_c2_lateral = nn.Conv2d(
            in_channels=512,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True
        )
        self.stage4_unit1_bn3 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.ssh_c2_lateral_bn = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.stage4_unit1_relu3 = nn.ReLU()
        self.ssh_c2_lateral_relu = nn.ReLU()
        self.stage4_unit1_conv3 = nn.Conv2d(
            in_channels=512,
            out_channels=2048,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 2
        self.stage4_unit2_bn1 = nn.BatchNorm2d(2048, eps=1.9999999494757503e-05, affine=False)
        self.stage4_unit2_relu1 = nn.ReLU()
        self.stage4_unit2_conv1 = nn.Conv2d(
            in_channels=2048,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage4_unit2_bn2 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.stage4_unit2_relu2 = nn.ReLU()
        self.stage4_unit2_conv2_pad = nn.ZeroPad2d(1)
        self.stage4_unit2_conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage4_unit2_bn3 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.stage4_unit2_relu3 = nn.ReLU()
        self.stage4_unit2_conv3 = nn.Conv2d(
            in_channels=512,
            out_channels=2048,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        ## Unit 4
        self.stage4_unit3_bn1 = nn.BatchNorm2d(2048, eps=1.9999999494757503e-05, affine=False)
        self.stage4_unit3_relu1 = nn.ReLU()
        self.stage4_unit3_conv1 = nn.Conv2d(
            in_channels=2048,
            out_channels=512,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage4_unit3_bn2 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.stage4_unit3_relu2 = nn.ReLU()
        self.stage4_unit3_conv2_pad = nn.ZeroPad2d(1)
        self.stage4_unit3_conv2 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        self.stage4_unit3_bn3 = nn.BatchNorm2d(512, eps=1.9999999494757503e-05, affine=False)
        self.stage4_unit3_relu3 = nn.ReLU()
        self.stage4_unit3_conv3 = nn.Conv2d(
            in_channels=512,
            out_channels=2048,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=False,
        )
        # Heh
        self.bn1 = nn.BatchNorm2d(2048, eps=1.9999999494757503e-05, affine=False)
        self.relu1 = nn.ReLU()
        self.ssh_c3_lateral = nn.Conv2d(
            in_channels=2048,
            out_channels=256,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_c3_lateral_bn = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.ssh_c3_lateral_relu = nn.ReLU()
        self.ssh_m3_det_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m3_det_context_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_context_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_c3_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.ssh_m3_det_conv1_bn = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m3_det_context_conv1_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m3_det_context_conv1_relu = nn.ReLU()
        # Heh 2
        self.ssh_m3_det_context_conv2_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_context_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m3_det_context_conv3_1_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_context_conv3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_c2_aggr_pad = nn.ZeroPad2d(1)
        self.ssh_c2_aggr = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m3_det_context_conv2_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m3_det_context_conv3_1_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_c2_aggr_bn = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m3_det_context_conv3_1_relu = nn.ReLU()
        self.ssh_c2_aggr_relu = nn.ReLU()
        self.ssh_m3_det_context_conv3_2_pad = nn.ZeroPad2d(1)
        self.ssh_m3_det_context_conv3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m2_det_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m2_det_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m2_det_context_conv1_pad = nn.ZeroPad1d(1)
        self.ssh_m2_det_context_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m2_red_up = nn.Upsample(scale_factor=2, mode="nearest")
        self.ssh_m3_det_context_conv3_2_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m2_det_conv1_bn = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m2_det_context_conv1_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m2_det_context_conv1_relu = nn.ReLU()
        # Heh 3
        self.ssh_m3_det_concat_relu = nn.ReLU()
        self.ssh_m2_det_context_conv2_pad = nn.ZeroPad2d(1)
        self.ssh_m2_det_context_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m2_det_context_conv3_1_pad = nn.ZeroPad2d(1)
        self.ssh_m2_det_context_conv3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            bias=True,
        )
        self.ssh_c1_aggr_pad = nn.ZeroPad2d(1)
        self.ssh_c1_aggr = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.face_rpn_cls_score_stride32 = nn.Conv2d(
            in_channels=512,
            out_channels=4,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.face_rpn_bbox_pred_stride32 = nn.Conv2d(
            in_channels=512,
            out_channels=8,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.face_rpn_landmark_pred_stride32 = nn.Conv2d(
            in_channels=512,
            out_channels=20,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m2_det_context_conv2_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m2_det_context_conv3_1_bn = nn.BatchNorm2d(128, 1.9999999494757503e-05, affine=False)
        self.ssh_c1_aggr_bn = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m2_det_context_conv3_1_relu = nn.ReLU()
        self.ssh_c1_aggr_relu = nn.ReLU()
        self.face_rpn_cls_prob_stride32 = nn.Softmax(dim=-1)
        self.ssh_m2_det_context_conv3_2_pad = nn.ZeroPad2d(1)
        self.ssh_m2_det_context_conv3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m1_det_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m1_det_context_conv1_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_context_conv1 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m2_det_context_conv3_2_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m1_det_conv1_bn = nn.BatchNorm2d(256, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m1_det_context_conv1_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m1_det_context_conv1_relu = nn.ReLU()
        self.ssh_m2_det_concat_relu = nn.ReLU()
        self.ssh_m1_det_context_conv2_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_context_conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m1_det_context_conv3_1_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_context_conv3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.face_rpn_cls_score_stride16 = nn.Conv2d(
            in_channels=512,
            out_channels=4,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.face_rpn_bbox_pred_stride16 = nn.Conv2d(
            in_channels=512,
            out_channels=8,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.face_rpn_landmark_pred_stride16 = nn.Conv2d(
            in_channels=512,
            out_channels=20,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m1_det_context_conv2_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m1_det_context_conv3_1_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m1_det_context_conv3_1_relu = nn.ReLU()
        self.face_rpn_cls_prob_stride16 = nn.Softmax(dim=-1)
        self.ssh_m1_det_context_conv3_2_pad = nn.ZeroPad2d(1)
        self.ssh_m1_det_context_conv3_2 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3,3),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.ssh_m1_det_context_conv3_2_bn = nn.BatchNorm2d(128, eps=1.9999999494757503e-05, affine=False)
        self.ssh_m1_det_concat_relu = nn.ReLU()
        self.face_rpn_cls_score_stride8 = nn.Conv2d(
            in_channels=512,
            out_channels=4,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.face_rpn_bbox_pred_stride8 = nn.Conv2d(
            in_channels=512,
            out_channels=8,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.face_rpn_landmark_pred_stride8 = nn.Conv2d(
            in_channels=512,
            out_channels=20,
            kernel_size=(1,1),
            stride=(1,1),
            padding="valid",
            bias=True,
        )
        self.face_rpn_cls_prob_stride8 = nn.Softmax(dim=-1)


    def forward(self, x):
        bn_data = self.bn_data(x)
        conv0_pad = self.conv0_pad(bn_data)
        conv0 = self.conv0(conv0_pad)
        bn0 = self.bn0(conv0)
        relu0 = self.relu0(bn0)
        pooling0_pad = self.pooling0_pad(relu0)
        pooling0 = self.pooling0(pooling0_pad)
        # Stage 1
        ## Unit 1
        s1_u1_bn1 = self.stage1_unit1_bn1(pooling0)
        s1_u1_relu1 = self.stage1_unit1_relu1(s1_u1_bn1)
        s1_u1_conv1 = self.stage1_unit1_conv1(s1_u1_relu1)
        s1_u1_sc = self.stage1_unit1_sc(s1_u1_relu1)
        s1_u1_bn2 = self.stage1_unit1_bn2(s1_u1_conv1)
        s1_u1_relu2 = self.stage1_unit1_relu2(s1_u1_bn2)
        s1_u1_conv2_pad = self.stage1_unit1_conv2_pad(s1_u1_relu2)
        s1_u1_conv2 = self.stage1_unit1_conv2(s1_u1_conv2_pad)
        s1_u1_bn3 = self.stage1_unit1_bn3(s1_u1_conv2)
        s1_u1_relu3 = self.stage1_unit1_relu3(s1_u1_bn3)
        s1_u1_conv3 = self.stage1_unit1_conv3(s1_u1_relu3)
        plus0_v1 = s1_u1_sc + s1_u1_conv3
        ## Unit 2
        s1_u2_bn1 = self.stage1_unit2_bn1(plus0_v1)
        s1_u2_relu1 = self.stage1_unit2_relu1(s1_u2_bn1)
        s1_u2_conv1 = self.stage1_unit2_conv1(s1_u2_relu1)
        s1_u2_bn2 = self.stage1_unit2_bn2(s1_u2_conv1)
        s1_u2_relu2 = self.stage1_unit2_relu2(s1_u2_bn2)
        s1_u2_conv2_pad = self.stage1_unit2_conv2_pad(s1_u2_relu2)
        s1_u2_conv2 = self.stage1_unit2_conv2(s1_u2_conv2_pad)
        s1_u2_bn3 = self.stage1_unit2_bn3(s1_u2_conv2)
        s1_u2_relu3 = self.stage1_unit2_relu3(s1_u2_bn3)
        s1_u2_conv3 = self.stage1_unit2_conv3(s1_u2_relu3)
        plus1_v2 = s1_u2_conv3 + plus0_v1
        ## Unit 3
        s1_u3_bn1 = self.stage1_unit3_bn1(plus1_v2)
        s1_u3_relu1 = self.stage1_unit3_relu1(s1_u3_bn1)
        s1_u3_conv1 = self.stage1_unit3_conv1(s1_u3_relu1)
        s1_u3_bn2 = self.stage1_unit3_bn2(s1_u3_conv1)
        s1_u3_relu2 = self.stage1_unit3_relu2(s1_u3_bn2)
        s1_u3_conv2_pad = self.stage1_unit3_conv2_pad(s1_u3_relu2)
        s1_u3_conv2 = self.stage1_unit3_conv2(s1_u3_conv2_pad)
        s1_u3_bn3 = self.stage1_unit3_bn3(s1_u3_conv2)
        s1_u3_relu3 = self.stage1_unit3_relu3(s1_u3_bn3)
        s1_u3_conv3 = self.stage1_unit3_conv3(s1_u3_relu3)
        plus2 = s1_u3_conv3 + plus1_v2
        # Stage 2
        ## Unit 1
        s2_u1_bn1 = self.stage2_unit1_bn1(plus2)
        s2_u1_relu1 = self.stage2_unit1_relu1(s2_u1_bn1)
        s2_u1_conv1 = self.stage2_unit1_conv1(s2_u1_relu1)
        s2_u1_sc = self.stage2_unit1_sc(s2_u1_relu1)
        s2_u1_bn2 = self.stage2_unit1_bn2(s2_u1_conv1)
        s2_u1_relu2 = self.stage2_unit1_relu2(s2_u1_bn2)
        s2_u1_conv2_pad = self.stage2_unit1_conv2_pad(s2_u1_relu2)
        s2_u1_conv2 = self.stage2_unit1_conv2(s2_u1_conv2_pad)
        s2_u1_bn3 = self.stage2_unit1_bn3(s2_u1_conv2)
        s2_u1_relu3 = self.stage2_unit1_relu3(s2_u1_bn3)
        s2_u1_conv3 = self.stage2_unit1_conv3(s2_u1_relu3)
        plus3 = s2_u1_conv3 + s2_u1_sc
        ## Unit 2
        s2_u2_bn1 = self.stage2_unit2_bn1(plus3)
        s2_u2_relu1 = self.stage2_unit2_relu1(s2_u2_bn1)
        s2_u2_conv1 = self.stage2_unit2_conv1(s2_u2_relu1)
        s2_u2_bn2 = self.stage2_unit2_bn2(s2_u2_conv1)
        s2_u2_relu2 = self.stage2_unit2_relu2(s2_u2_bn2)
        s2_u2_conv2_pad = self.stage2_unit2_conv2_pad(s2_u2_relu2)
        s2_u2_conv2 = self.stage2_unit2_conv2(s2_u2_conv2_pad)
        s2_u2_bn3 = self.stage2_unit2_bn3(s2_u2_conv2)
        s2_u2_relu3 = self.stage2_unit2_relu3(s2_u2_bn3)
        s2_u2_conv3 = self.stage2_unit2_conv3(s2_u2_relu3)
        plus4 = s2_u2_conv3 + plus3
        ## Unit 3
        s2_u3_bn1 = self.stage2_unit3_bn1(plus4)
        s2_u3_relu1 = self.stage2_unit3_relu1(s2_u3_bn1)
        s2_u3_conv1 = self.stage2_unit3_conv1(s2_u3_relu1)
        s2_u3_bn2 = self.stage2_unit3_bn2(s2_u3_conv1)
        s2_u3_relu2 = self.stage2_unit3_relu2(s2_u3_bn2)
        s2_u3_conv2_pad = self.stage2_unit3_conv2_pad(s2_u3_relu2)
        s2_u3_conv2 = self.stage2_unit3_conv2(s2_u3_conv2_pad)
        s2_u3_bn3 = self.stage2_unit3_bn3(s2_u3_conv2)
        s2_u3_relu3 = self.stage2_unit3_relu3(s2_u3_bn3)
        s2_u3_conv3 = self.stage2_unit3_conv3(s2_u3_relu3)
        plus5 = s2_u3_conv3 + plus4
        ## Unit 4
        s2_u4_bn1 = self.stage2_unit4_bn1(plus5)
        s2_u4_relu1 = self.stage2_unit4_relu1(s2_u4_bn1)
        s2_u4_conv1 = self.stage2_unit4_conv1(s2_u4_relu1)
        s2_u4_bn2 = self.stage2_unit4_bn2(s2_u4_conv1)
        s2_u4_relu2 = self.stage2_unit4_relu2(s2_u4_bn2)
        s2_u4_conv2_pad = self.stage2_unit4_conv2_pad(s2_u4_relu2)
        s2_u4_conv2 = self.stage2_unit4_conv2(s2_u4_conv2_pad)
        s2_u4_bn3 = self.stage2_unit4_bn3(s2_u4_conv2)
        s2_u4_relu3 = self.stage2_unit4_relu3(s2_u4_bn3)
        s2_u4_conv3 = self.stage2_unit4_conv3(s2_u4_relu3)
        plus6 = s2_u4_conv3 + plus5
        # Stage 3
        ## Unit 1
        s3_u1_bn1 = self.stage3_unit1_bn1(plus6)
        s3_u1_relu1 = self.stage3_unit1_relu1(s3_u1_bn1)
        s3_u1_conv1 = self.stage3_unit1_conv1(s3_u1_relu1)
        s3_u1_sc = self.stage3_unit1_sc(s3_u1_relu1)
        s3_u1_bn2 = self.stage3_unit1_bn2(s3_u1_conv1)
        s3_u1_relu2 = self.stage3_unit1_relu2(s3_u1_bn2)
        s3_u1_conv2_pad = self.stage3_unit1_conv2_pad(s3_u1_relu2)
        s3_u1_conv2 = self.stage3_unit1_conv2(s3_u1_conv2_pad)
        ssh_m1_red_conv = self.ssh_m1_red_conv(s3_u1_relu2)
        s3_u1_bn3 = self.stage3_unit1_bn3(s3_u1_conv2)
        ssh_m1_red_conv_bn = self.ssh_m1_red_conv_bn(ssh_m1_red_conv)
        s3_u1_relu3 = self.stage3_unit1_relu3(s3_u1_bn3)
        ssh_m1_red_conv_relu = self.ssh_m1_red_conv_relu(ssh_m1_red_conv_bn)
        s3_u1_conv3 = self.stage3_unit1_conv3(s3_u1_relu3)
        plus7 = s3_u1_conv3 + s3_u1_sc
        ## Unit 2
        s3_u2_bn1 = self.stage3_unit2_bn1(plus7)
        s3_u2_relu1 = self.stage3_unit2_relu1(s3_u2_bn1)
        s3_u2_conv1 = self.stage3_unit2_conv1(s3_u2_relu1)
        s3_u2_bn2 = self.stage3_unit2_bn2(s3_u2_conv1)
        s3_u2_relu2 = self.stage3_unit2_relu2(s3_u2_bn2)
        s3_u2_conv2_pad = self.stage3_unit2_conv2_pad(s3_u2_relu2)
        s3_u2_conv2 = self.stage3_unit2_conv2(s3_u2_conv2_pad)
        s3_u2_bn3 = self.stage3_unit2_bn3(s3_u2_conv2)
        s3_u2_relu3 = self.stage3_unit2_relu3(s3_u2_bn3)
        s3_u2_conv3 = self.stage3_unit2_conv3(s3_u2_relu3)
        plus8 = s3_u2_conv3 + plus7
        ## Unit 3
        s3_u3_bn1 = self.stage3_unit3_bn1(plus8)
        s3_u3_relu1 = self.stage3_unit3_relu1(s3_u3_bn1)
        s3_u3_conv1 = self.stage3_unit3_conv1(s3_u3_relu1)
        s3_u3_bn2 = self.stage3_unit3_bn2(s3_u3_conv1)
        s3_u3_relu2 = self.stage3_unit3_relu2(s3_u3_bn2)
        s3_u3_conv2_pad = self.stage3_unit3_conv2_pad(s3_u3_relu2)
        s3_u3_conv2 = self.stage3_unit3_conv2(s3_u3_conv2_pad)
        s3_u3_bn3 = self.stage3_unit3_bn3(s3_u3_conv2)
        s3_u3_relu3 = self.stage3_unit3_relu3(s3_u3_bn3)
        s3_u3_conv3 = self.stage3_unit3_conv3(s3_u3_relu3)
        plus9 = s3_u3_conv3 + plus8
        ## Unit 4
        s3_u4_bn1 = self.stage3_unit4_bn1(plus9)
        s3_u4_relu1 = self.stage3_unit4_relu1(s3_u4_bn1)
        s3_u4_conv1 = self.stage3_unit4_conv1(s3_u4_relu1)
        s3_u4_bn2 = self.stage3_unit4_bn2(s3_u4_conv1)
        s3_u4_relu2 = self.stage3_unit4_relu2(s3_u4_bn2)
        s3_u4_conv2_pad = self.stage3_unit4_conv2_pad(s3_u4_relu2)
        s3_u4_conv2 = self.stage3_unit4_conv2(s3_u4_conv2_pad)
        s3_u4_bn3 = self.stage3_unit4_bn3(s3_u4_conv2)
        s3_u4_relu3 = self.stage3_unit4_relu3(s3_u4_bn3)
        s3_u4_conv3 = self.stage3_unit4_conv3(s3_u4_relu3)
        plus10 = s3_u4_conv3 + plus9
        ## Unit 4
        s3_u5_bn1 = self.stage3_unit5_bn1(plus10)
        s3_u5_relu1 = self.stage3_unit5_relu1(s3_u5_bn1)
        s3_u5_conv1 = self.stage3_unit5_conv1(s3_u5_relu1)
        s3_u5_bn2 = self.stage3_unit5_bn2(s3_u5_conv1)
        s3_u5_relu2 = self.stage3_unit5_relu2(s3_u5_bn2)
        s3_u5_conv2_pad = self.stage3_unit5_conv2_pad(s3_u5_relu2)
        s3_u5_conv2 = self.stage3_unit5_conv2(s3_u5_conv2_pad)
        s3_u5_bn3 = self.stage3_unit5_bn3(s3_u5_conv2)
        s3_u5_relu3 = self.stage3_unit5_relu3(s3_u5_bn3)
        s3_u5_conv3 = self.stage3_unit5_conv3(s3_u5_relu3)
        plus11 = s3_u5_conv3 + plus10
        ## Unit 6
        s3_u6_bn1 = self.stage3_unit6_bn1(plus11)
        s3_u6_relu1 = self.stage3_unit6_relu1(s3_u6_bn1)
        s3_u6_conv1 = self.stage3_unit6_conv1(s3_u6_relu1)
        s3_u6_bn2 = self.stage3_unit6_bn2(s3_u6_conv1)
        s3_u6_relu2 = self.stage3_unit6_relu2(s3_u6_bn2)
        s3_u6_conv2_pad = self.stage3_unit6_conv2_pad(s3_u6_relu2)
        s3_u6_conv2 = self.stage3_unit6_conv2(s3_u6_conv2_pad)
        s3_u6_bn3 = self.stage3_unit6_bn3(s3_u6_conv2)
        s3_u6_relu3 = self.stage3_unit6_relu3(s3_u6_bn3)
        s3_u6_conv3 = self.stage3_unit6_conv3(s3_u6_relu3)
        plus12 = s3_u6_conv3 + plus11
        # Stage 4
        ## Unit 1
        s4_u1_bn1 = self.stage4_unit1_bn1(plus12)
        s4_u1_relu1 = self.stage4_unit1_relu1(s4_u1_bn1)
        s4_u1_conv1 = self.stage4_unit1_conv1(s4_u1_relu1)
        s4_u1_sc = self.stage4_unit1_sc(s4_u1_relu1)
        s4_u1_bn2 = self.stage4_unit1_bn2(s4_u1_conv1)
        s4_u1_relu2 = self.stage4_unit1_relu2(s4_u1_bn2)
        s4_u1_conv2_pad = self.stage4_unit1_conv2_pad(s4_u1_relu2)
        s4_u1_conv2 = self.stage4_unit1_conv2(s4_u1_conv2_pad)
        ssh_c2_lateral = self.ssh_c2_lateral(s4_u1_relu2)
        s4_u1_bn3 = self.stage4_unit1_bn3(s4_u1_conv2)
        ssh_c2_lateral_bn = self.ssh_c2_lateral_bn(ssh_c2_lateral)
        s4_u1_relu3 = self.stage4_unit1_relu3(s4_u1_bn3)
        ssh_c2_lateral_relu = self.ssh_c2_lateral_relu(ssh_c2_lateral_bn)
        s4_u1_conv3 = self.stage4_unit1_conv3(s4_u1_relu3)
        plus13 = s4_u1_conv3 + s4_u1_sc
        ## Unit 2
        s4_u2_bn1 = self.stage4_unit2_bn1(plus13)
        s4_u2_relu1 = self.stage4_unit2_relu1(s4_u2_bn1)
        s4_u2_conv1 = self.stage4_unit2_conv1(s4_u2_relu1)
        s4_u2_bn2 = self.stage4_unit2_bn2(s4_u2_conv1)
        s4_u2_relu2 = self.stage4_unit2_relu2(s4_u2_bn2)
        s4_u2_conv2_pad = self.stage4_unit2_conv2_pad(s4_u2_relu2)
        s4_u2_conv2 = self.stage4_unit2_conv2(s4_u2_conv2_pad)
        s4_u2_bn3 = self.stage4_unit2_bn3(s4_u2_conv2)
        s4_u2_relu3 = self.stage4_unit2_relu3(s4_u2_bn3)
        s4_u2_conv3 = self.stage4_unit2_conv3(s4_u2_relu3)
        plus14 = s4_u2_conv3 + plus13
        ## Unit 3
        s4_u3_bn1 = self.stage4_unit3_bn1(plus14)
        s4_u3_relu1 = self.stage4_unit3_relu1(s4_u3_bn1)
        s4_u3_conv1 = self.stage4_unit3_conv1(s4_u3_relu1)
        s4_u3_bn2 = self.stage4_unit3_bn2(s4_u3_conv1)
        s4_u3_relu2 = self.stage4_unit3_relu2(s4_u3_bn2)
        s4_u3_conv2_pad = self.stage4_unit3_conv2_pad(s4_u3_relu2)
        s4_u3_conv2 = self.stage4_unit3_conv2(s4_u3_conv2_pad)
        s4_u3_bn3 = self.stage4_unit3_bn3(s4_u3_conv2)
        s4_u3_relu3 = self.stage4_unit3_relu3(s4_u3_bn3)
        s4_u3_conv3 = self.stage4_unit3_conv3(s4_u3_relu3)
        plus15 = s4_u3_conv3 + plus14
        ## Heh
        bn1 = self.bn1(plus15)
        relu1 = self.relu1(bn1)
        ssh_c3_lateral = self.ssh_c3_lateral(relu1)
        ssh_c3_lateral_bn = self.ssh_c3_lateral_bn(ssh_c3_lateral)
        ssh_c3_lateral_relu = self.ssh_c3_lateral_relu(ssh_c3_lateral_bn)
        ssh_m3_det_conv1_pad = self.ssh_m3_det_context_conv1_pad(ssh_c3_lateral_relu)
        ssh_m3_det_conv1 = self.ssh_m3_det_conv1(ssh_m3_det_conv1_pad)
        ssh_m3_det_context_conv1_pad = self.ssh_m3_det_context_conv1_pad(ssh_c3_lateral_relu)
        ssh_m3_det_context_conv1 = self.ssh_m3_det_context_conv1(ssh_m3_det_context_conv1_pad)
        ssh_c3_up = self.ssh_c3_up(ssh_c3_lateral_relu)
        ssh_m3_det_conv1_bn = self.ssh_m3_det_conv1_bn(ssh_m3_det_conv1)
        ssh_m3_det_context_conv1_bn = self.ssh_m3_det_context_conv1_bn(ssh_m3_det_context_conv1)
        x1_shape = ssh_c3_up.shape
        x2_shape = ssh_c2_lateral_relu.shape
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        crop0 = ssh_c3_up[:, :, offsets[1]:offsets[1] + x2_shape[2], offsets[2]:offsets[2] + x2_shape[3]]
        ssh_m3_det_context_conv1_relu = self.ssh_m3_det_context_conv1_relu(ssh_m3_det_context_conv1_bn)
        plus0_v2 = ssh_c2_lateral_relu + crop0
        ## Heh 2
        ssh_m3_det_context_conv2_pad = self.ssh_m3_det_context_conv2_pad(ssh_m3_det_context_conv1_relu)
        ssh_m3_det_context_conv2 = self.ssh_m3_det_context_conv2(ssh_m3_det_context_conv2_pad)
        ssh_m3_det_context_conv3_1_pad = self.ssh_m3_det_context_conv3_1_pad(ssh_m3_det_context_conv1_relu)
        ssh_m3_det_context_conv3_1 = self.ssh_m3_det_context_conv3_1(ssh_m3_det_context_conv3_1_pad)
        ssh_c2_aggr_pad = self.ssh_c2_aggr_pad(plus0_v2)
        ssh_c2_aggr = self.ssh_c2_aggr(ssh_c2_aggr_pad)
        ssh_m3_det_context_conv2_bn = self.ssh_m3_det_context_conv2_bn(ssh_m3_det_context_conv2)
        ssh_m3_det_context_conv3_1_bn = self.ssh_m3_det_context_conv3_1_bn(ssh_m3_det_context_conv3_1)
        ssh_c2_aggr_bn = self.ssh_c2_aggr_bn(ssh_c2_aggr)
        ssh_m3_det_context_conv3_1_relu = self.ssh_m3_det_context_conv3_1_relu(ssh_m3_det_context_conv3_1_bn)
        ssh_c2_aggr_relu = self.ssh_c2_aggr_relu(ssh_c2_aggr_bn)
        ssh_m3_det_context_conv3_2_pad = self.ssh_m3_det_context_conv3_2_pad(ssh_m3_det_context_conv3_1_relu)
        ssh_m3_det_context_conv3_2 = self.ssh_m3_det_context_conv3_2(ssh_m3_det_context_conv3_2_pad)
        ssh_m2_det_conv1_pad = self.ssh_m2_det_conv1_pad(ssh_c2_aggr_relu)
        ssh_m2_det_conv1 = self.ssh_m2_det_conv1(ssh_m2_det_conv1_pad)
        ssh_m2_det_context_conv1_pad = self.ssh_m2_det_context_conv1_pad(ssh_c2_aggr_relu)
        ssh_m2_det_context_conv1 = self.ssh_m2_det_context_conv1(ssh_m2_det_context_conv1_pad)
        ssh_m2_red_up = self.ssh_m2_red_up(ssh_c2_aggr_relu)
        ssh_m3_det_context_conv3_2_bn = self.ssh_m3_det_context_conv3_2_bn(ssh_m3_det_context_conv3_2)
        ssh_m2_det_conv1_bn = self.ssh_m2_det_conv1_bn(ssh_m2_det_conv1)
        ssh_m2_det_context_conv1_bn = self.ssh_m2_det_context_conv1_bn(ssh_m2_det_context_conv1)
        x1_shape = ssh_m2_red_up.shape
        x2_shape = ssh_m1_red_conv_relu.shape
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        crop1 = ssh_m2_red_up[:, :, offsets[1]:offsets[1] + x2_shape[2], offsets[2]:offsets[2] + x2_shape[3]]
        ssh_m3_det_concat = torch.cat(
            [ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn], dim=1
        )
        ssh_m2_det_context_conv1_relu = self.ssh_m2_det_context_conv1_relu(ssh_m2_det_context_conv1_bn)
        plus1_v1 = ssh_m1_red_conv_relu + crop1
        # Heh 3
        ssh_m3_det_concat_relu = self.ssh_m3_det_concat_relu(ssh_m3_det_concat)
        ssh_m2_det_context_conv2_pad = self.ssh_m2_det_context_conv2_pad(ssh_m2_det_context_conv1_relu)
        ssh_m2_det_context_conv2 = self.ssh_m2_det_context_conv2(ssh_m2_det_context_conv2_pad)
        ssh_m2_det_context_conv3_1_pad = self.ssh_m2_det_context_conv3_1_pad(ssh_m2_det_context_conv1_relu)
        ssh_m2_det_context_conv3_1 = self.ssh_m2_det_context_conv3_1(ssh_m2_det_context_conv3_1_pad)
        ssh_c1_aggr_pad = self.ssh_c1_aggr_pad(plus1_v1)
        ssh_c1_aggr = self.ssh_c1_aggr(ssh_c1_aggr_pad)
        face_rpn_cls_score_stride32 = self.face_rpn_cls_score_stride32(ssh_m3_det_concat_relu)
        inter_1 = torch.cat(
            [face_rpn_cls_score_stride32[:, :, :, 0], face_rpn_cls_score_stride32[:, :, :, 1]], dim=1
        )
        inter_2 = torch.cat(
            [face_rpn_cls_score_stride32[:, :, :, 2], face_rpn_cls_score_stride32[:, :, :, 3]], dim=1
        )
        final = torch.stack([inter_1, inter_2], dim=0)
        face_rpn_cls_score_reshape_stride32 = final.permute(1, 2, 3, 0)
        face_rpn_bbox_pred_stride32 = self.face_rpn_bbox_pred_stride32(ssh_m3_det_concat_relu)
        face_rpn_landmark_pred_stride32 = self.face_rpn_landmark_pred_stride32(ssh_m3_det_concat_relu)
        ssh_m2_det_context_conv2_bn = self.ssh_m2_det_context_conv2_bn(ssh_m2_det_context_conv2)
        ssh_m2_det_context_conv3_1_bn = self.ssh_m2_det_context_conv3_1_bn(ssh_m2_det_context_conv3_1)
        ssh_c1_aggr_bn = self.ssh_c1_aggr_bn(ssh_c1_aggr)
        ssh_m2_det_context_conv3_1_relu = self.ssh_m2_det_context_conv3_1_relu(ssh_m2_det_context_conv3_1_bn)
        ssh_c1_aggr_relu = self.ssh_c1_aggr_relu(ssh_c1_aggr_bn)
        face_rpn_cls_prob_stride32 = self.face_rpn_cls_prob_stride32(face_rpn_cls_score_reshape_stride32)
        input_shape = face_rpn_cls_prob_stride32.shape
        sz = input_shape[1] // 2
        inter_1 = face_rpn_cls_prob_stride32[:, :sz, :, 0]
        inter_2 = face_rpn_cls_prob_stride32[:, :sz, :, 1]
        inter_3 = face_rpn_cls_prob_stride32[:, sz:, :, 0]
        inter_4 = face_rpn_cls_prob_stride32[:, sz:, :, 1]
        final = torch.stack([inter_1, inter_3, inter_2, inter_4], dim=0) 
        face_rpn_cls_prob_reshape_stride32 = final.permute(1, 2, 3, 0)
        ssh_m2_det_context_conv3_2_pad = self.ssh_m2_det_context_conv3_2_pad(ssh_m2_det_context_conv3_1_relu)
        ssh_m2_det_context_conv3_2 = self.ssh_m2_det_context_conv3_2(ssh_m2_det_context_conv3_2_pad)
        ssh_m1_det_conv1_pad = self.ssh_m1_det_conv1_pad(ssh_c1_aggr_relu)
        ssh_m1_det_conv1 = self.ssh_m1_det_conv1(ssh_m1_det_conv1_pad)
        ssh_m1_det_context_conv1_pad = self.ssh_m1_det_context_conv1_pad(ssh_c1_aggr_relu)
        ssh_m1_det_context_conv1 = self.ssh_m1_det_context_conv1(ssh_m1_det_context_conv1_pad)
        ssh_m2_det_context_conv3_2_bn = self.ssh_m2_det_context_conv3_2_bn(ssh_m2_det_context_conv3_2)
        ssh_m1_det_conv1_bn = self.ssh_m1_det_conv1_bn(ssh_m1_det_conv1)
        ssh_m1_det_context_conv1_bn = self.ssh_m1_det_context_conv1_bn(ssh_m1_det_context_conv1)
        print(ssh_m2_det_conv1_bn.shape)
        print(ssh_m2_det_context_conv2_bn.shape)
        print(ssh_m2_det_context_conv3_2_bn.shape)
        ssh_m2_det_concat = torch.cat(
            [ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, ssh_m2_det_context_conv3_2_bn], dim=1
        )
        print(ssh_m2_det_concat.shape)
        ssh_m1_det_context_conv1_relu = self.ssh_m1_det_context_conv1_relu(ssh_m1_det_context_conv1_bn)
        ssh_m2_det_concat_relu = self.ssh_m2_det_concat_relu(ssh_m2_det_concat)
        ssh_m1_det_context_conv2_pad = self.ssh_m1_det_context_conv2_pad(ssh_m1_det_context_conv1_relu)
        ssh_m1_det_context_conv2 = self.ssh_m1_det_context_conv2(ssh_m1_det_context_conv2_pad)
        ssh_m1_det_context_conv3_1_pad = self.ssh_m1_det_context_conv3_1_pad(ssh_m1_det_context_conv1_relu)
        ssh_m1_det_context_conv3_1 = self.ssh_m1_det_context_conv3_1(ssh_m1_det_context_conv3_1_pad)
        face_rpn_cls_score_stride16 = self.face_rpn_cls_score_stride16(ssh_m2_det_concat_relu)
        # Assuming face_rpn_cls_score_stride16 is in NCHW format (PyTorch default)
        inter_1 = torch.cat(
            [face_rpn_cls_score_stride16[:, 0, :, :], face_rpn_cls_score_stride16[:, 1, :, :]], dim=1
        )
        inter_2 = torch.cat(
            [face_rpn_cls_score_stride16[:, 2, :, :], face_rpn_cls_score_stride16[:, 3, :, :]], dim=1
        )
        final = torch.stack([inter_1, inter_2], dim=0)  # Stack along a new dimension
        face_rpn_cls_score_reshape_stride16 = final.permute(1, 2, 3, 0)  # Equivalent to tf.transpose
        face_rpn_bbox_pred_stride16 = self.face_rpn_bbox_pred_stride16(ssh_m2_det_concat_relu)
        face_rpn_landmark_pred_stride16 = self.face_rpn_landmark_pred_stride16(ssh_m2_det_concat_relu)
        ssh_m1_det_context_conv2_bn = self.ssh_m1_det_context_conv2_bn(ssh_m1_det_context_conv2)
        ssh_m1_det_context_conv3_1_bn = self.ssh_m1_det_context_conv3_1_bn(ssh_m1_det_context_conv3_1)
        ssh_m1_det_context_conv3_1_relu = self.ssh_m1_det_context_conv3_1_relu(ssh_m1_det_context_conv3_1_bn)
        face_rpn_cls_prob_stride16 = self.face_rpn_cls_prob_stride16(face_rpn_cls_score_reshape_stride16)
        input_shape = face_rpn_cls_prob_stride16.shape
        sz = input_shape[1] // 2
        inter_1 = face_rpn_cls_prob_stride16[:, :sz, :, 0]
        inter_2 = face_rpn_cls_prob_stride16[:, :sz, :, 1]
        inter_3 = face_rpn_cls_prob_stride16[:, sz:, :, 0]
        inter_4 = face_rpn_cls_prob_stride16[:, sz:, :, 1]
        final = torch.stack([inter_1, inter_3, inter_2, inter_4], dim=0)
        face_rpn_cls_prob_reshape_stride16 = final.permute(1, 2, 3, 0)
        ssh_m1_det_context_conv3_2_pad = self.ssh_m1_det_context_conv3_2_pad(ssh_m1_det_context_conv3_1_relu)
        ssh_m1_det_context_conv3_2 = self.ssh_m1_det_context_conv3_2(ssh_m1_det_context_conv3_2_pad)
        ssh_m1_det_context_conv3_2_bn = self.ssh_m1_det_context_conv3_2_bn(ssh_m1_det_context_conv3_2)
        ssh_m1_det_concat = torch.cat(
            [ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, ssh_m1_det_context_conv3_2_bn], dim=1
        )
        ssh_m1_det_concat_relu = self.ssh_m1_det_concat_relu(ssh_m1_det_concat)
        face_rpn_cls_score_stride8 = self.face_rpn_cls_score_stride8(ssh_m1_det_concat_relu)
        
        inter_1 = torch.cat(
            [face_rpn_cls_score_stride8[:, 0, :, :], face_rpn_cls_score_stride8[:, 1, :, :]], dim=1
        )
        inter_2 = torch.cat(
            [face_rpn_cls_score_stride8[:, 2, :, :], face_rpn_cls_score_stride8[:, 3, :, :]], dim=1
        )
        final = torch.stack([inter_1, inter_2], dim=0)
        face_rpn_cls_score_reshape_stride8 = final.permute(1, 2, 3, 0)
        face_rpn_bbox_pred_stride8 = self.face_rpn_bbox_pred_stride8(ssh_m1_det_concat_relu)
        face_rpn_landmark_pred_stride8 = self.face_rpn_landmark_pred_stride8(ssh_m1_det_concat_relu)
        face_rpn_cls_prob_stride8 = self.face_rpn_cls_prob_stride8(face_rpn_cls_score_reshape_stride8)
        input_shape = face_rpn_cls_prob_stride8.shape
        sz = input_shape[1] // 2  # Integer division
        # Slicing operations (adjusting for PyTorch's NCHW format)
        inter_1 = face_rpn_cls_prob_stride8[:, :sz, :, 0]
        inter_2 = face_rpn_cls_prob_stride8[:, :sz, :, 1]
        inter_3 = face_rpn_cls_prob_stride8[:, sz:, :, 0]
        inter_4 = face_rpn_cls_prob_stride8[:, sz:, :, 1]
        # Stack tensors along a new dimension
        final = torch.stack([inter_1, inter_3, inter_2, inter_4], dim=0)
        # Transpose the final tensor
        face_rpn_cls_prob_reshape_stride8 = final.permute(1, 2, 3, 0)
        return [
            face_rpn_cls_prob_reshape_stride32,
            face_rpn_bbox_pred_stride32,
            face_rpn_landmark_pred_stride32,
            face_rpn_cls_prob_reshape_stride16,
            face_rpn_bbox_pred_stride16,
            face_rpn_landmark_pred_stride16,
            face_rpn_cls_prob_reshape_stride8,
            face_rpn_bbox_pred_stride8,
            face_rpn_landmark_pred_stride8,
        ]
