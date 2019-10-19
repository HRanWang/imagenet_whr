from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import math
import torch
import numpy as np

__all__ = ['resnet','resnet50']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)


def conv_1_3x3():
    return nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 'SAME'
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True))
                         # TODO: nn.MaxPool2d(kernel_size=3, stride=2, padding=0))  # 'valid'

def conv_1_3x3_dconv():
    return nn.Sequential(Dconv_shuffle(3, 64, 3, 1, 1),
                         nn.BatchNorm2d(64),
                         nn.ReLU(inplace=True))
                         # TODO: nn.MaxPool2d(kernel_size=3, stride=2, padding=0))  # 'valid'

class bottleneck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2)):
        super(bottleneck, self).__init__()
        plane1, plane2, plane3 = planes
        self.outchannels = plane3
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=strides, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.conv4 = nn.Conv2d(inplanes, plane3, kernel_size=1, stride=strides, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.conv4(input_tensor)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out


class identity_block3(nn.Module):
    def __init__(self, inplanes, planes, kernel_size):
        super(identity_block3, self).__init__()
        plane1, plane2, plane3 = planes
        self.outchannels = plane3
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)
        self.conv2 = nn.Conv2d(plane1, plane2, kernel_size=kernel_size, stride=1, padding=int((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm2d(plane2)
        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor, return_conv3_out=False):  # return_conv3_out is only served for grad_cam.py
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out_conv3 = self.conv3(out)
        out = self.bn3(out_conv3)

        out += input_tensor
        out = self.relu(out)
        if return_conv3_out:
            return out, out_conv3
        else:
            return out
class Dconv_shuffle(nn.Module):
    """
    Deformable convolution with random shuffling of the feature map.
    Random shuffling only happened within each page independently.
    The sampling locations are generated for each forward pass during the training.
    """
    def __init__(self, inplane, outplane, kernel_size, stride, padding):
        super(Dconv_shuffle, self).__init__()
        print('cifar Dconv_shuffle is used')
        self.dilated_conv = nn.Conv2d(inplane, outplane, kernel_size=kernel_size, stride=stride, padding=padding,
                                      bias=False)
        self.indices = None

    def _setup(self, inplane, spatial_size):
        self.indices = np.empty((inplane, spatial_size), dtype=np.int64)
        for i in range(inplane):
            self.indices[i, :] = np.arange(self.indices.shape[1])+ i*self.indices.shape[1]

    def forward(self, x):

        x_shape = x.size()  # [128, 3, 32, 32]
        x = x.view(x_shape[0], -1)
        if self.indices is None:
            self._setup(x_shape[1], x_shape[2]*x_shape[3])
        for i in range(x_shape[1]):
            np.random.shuffle(self.indices[i])
        x = x[:, torch.from_numpy(self.indices)].view(x_shape)
        return self.dilated_conv(x)
class bottleneck_shuffle(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, strides=(2, 2), type='error'):
        super(bottleneck_shuffle, self).__init__()
        plane1, plane2, plane3 = planes
        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)

        self.dconv1 = Dconv_shuffle(plane1, plane2, kernel_size=kernel_size, stride=strides, padding=1)
        self.bn2 = nn.BatchNorm2d(plane2)

        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)

        self.dconv2 = Dconv_shuffle(inplanes, plane3, kernel_size=1, stride=strides, padding=0)
        self.bn4 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dconv1(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = self.dconv2(input_tensor)
        shortcut = self.bn4(shortcut)

        out += shortcut
        out = self.relu(out)
        return out

class identity_block3_shuffle(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, type='error'):
        super(identity_block3_shuffle, self).__init__()
        plane1, plane2, plane3 = planes

        self.conv1 = nn.Conv2d(inplanes, plane1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(plane1)

        self.dconv = Dconv_shuffle(plane1, plane2, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(plane2)

        self.conv3 = nn.Conv2d(plane2, plane3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(plane3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dconv(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        x_shape = input_tensor.size()  # [128, 3, 32, 32]
        x = input_tensor.view(x_shape[0], x_shape[1] * x_shape[2] * x_shape[3])  # [128, 3*32*32]
        shuffled_input = torch.empty(x_shape[0], x_shape[1], x_shape[2], x_shape[3]).cuda(0)
        perm = torch.empty(0).float()
        for i in range(x_shape[1]):
            a = torch.randperm(x_shape[2] * x_shape[3]) + i * x_shape[2] * x_shape[3]
            perm = torch.cat((perm, a.float()), 0)
        shuffled_input[:, :, :, :] = x[:, perm.long()].view(x_shape[0], x_shape[1], x_shape[2], x_shape[3])

        out += shuffled_input
        out = self.relu(out)
        return out

class Resnet50(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top, layer=99, type='none'):
        print('resnet50 is used')
        super(Resnet50, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        block_ex = 4

        # Define the building blocks
        if layer > 0:
            self.conv_3x3 = conv_1_3x3()
        else:
            self.conv_3x3 = conv_1_3x3_dconv()

        if layer > 10:
            self.bottleneck_1 = bottleneck(16*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3, strides=(1, 1))
        else:
            self.bottleneck_1 = bottleneck_shuffle(16*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3, strides=(1, 1), type=type)
        if layer > 11:
            self.identity_block_1_1 = identity_block3(64*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3)
        else:
            self.identity_block_1_1 = identity_block3_shuffle(64*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3, type=type)
        if layer > 12:
            self.identity_block_1_2 = identity_block3(64*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3)
        else:
            self.identity_block_1_2 = identity_block3_shuffle(64*block_ex, [16*block_ex, 16*block_ex, 64*block_ex], kernel_size=3, type=type)

        if layer > 20:
            self.bottleneck_2 = bottleneck(64*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3, strides=(2, 2))
        else:
            self.bottleneck_2 = bottleneck_shuffle(64*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3, strides=(2, 2), type=type)
        if layer > 21:
            self.identity_block_2_1 = identity_block3(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3)
        else:
            self.identity_block_2_1 = identity_block3_shuffle(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3, type=type)
        if layer > 22:
            self.identity_block_2_2 = identity_block3(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3)
        else:
            self.identity_block_2_2 = identity_block3_shuffle(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3, type=type)
        if layer > 23:
            self.identity_block_2_3 = identity_block3(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3)
        else:
            self.identity_block_2_3 = identity_block3_shuffle(128*block_ex, [32*block_ex, 32*block_ex, 128*block_ex], kernel_size=3, type=type)

        if layer > 30:
            self.bottleneck_3 = bottleneck(128*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3, strides=(2, 2))
        else:
            self.bottleneck_3 = bottleneck_shuffle(128*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3, strides=(2, 2), type=type)
        if layer > 31:
            self.identity_block_3_1 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        else:
            self.identity_block_3_1 = identity_block3_shuffle(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3, type=type)
        if layer > 32:
            self.identity_block_3_2 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        else:
            self.identity_block_3_2 = identity_block3_shuffle(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3, type=type)
        if layer > 33:
            self.identity_block_3_3 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        else:
            self.identity_block_3_3 = identity_block3_shuffle(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3, type=type)
        if layer > 34:
            self.identity_block_3_4 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        else:
            self.identity_block_3_4 = identity_block3_shuffle(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3, type=type)
        if layer > 35:
            self.identity_block_3_5 = identity_block3(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3)
        else:
            self.identity_block_3_5 = identity_block3_shuffle(256*block_ex, [64*block_ex, 64*block_ex, 256*block_ex], kernel_size=3, type=type)

        if layer > 40:
            self.bottleneck_4 = bottleneck(256*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3, strides=(2, 2))
        else:
            self.bottleneck_4 = bottleneck_shuffle(256*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3, strides=(2, 2), type=type)
        if layer > 41:
            self.identity_block_4_1 = identity_block3(512*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3)
        else:
            self.identity_block_4_1 = identity_block3_shuffle(512*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3, type=type)
        if layer > 42:
            self.identity_block_4_2 = identity_block3(512*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3)
        else:
            self.identity_block_4_2 = identity_block3_shuffle(512*block_ex, [128*block_ex, 128*block_ex, 512*block_ex], kernel_size=3, type=type)

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # TODO: check the final size
        self.fc = nn.Linear(512*block_ex, num_classes)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # raise Exception('You are using a model without BN!!!')
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        # print(input_x.size())
        x = self.conv_3x3(input_x)
        # np.save('/nethome/yuefan/fanyue/dconv/fm3x3.npy', x.detach().cpu().numpy())
        # print(x.size())
        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x = self.identity_block_1_2(x)
        # print(x.size())
        x = self.bottleneck_2(x)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x = self.identity_block_2_3(x)
        # print(x.size())
        x = self.bottleneck_3(x)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x = self.identity_block_3_5(x)
        # print(x.size())
        x = self.bottleneck_4(x)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)
        # print("feature shape:", x.size())

        if self.include_top:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            # TODO: why there is no dropout
            x = self.fc(x)
        return x

class ResnetWHR(nn.Module):
    def __init__(self, dropout_rate, num_classes, include_top):
        """
        This is ResNet50 for PCB verison
        """
        super(ResnetWHR, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.include_top = include_top
        self.num_features = 512

        # Define the building blocks
        self.conv_3x3 = conv_1_3x3()

        self.bottleneck_1 = bottleneck(16, [16, 16, 64], kernel_size=3, strides=(1, 1))
        self.identity_block_1_1 = identity_block3(64, [16, 16, 64], kernel_size=3)
        self.identity_block_1_2 = identity_block3(64, [16, 16, 64], kernel_size=3)

        self.bottleneck_2 = bottleneck(64, [32, 32, 128], kernel_size=3, strides=(2, 2))
        self.identity_block_2_1 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_2 = identity_block3(128, [32, 32, 128], kernel_size=3)
        self.identity_block_2_3 = identity_block3(128, [32, 32, 128], kernel_size=3)

        self.bottleneck_3 = bottleneck(128, [64, 64, 256], kernel_size=3, strides=(2, 2))
        self.identity_block_3_1 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_2 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_3 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_4 = identity_block3(256, [64, 64, 256], kernel_size=3)
        self.identity_block_3_5 = identity_block3(256, [64, 64, 256], kernel_size=3)

        self.bottleneck_4 = bottleneck(256, [128, 128, 512], kernel_size=3, strides=(2, 2))
        self.identity_block_4_1 = identity_block3(512, [128, 128, 512], kernel_size=3)
        self.identity_block_4_2 = identity_block3(512, [128, 128, 512], kernel_size=3)

        # =======================================top=============================================
        # self.se1 = SELayer(64)
        # self.se2 = SELayer(128)
        # self.se3 = SELayer(256)

        # self.local_conv_layer1 = nn.Conv2d(64, self.num_features, kernel_size=1, padding=0, bias=False)
        # self.local_conv_layer2 = nn.Conv2d(128, self.num_features, kernel_size=1, padding=0, bias=False)
        # self.local_conv_layer3 = nn.Conv2d(256, self.num_features, kernel_size=1, padding=0, bias=False)
        # self.instance_layer1 = nn.Linear(self.num_features, self.num_classes)
        # self.instance_layer2 = nn.Linear(self.num_features, self.num_classes)
        # self.instance_layer3 = nn.Linear(self.num_features, self.num_classes)

        self.instance0 = nn.Linear(self.num_features, self.num_classes)
        self.instance1 = nn.Linear(self.num_features, self.num_classes)
        self.instance2 = nn.Linear(self.num_features, self.num_classes)
        self.instance3 = nn.Linear(self.num_features, self.num_classes)
        self.instance4 = nn.Linear(self.num_features, self.num_classes)
        # self.linear_list = []
        # for i in range(16):
        #     self.linear_list.append(nn.Linear(self.num_features, self.num_classes).cuda())

        # self.local_conv = nn.Conv2d(self.num_features, self.num_features, kernel_size=1, padding=0, bias=False)
        # self.local_bn = nn.BatchNorm2d(self.num_features)

        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_x):
        x = self.conv_3x3(input_x)

        x = self.bottleneck_1(x)
        x = self.identity_block_1_1(x)
        x_layer1 = self.identity_block_1_2(x)

        x = self.bottleneck_2(x_layer1)
        x = self.identity_block_2_1(x)
        x = self.identity_block_2_2(x)
        x_layer2 = self.identity_block_2_3(x)

        x = self.bottleneck_3(x_layer2)
        x = self.identity_block_3_1(x)
        x = self.identity_block_3_2(x)
        x = self.identity_block_3_3(x)
        x = self.identity_block_3_4(x)
        x_layer3 = self.identity_block_3_5(x)

        x = self.bottleneck_4(x_layer3)
        x = self.identity_block_4_1(x)
        x = self.identity_block_4_2(x)

        # x_layer1 = self.se1(x_layer1)
        # x_layer1 = nn.functional.avg_pool2d(x_layer1, kernel_size=(32, 32), stride=(1, 1))
        # x_layer1 = self.local_conv_layer1(x_layer1)
        # x_layer1 = x_layer1.contiguous().view(x_layer1.size(0), -1)
        # x_layer1 = self.instance_layer1(x_layer1)
        #
        # x_layer2 = self.se2(x_layer2)
        # x_layer2 = nn.functional.avg_pool2d(x_layer2, kernel_size=(16, 16), stride=(1, 1))
        # x_layer2 = self.local_conv_layer2(x_layer2)
        # x_layer2 = x_layer2.contiguous().view(x_layer2.size(0), -1)
        # x_layer2 = self.instance_layer2(x_layer2)
        #
        # x_layer3 = self.se3(x_layer3)
        # x_layer3 = nn.functional.avg_pool2d(x_layer3, kernel_size=(8, 8), stride=(1, 1))
        # x_layer3 = self.local_conv_layer3(x_layer3)
        # x_layer3 = x_layer3.contiguous().view(x_layer3.size(0), -1)
        # x_layer3 = self.instance_layer3(x_layer3)

        sx = x.size(2) / 4
        x = nn.functional.avg_pool2d(x, kernel_size=(sx, x.size(3)), stride=(sx, x.size(3)))  # 4x1

        # x = self.local_conv(x)
        # x = self.local_bn(x)
        # x = nn.functional.relu(x)

        x4 = nn.functional.avg_pool2d(x, kernel_size=(4, 1), stride=(1, 1))
        x4 = x4.contiguous().view(x4.size(0), -1)
        c4 = self.instance4(x4)

        # x = x.view(x.size(0), x.size(1), 16)
        # c_list = []
        # for i in range(16):
        #     x_offset = torch.empty(x.size(0), 512).cuda(0)
        #     # print(x_offset[:, :, :].size(), x[:, :, i].size())
        #     x_offset[:, :] = x[:, :, i]
        #     tmp = self.linear_list[i](x_offset)
        #     c_list.append(tmp)

        x = x.chunk(4, dim=2)
        x0 = x[0].contiguous().view(x[0].size(0), -1)
        x1 = x[1].contiguous().view(x[1].size(0), -1)
        x2 = x[2].contiguous().view(x[2].size(0), -1)
        x3 = x[3].contiguous().view(x[3].size(0), -1)
        c0 = self.instance0(x0)
        c1 = self.instance1(x1)
        c2 = self.instance2(x2)
        c3 = self.instance3(x3)
        return c0, c1, c2, c3, c4#c_list, c4##, x_layer1, x_layer2, x_layer3

def resnet50(**kwargs):
    """
    Constructs a ResNet model.
    """
    return Resnet50(**kwargs)