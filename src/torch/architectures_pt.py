import torch
from src.tf import architecture_ops
from src.torch import ops_pt


class Decoder128(torch.nn.Module):
    def __init__(self, in_channels, n_reconstruction_channels):
        super().__init__()

        self.in_channels = in_channels
        self.n_reconstruction_channels = n_reconstruction_channels

        self.n_filters = [
            [512, 512],
            [512, 256],
            [256, 256],
            [256, 128],
            [128, 64],  # only so much for reconstruction_dim 128
            # [256, 128],
            # [128, 128],
            # [128, 64],
        ]

        self.layers = []
        channels_A = self.in_channels[0]
        for in_c, F in zip(self.in_channels[1:], self.n_filters):
            l = architecture_ops.NCCUC(
                channels_A, channels_conv=F[0], channels_out=F[1]
            )
            self.layers.append(l)
            channels_A = F[1] + in_c
        self.layers = torch.nn.ModuleList(self.layers)
        self.final_conv = torch.nn.Conv2d(
            channels_A,
            self.n_reconstruction_channels,
            kernel_size=6,
            stride=1,
            padding=0,
        )

    def forward(self, x_list):
        """ x_list contains feature maps to merge in decreasing size order.

            x_list = [
                feature_map_128,
                feature_map_64,
                feature_map_32,
                feature_map_16,
                feature_map_8,
            ]
        """
        x_list_reverse = x_list[::-1]

        A = x_list_reverse[0]
        for l, B in zip(self.layers, x_list_reverse[1:]):
            A = l(A, B)

        A = torch.nn.ConstantPad2d(padding=(2, 3, 3, 2), value=0)(
            A
        )  # Mimiks same padding for kernel size 6 and stride 1
        A = self.final_conv(A)
        A = torch.nn.Sigmoid()(A)
        return A


class SeperateHourglass_128(torch.nn.Module):
    H = 128  # assumed input height during forward pass

    def __init__(
        self, in_channels, n_landmarks, n_features, n_features_1, n_features_2
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_landmarks = n_landmarks
        self.n_features = n_features
        self.n_features_1 = n_features_1
        self.n_features_2 = n_features_2

        self.conv_bn_relu_1 = architecture_ops.ConvBnRelu(
            self.in_channels, 64, kernel_size=6, strides=2
        )

        self.conv_bn_relu_2 = architecture_ops.ConvBnRelu(
            self.n_features_1, self.n_features_1, kernel_size=1, strides=1
        )
        self.conv_2_1 = torch.nn.Conv2d(
            self.n_features_1,
            self.n_features_1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv_2_2 = torch.nn.Conv2d(
            self.n_features_1,
            self.n_landmarks,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv_2_3 = torch.nn.Conv2d(
            self.n_landmarks,
            self.n_features_1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.res1 = architecture_ops.ResidualBlock(64, self.n_features_1)
        nLow = 4
        n_Low_feat = 1
        self.dropout_rate = 0.2

        self.hg1 = Hourglass(
            self.n_features_1, self.n_features_1, Bottleneck, 1, 64, nLow
        )
        self.hg2 = Hourglass(
            self.n_features_1, self.n_features_2, Bottleneck, 1, 64, n_Low_feat
        )

        self.conv_bn_relu_3 = architecture_ops.ConvBnRelu(
            self.n_features_2, self.n_features_2, kernel_size=1, strides=1
        )
        self.conv_3_1 = torch.nn.Conv2d(
            self.n_features_2, self.n_features, 1, 1, padding=0
        )

    def forward(self, x):
        hg = [None] * 2
        ll = [None] * 2
        ll_ = [None] * 2
        drop = [None] * 2
        out = [None] * 2
        out_ = [None] * 2
        sum_ = [None] * 2

        pad1 = torch.nn.ConstantPad2d(2, value=0)(x)
        conv1 = self.conv_bn_relu_1(pad1)
        r3 = self.res1(conv1)

        hg[0] = self.hg1(r3)
        drop[0] = torch.nn.Dropout(self.dropout_rate)(hg[0])
        ll[0] = self.conv_bn_relu_2(drop[0])
        ll_[0] = self.conv_2_1(ll[0])
        out[0] = self.conv_2_2(ll[0])
        out_[0] = self.conv_2_3(ops_pt.spatial_softmax(out[0]))
        sum_[0] = out_[0] + r3

        hg[1] = self.hg2(sum_[0])
        drop[1] = torch.nn.Dropout(self.dropout_rate)(hg[1])
        ll[1] = self.conv_bn_relu_3(drop[1])
        out[1] = self.conv_3_1(ll[1])
        features = out[1]

        return ops_pt.spatial_softmax(out[0]), features


class Bottleneck(torch.nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


import torch.nn as nn
import torch.nn.functional as F


class Hourglass(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, block, num_blocks, planes, depth,
    ):
        """Single Hourglass Network
        
        Parameters
        ----------
        block : Callable
            black function to use
        num_blocks : int
            how many blocks to use at each stage
        planes : int
            num features to use at each stage
        depth : int
            how many times to downsample
        """
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

        self.n_out = out_channels

        self.initial_conv = torch.nn.Conv2d(
            in_channels, 2 * planes, kernel_size=5, padding=2, stride=1, bias=False
        )
        self.final_conv = torch.nn.Conv2d(
            2 * planes, out_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        y = torch.nn.ReLU()(self.initial_conv(x))
        y = self._hour_glass_forward(self.depth, y)
        y = self.final_conv(y)
        return y


class Discriminator_Patch(torch.nn.Module):
    def __init__(self, in_channels=19):
        super().__init__()

        self.padding = "VALID"

        # Have to seperately handle batchnorms to add training=True flag during evaluation
        # see here https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
        self.batch_norms = [
            torch.nn.BatchNorm2d(32),
            torch.nn.BatchNorm2d(64),
            torch.nn.BatchNorm2d(128),
            torch.nn.BatchNorm2d(256),
        ]

        self.convs = [
            torch.nn.Conv2d(in_channels, 32, 4, 1, padding=0),
            torch.nn.Conv2d(32, 64, 4, stride=2, padding=0),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=0),
            torch.nn.Conv2d(128, 256, 4, stride=2, padding=0),
        ]
        self.final_fc = torch.nn.Linear(4 * 4 * 256, 1, bias=False)

    def forward(self, x):
        """x has to be a tensor with shape (None, 3, 49, 49)"""
        for conv, bn in zip(self.convs, self.batch_norms):
            x = bn(conv(x))

        x = x.view((-1, 4 * 4 * 256))
        logits = self.final_fc(x)
        probs = torch.sigmoid(logits)
        return probs, logits
