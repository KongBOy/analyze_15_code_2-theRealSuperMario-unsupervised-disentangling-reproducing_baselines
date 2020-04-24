from utils import wrappy
import tensorflow as tf
from architecture_ops import _residual, _conv_bn_relu, _conv, nccuc
from ops import softmax, get_features
import tensorflow.keras as tfk
import torch

import architecture_ops

import ops
import ops_pt


@wrappy
def discriminator_patch(image, train):
    padding = "VALID"
    x0 = image
    x1 = tf.layers.conv2d(
        x0,
        32,
        4,
        strides=1,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_0",
    )  # 46
    x1 = tf.layers.batch_normalization(x1, training=train, name="bn_0")
    x1 = tf.layers.conv2d(
        x1,
        64,
        4,
        strides=2,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_1",
    )  # 44
    x1 = tf.layers.batch_normalization(x1, training=train, name="bn_1")
    x2 = tf.layers.conv2d(
        x1,
        128,
        4,
        strides=2,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_2",
    )  # 10
    x2 = tf.layers.batch_normalization(x2, training=train, name="bn_2")
    x3 = tf.layers.conv2d(
        x2,
        256,
        4,
        strides=2,
        padding=padding,
        activation=tf.nn.leaky_relu,
        name="conv_3",
    )  # 4
    x3 = tf.layers.batch_normalization(x3, training=train, name="bn_3")
    x4 = tf.reshape(x3, shape=[-1, 4 * 4 * 256])
    x4 = tf.layers.dense(x4, 1, name="last_fc")
    return tf.nn.sigmoid(x4), x4


class Disicriminator_Patch(tfk.Model):
    def __init__(self, config):
        super().__init__()

        self.padding = "VALID"

        # Have to seperately handle batchnorms to add training=True flag during evaluation
        # see here https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
        self.batch_norms = [
            tfk.layers.BatchNormalization(),
            tfk.layers.BatchNormalization(),
            tfk.layers.BatchNormalization(),
            tfk.layers.BatchNormalization(),
        ]

        self.convs = [
            tfk.layers.Conv2D(
                32, 4, strides=1, padding=self.padding, activation=tf.nn.leaky_relu
            ),
            tfk.layers.Conv2D(
                64, 4, strides=2, padding=self.padding, activation=tf.nn.leaky_relu
            ),
            tfk.layers.Conv2D(
                128, 4, strides=2, padding=self.padding, activation=tf.nn.leaky_relu
            ),
            tfk.layers.Conv2D(
                256, 4, strides=2, padding=self.padding, activation=tf.nn.leaky_relu
            ),
        ]

        self.final_fc = tfk.layers.Dense(1)

        input_shape = (config["batch_size"], 49, 49, 3)
        self.build(input_shape)

    def call(self, x, training):
        for conv, bn in zip(self.convs, self.batch_norms):
            x = bn(conv(x), training=training)

        x = tf.reshape(x, (-1, 4 * 4 * 256))
        logits = self.final_fc(x)
        probs = tf.nn.sigmoid(logits)
        return probs, logits


@wrappy
def decoder(encoding_list, train, reconstr_dim, n_reconstruction_channels):
    """
    :param encoding_list:
        list of feature maps at each stage to merge.
        For `reconstr_dim = 128?` this is something like
        encoding_list = [
            tf.zeros((1, 128 // (2 ** i), 128 // (2 ** i), 128)) for i in range(6)
        ]
    :param train:
    :param reconstr_dim:
    :param n_reconstruction_channels:
    :return:
    """
    padding = "SAME"

    input = encoding_list[-1]  # 128 channels
    conv1 = nccuc(
        input, encoding_list[-2], [512, 512], padding, train, name=1
    )  # 8, 64 channels
    conv2 = nccuc(
        conv1, encoding_list[-3], [512, 256], padding, train, name=2
    )  # 16, 384 channels
    conv3 = nccuc(conv2, encoding_list[-4], [256, 256], padding, train, name=3)  # 32

    if reconstr_dim == 128:
        conv4 = nccuc(
            conv3, encoding_list[-5], [256, 128], padding, train, name=4
        )  # 64
        conv5 = nccuc(
            conv4, encoding_list[-6], [128, 64], padding, train, name=5
        )  # 128
        conv6 = tf.layers.conv2d(
            conv5,
            n_reconstruction_channels,
            6,
            strides=1,
            padding="SAME",
            activation=tf.nn.sigmoid,
            name="conv_6",
        )
        reconstruction = conv6  # 128

    if reconstr_dim == 256:
        conv4 = nccuc(
            conv3, encoding_list[-5], [256, 128], padding, train, name=4
        )  # 64
        conv5 = nccuc(
            conv4, encoding_list[-6], [128, 128], padding, train, name=5
        )  # 128
        conv6 = nccuc(
            conv5, encoding_list[-7], [128, 64], padding, train, name=6
        )  # 256
        conv7 = tf.layers.conv2d(
            conv6,
            n_reconstruction_channels,
            6,
            strides=1,
            padding="SAME",
            activation=tf.nn.sigmoid,
            name="conv_7",
        )
        reconstruction = conv7  # 256
    return reconstruction


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
        for in_c, F in zip(self.in_channels, self.n_filters):
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
            print(A.shape)

        A = torch.nn.ConstantPad2d(padding=(2, 3, 3, 2), value=0)(
            A
        )  # Mimiks same padding for kernel size 6 and stride 1
        A = self.final_conv(A)
        A = torch.nn.Sigmoid()(A)
        return A


# TODO: decoder 256s


def _hourglass(inputs, n, numOut, train, name="hourglass"):
    """ Hourglass Module
    Args:
        inputs	: Input Tensor
        n		: Number of downsampling step
        numOut	: Number of Output Features (channels)
        name	: Name of the block
    """
    with tf.variable_scope(name):
        # Upper Branch
        up_1 = _residual(inputs, numOut, train=train, name="up_1")
        # Lower Branch
        low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding="VALID")
        low_1 = _residual(low_, numOut, train=train, name="low_1")

        if n > 0:
            low_2 = _hourglass(low_1, n - 1, numOut, train=train, name="low_2")
        else:
            low_2 = _residual(low_1, numOut, train=train, name="low_2")

        low_3 = _residual(low_2, numOut, train=train, name="low_3")
        up_2 = tf.image.resize_nearest_neighbor(
            low_3, tf.shape(low_3)[1:3] * 2, name="upsampling"
        )
        return tf.add_n([up_2, up_1], name="out_hg")


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


@wrappy
def seperate_hourglass(inputs, train, n_landmark, n_features, nFeat_1, nFeat_2):
    _, h, w, c = inputs.get_shape().as_list()
    nLow = 4  # hourglass preprocessing reduces by factor two hourglass by factor 16 (2⁴)  e.g. 128 -> 4
    n_Low_feat = 1
    dropout_rate = 0.2

    # Storage Table
    hg = [None] * 2
    ll = [None] * 2
    ll_ = [None] * 2
    drop = [None] * 2
    out = [None] * 2
    out_ = [None] * 2
    sum_ = [None] * 2

    with tf.variable_scope("model"):
        with tf.variable_scope("preprocessing"):
            if h == 256:
                pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name="pad_1")
                conv1 = _conv_bn_relu(
                    pad1,
                    filters=64,
                    train=train,
                    kernel_size=6,
                    strides=2,
                    name="conv_256_to_128",
                )
                r1 = _residual(conv1, num_out=128, train=train, name="r1")
                pool1 = tf.contrib.layers.max_pool2d(
                    r1, [2, 2], [2, 2], padding="VALID"
                )
                r2 = _residual(pool1, num_out=int(nFeat_1 / 2), train=train, name="r2")
                r3 = _residual(r2, num_out=nFeat_1, train=train, name="r3")

            elif h == 128:
                pad1 = tf.pad(
                    inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name="pad_1"
                )  # shape [1, 132, 132, 10]
                conv1 = _conv_bn_relu(
                    pad1,
                    filters=64,
                    train=train,
                    kernel_size=6,
                    strides=2,
                    name="conv_64_to_32",
                )  # shape [1, 64, 64, 64]
                r3 = _residual(conv1, num_out=nFeat_1, train=train, name="r3")
                # shape [1, 64, 64, nFeat_1]
            elif h == 64:
                pad1 = tf.pad(inputs, [[0, 0], [3, 2], [3, 2], [0, 0]], name="pad_1")
                conv1 = _conv_bn_relu(
                    pad1,
                    filters=64,
                    train=train,
                    kernel_size=6,
                    strides=1,
                    name="conv_64_to_32",
                )
                r3 = _residual(conv1, num_out=nFeat_1, train=train, name="r3")

            else:
                raise ValueError

        with tf.variable_scope("stage_0"):
            hg[0] = _hourglass(
                r3, nLow, nFeat_1, train=train, name="hourglass"
            )  # [1, 64, 64, nFeat_1]
            drop[0] = tf.layers.dropout(
                hg[0], rate=dropout_rate, training=train, name="dropout"
            )  # [1, 64, 64, nFeat_1]
            ll[0] = _conv_bn_relu(
                drop[0],
                nFeat_1,
                train=train,
                kernel_size=1,
                strides=1,
                pad="VALID",
                name="conv",
            )  # [1, 64, 64, nFeat_1]
            ll_[0] = _conv(ll[0], nFeat_1, 1, 1, "VALID", "ll")  # [1, 64, 64, nFeat_1]
            out[0] = _conv(
                ll[0], n_landmark, 1, 1, "VALID", "out"
            )  # [1, 64, 64, n_landmark]
            out_[0] = _conv(
                softmax(out[0]), nFeat_1, 1, 1, "VALID", "out_"
            )  # [1, 64, 64, nFeat_1]
            sum_[0] = tf.add_n([out_[0], r3], name="merge")  # [1, 64, 64, nFeat_1]

        with tf.variable_scope("stage_1"):
            hg[1] = _hourglass(
                sum_[0], n_Low_feat, nFeat_2, train=train, name="hourglass"
            )  # [1, 64, 64, nFeat_2]
            drop[1] = tf.layers.dropout(
                hg[1], rate=dropout_rate, training=train, name="dropout"
            )  # [1, 64, 64, nFeat_2]
            ll[1] = _conv_bn_relu(
                drop[1],
                nFeat_2,
                train=train,
                kernel_size=1,
                strides=1,
                pad="VALID",
                name="conv",
            )  # [1, 64, 64, nFeat_2]

            out[1] = _conv(ll[1], n_features, 1, 1, "VALID", "out")
            # [1, 64, 64, n_features]

        features = out[1]  # [1, 64, 64, nFeat_2]
        return softmax(out[0]), features


decoder_map = {"standard": decoder}
encoder_map = {"seperate": seperate_hourglass}


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
