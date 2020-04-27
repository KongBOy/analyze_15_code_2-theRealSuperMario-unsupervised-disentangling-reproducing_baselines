import tensorflow as tf
import numpy as np
from src.tf.utils import wrappy
import torch


def _conv(inputs, filters, kernel_size=1, strides=1, pad="VALID", name="conv"):
    with tf.variable_scope(name):
        # Kernel for convolution, Xavier Initialisation
        kernel = tf.get_variable(
            shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            name="weights",
        )
        conv = tf.nn.conv2d(
            inputs, kernel, [1, strides, strides, 1], padding=pad, data_format="NHWC"
        )
        return conv


class ConvBnRelu(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, strides=1, pad=0):
        """conv without bias
        
        Parameters
        ----------
        torch : [type]
            [description]
        in_channels : [type]
            [description]
        out_channels : [type]
            [description]
        kernel_size : int, optional
            [description], by default 1
        strides : int, optional
            [description], by default 1
        pad : int, optional
            [description], by default 0
        """
        super(ConvBnRelu, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=strides,
            padding=pad,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return torch.nn.ReLU()(self.bn(self.conv(x)))


def _conv_bn_relu(
    inputs, filters, train, kernel_size=1, strides=1, pad="VALID", name="conv_bn_relu"
):
    with tf.variable_scope(name):
        kernel = tf.get_variable(
            shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
            initializer=tf.contrib.layers.xavier_initializer(uniform=False),
            name="weights",
        )
        conv = tf.nn.conv2d(
            inputs, kernel, [1, strides, strides, 1], padding=pad, data_format="NHWC"
        )
        norm = tf.nn.relu(
            tf.layers.batch_normalization(
                conv, momentum=0.9, epsilon=1e-5, training=train, name="bn"
            )
        )
        return norm


def _conv_block(inputs, numOut, train, name="conv_block"):
    with tf.variable_scope(name):
        with tf.variable_scope("norm_1"):
            norm_1 = tf.nn.relu(
                tf.layers.batch_normalization(
                    inputs, momentum=0.9, epsilon=1e-5, training=train, name="bn"
                )
            )

            conv_1 = _conv(
                norm_1,
                int(numOut / 2),
                kernel_size=1,
                strides=1,
                pad="VALID",
                name="conv",
            )
        with tf.variable_scope("norm_2"):
            norm_2 = tf.nn.relu(
                tf.layers.batch_normalization(
                    conv_1, momentum=0.9, epsilon=1e-5, training=train, name="bn"
                )
            )

            pad = tf.pad(norm_2, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name="pad")
            conv_2 = _conv(
                pad, int(numOut / 2), kernel_size=3, strides=1, pad="VALID", name="conv"
            )
        with tf.variable_scope("norm_3"):

            norm_3 = tf.nn.relu(
                tf.layers.batch_normalization(
                    conv_2, momentum=0.9, epsilon=1e-5, training=train, name="bn"
                )
            )

            conv_3 = _conv(
                norm_3, int(numOut), kernel_size=1, strides=1, pad="VALID", name="conv"
            )
        return conv_3


class ConvBLock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """conv without bias
        
        Parameters
        ----------
        torch : [type]
            [description]
        in_channels : [type]
            [description]
        out_channels : [type]
            [description]
        """
        super(ConvBLock, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels // 2, 1, stride=1, padding=0, bias=False
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels // 2, 3, stride=1, padding=0, bias=False
        )
        self.conv3 = torch.nn.Conv2d(
            out_channels // 2, out_channels, 1, stride=1, padding=0, bias=False
        )
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels // 2)
        self.bn3 = torch.nn.BatchNorm2d(out_channels // 2)

    def forward(self, x):
        y = torch.nn.ReLU()(self.bn1(x))
        y = self.conv1(y)
        y = torch.nn.ReLU()(self.bn2(y))
        y = torch.nn.ConstantPad2d(1, 0)(y)
        y = self.conv2(y)
        y = torch.nn.ReLU()(self.bn3(y))
        y = self.conv3(y)

        return y


def _skip_layer(inputs, num_out, name="skip_layer"):
    with tf.variable_scope(name):
        if inputs.get_shape().as_list()[3] == num_out:
            return inputs
        else:
            conv = _conv(inputs, num_out, kernel_size=1, strides=1, name="conv")
            return conv


def _residual(inputs, num_out, train, name="residual_block"):
    with tf.variable_scope(name):
        convb = _conv_block(inputs, num_out, train=train)
        skipl = _skip_layer(inputs, num_out)
        return tf.add_n([convb, skipl], name="res_block")


@wrappy
def nccuc(input_A, input_B, n_filters, padding, training, name):
    with tf.variable_scope("layer{}".format(name)):
        for i, F in enumerate(n_filters):
            if i < 1:
                x0 = input_A
                x1 = tf.layers.conv2d(
                    x0,
                    F,
                    (4, 4),
                    strides=(1, 1),
                    activation=None,
                    padding=padding,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                    name="conv_{}".format(i + 1),
                )
                x1 = tf.layers.batch_normalization(
                    x1, training=training, name="bn_{}".format(i + 1)
                )
                x1 = tf.nn.relu(x1, name="relu{}_{}".format(name, i + 1))

            elif i == 1:
                up_conv = tf.layers.conv2d_transpose(
                    x1,
                    filters=F,
                    kernel_size=4,
                    strides=2,
                    padding=padding,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                    name="upsample_{}".format(name),
                )

                up_conv = tf.nn.relu(up_conv, name="relu{}_{}".format(name, i + 1))
                return tf.concat(
                    [up_conv, input_B], axis=-1, name="concat_{}".format(name)
                )

            else:
                return x1


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        """
        super(ResidualBlock, self).__init__()
        self.x_conv = ConvBLock(in_channels, out_channels)
        if in_channels != out_channels:
            self.r_conv = ConvBLock(in_channels, out_channels)
        else:
            self.r_conv = None

    def forward(self, x):
        r = x
        y = self.x_conv(x)
        if self.r_conv:
            r = self.r_conv(r)
        return y + r


class NCCUC(torch.nn.Module):
    def __init__(self, in_channels, channels_conv, channels_out):
        """
        performs the following operatio:

        inputs: A, B

        x:= A 
        --> x := relu(bn(conv2d( x ))) 
        --> x := relu(conv_transpose2d(x))
        --> x := concat([x, B])
        return x

        To use running mean from training, put model in eval mode before applying batch norm.
        
        Parameters
        ----------
        torch : [type]
            [description]
        in_channels : [type]
            [description]
        channels_conv : [type]
            [description]
        channels_out : [type]
            [description]
        """
        super(NCCUC, self).__init__()

        self.conv = torch.nn.Conv2d(
            in_channels, channels_conv, kernel_size=3, stride=1, padding=1
        )
        self.conv_transpose = torch.nn.ConvTranspose2d(
            channels_conv, channels_out, kernel_size=4, stride=2, padding=1
        )
        self.bn = torch.nn.BatchNorm2d(channels_conv)

    def forward(self, A, B):
        """[summary]
        
        Parameters
        ----------
        A : torch.Tensor
            tensor shaped [N, C_A, H // 2, W // 2]
        B : torch.Tensor
            tensor shaped [N, C_B, H, W]
        
        Returns
        -------
        torch.Tensor
            tensor shaped [N, C_B + channels_out, H, W]
        """
        x = A
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.ReLU()(x)
        x = self.conv_transpose(x)
        x = torch.nn.ReLU()(x)
        y = torch.cat([x, B], dim=1)
        return y


def get_padding(kernel_size, stride, padding):
    """ get padding integer by padding description ["SAME", "VALID"] and kernel_size integer """
    # TODO: use actual formula

    if padding == "SAME":
        if kernel_size == 1 and stride == 1:
            return 0
        elif kernel_size == 3 and stride == 1:
            return 1
    raise NotImplementedError
