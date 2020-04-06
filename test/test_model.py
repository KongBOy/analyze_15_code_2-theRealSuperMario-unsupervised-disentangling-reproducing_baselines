import pytest
import numpy as np

import torch
import tensorflow as tf
from supermariopy.ptutils import nn as smptnn
from supermariopy.tfutils import nn as smtfnn

import sys

sys.path.insert(0, "/home/sandro/Projekte/github_projects/unsupervised-disentangling/")
tf.enable_eager_execution()


from architecture_ops import nccuc


class Test_Model:
    def test_model(self):
        from model import Model, ModelArgs
        from supermariopy.tfutils import tps

        N = 1
        H = 128
        W = 128
        C = 3
        image_batch = np.zeros((N, H, W, C), dtype=np.float32)
        image_batch_tiled = tf.tile(image_batch, [2, 1, 1, 1])
        arg = ModelArgs(bn=1)

        tps_params = tps.no_transformation_parameters(2 * N)
        tps_param_dic = tps.tps_parameters(**tps_params)
        from dotmap import DotMap

        tps_param_dic = DotMap(tps_param_dic)

        model = Model(
            image_batch_tiled, arg, tps_param_dic, optimize=False, visualize=False
        )


def test_convs():
    N = 1
    H = 128
    W = 128
    C = 10
    x = tf.ones((N, H, W, C))
    x = tf.layers.Conv2D(C, kernel_size=1, strides=(1, 1), padding="VALID")(x)
    assert smnn.shape_as_list(x) == [1, H, W, C]

    x = tf.ones((N, H, W, C))
    x = tf.layers.Conv2D(C, kernel_size=3, strides=(1, 1), padding="VALID")(x)
    assert smnn.shape_as_list(x) == [1, H - 2, W - 2, C]

