import tensorflow as tf
import numpy as np
import torch

import sys

sys.path.insert(0, "/home/sandro/Projekte/github_projects/unsupervised-disentangling/")
# tf.enable_eager_execution()
from src.tf import ops
from src.torch import ops_pt


def tf_allclose_pt(t_tf, t_pt):
    """ convert both tensors to numpy and assert """
    t_tf = np.array(t_tf)
    t_pt = t_pt.numpy()
    return np.allclose(t_tf, t_pt)


def test_part_map_to_mu_L_inv():
    from supermariopy.tfutils import nn as tfnn

    P = tf.ones((1, 1, 2))
    stddev = tf.ones((1, 1, 2)) * 0.05
    w = 128
    h = 128
    hm = tfnn.tf_hm(P, w, h, stddev)

    mu, L_inv = ops.part_map_to_mu_L_inv(hm, scal=tf.ones((1, 1)))

    hm_pt = np.array(hm).transpose((0, 3, 1, 2))
    hm_pt = torch.tensor(hm_pt)
    mu_pt, L_inv_pt = ops.part_map_to_mu_L_inv_pt(hm_pt, scal=torch.ones((1, 1)))

    assert tf_allclose_pt(mu, mu_pt)
    assert tf_allclose_pt(L_inv, L_inv_pt)


def test_augm():
    x = torch.ones((1, 128, 128, 3))
    augm = ops_pt.augm(x, 0.1, 0.1, 0.1, 0.1, 0.1)


def test_abs_det_jacobian():
    x = torch.ones((2, 2, 128, 128))

    y = ops_pt.AbsDetJacobian(x)


def test_rand_scal():
    tf.set_random_seed(42)
    torch.manual_seed(42)

    y = ops.random_scal(2, 0, 1)
    y_pt = ops_pt.random_scal(2, 0, 1)
    assert y.shape == y_pt.shape


def test_feat_mu_to_enc():
    # TODO
    pass
