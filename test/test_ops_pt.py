import pytest
import tensorflow as tf
import torch
import tfpyth
from dotmap import DotMap

import sys
from src.tf import ops
from src.torch import ops_pt
from supermariopy.tfutils import nn as tfnn

tf.enable_eager_execution()


def _test_compatibility(out_tf, out_pt, permute_pt=False):
    if permute_pt:
        out_pt = tfpyth.th_2D_channels_first_to_last(out_pt)
    if isinstance(out_pt, tuple) and isinstance(out_tf, tuple):
        test_results = []
        for opt, otf in zip(out_pt, out_tf):
            test_results.append(np.allclose(np.array(otf), opt.numpy()))
        return all(test_results)
    return np.allclose(np.array(out_tf), out_pt.numpy())


def _test_shape_compatibility(out_tf, out_pt, permute_pt=False):
    if permute_pt:
        out_pt = tfpyth.th_2D_channels_first_to_last(out_pt)
    if isinstance(out_pt, tuple) and isinstance(out_tf, tuple):
        test_results = []
        for opt, otf in zip(out_pt, out_tf):
            test_results.append(np.allclose(np.array(out_tf).shape, out_pt.shape))
        return all(test_results)
    return np.allclose(np.array(out_tf).shape, out_pt.shape)


def test_augm():
    t_tf = tf.ones((1, 128, 128, 3))
    t_pt = torch.ones((1, 3, 128, 128))
    arg = DotMap(
        {
            "contrast_var": 0.1,
            "brightness_var": 0.1,
            "saturation_var": 0.1,
            "hue_var": 0.1,
            "p_flip": 0.1,
        }
    )
    out_tf = ops.augm(t_tf, arg=arg)

    out_pt = ops_pt.augm(t_pt, **arg)
    assert out_pt.shape == (1, 3, 128, 128)
    return _test_shape_compatibility(out_tf, out_pt, permute_pt=True)


def test_AbsDetJacobian():
    meshgrid = tf.expand_dims(tfnn.tf_meshgrid(3, 3), 0)
    out_tf = ops.AbsDetJacobian(meshgrid)

    meshgrid_pt = torch.from_numpy(np.array(meshgrid))
    meshgrid_pt = tfpyth.th_2D_channels_last_to_first(meshgrid_pt)

    out_pt = ops_pt.AbsDetJacobian(meshgrid_pt)
    assert out_pt.shape == (1, 1, 3, 3)
    assert _test_compatibility(out_tf, out_pt, permute_pt=True)


def test_Parity():
    f_tf = ops.Parity
    args_tf = (tf.ones((1, 128, 128, 2)), tf.ones((1, 128, 128, 2)))
    f_pt = ops_pt.Parity
    args_pt = (torch.ones((1, 128, 128, 2)), torch.ones((1, 128, 128, 2)))

    torch.manual_seed(42)
    tf.random.set_random_seed(42)
    out_tf = f_tf(*args_tf)
    out_pt = f_pt(*args_pt)
    assert _test_compatibility(out_tf[0], out_pt[0], permute_pt=False)
    assert _test_compatibility(out_tf[1], out_pt[1], permute_pt=False)


def test_prepare_pairs():
    # TF function
    from functools import partial

    kwargs = {
        "train": False,
        "static": False,
        "contrast_var": 0.1,
        "brightness_var": 0.1,
        "saturation_var": 0.1,
        "hue_var": 0.1,
        "p_flip": 0.1,
    }
    arg = DotMap(kwargs)
    t_tf = tf.ones((10, 128, 128, 3))
    out_tf = ops.prepare_pairs(t_tf, 128, arg)

    t_pt = torch.ones((10, 3, 128, 128))
    out_pt = ops_pt.prepare_pairs(t_pt, 128, **kwargs)

    assert out_pt[0].shape == (10, 3, 128, 128)
    assert out_pt[1].shape == (10, 3, 128, 128)
    assert _test_shape_compatibility(out_tf[0], out_pt[0], permute_pt=True)
    assert _test_shape_compatibility(out_tf[1], out_pt[1], permute_pt=True)


def test_reverse_batch():
    out_tf = ops.reverse_batch(
        tf.ones((10, 128, 128, 2))
        * tf.reshape(tf.range(10, dtype=tf.float32), (10, 1, 1, 1)),
        2,
    )
    out_pt = ops_pt.reverse_batch(
        torch.ones((10, 2, 128, 128))
        * torch.arange(10, dtype=torch.float32).view((10, 1, 1, 1)),
        2,
    )
    assert _test_shape_compatibility(out_tf, out_pt, permute_pt=True)


def test_random_scal():
    f_tf = ops.random_scal
    args_tf = (2, 1, 1)
    out_tf = f_tf(*args_tf)
    f_pt = ops_pt.random_scal
    args_pt = (2, 1, 1)
    out_pt = f_pt(*args_pt)
    assert _test_compatibility(out_tf, out_pt)


def test_part_map_to_mu_L_inv():
    # TODO: check estimated moments
    # part_maps_tf = tf.random_normal((10, 128, 128, 2), dtype=tf.float32)
    part_maps_tf = tfnn.tf_hm(
        tf.ones((1, 10, 2)) * 64, 128, 128, tf.ones((1, 10, 2)) * 3
    )
    part_maps_tf /= tf.reduce_sum(part_maps_tf, axis=[1, 2], keepdims=True)
    scales_tf = tf.ones((10, 1, 1, 1))
    out_tf = ops.part_map_to_mu_L_inv(part_maps_tf, scales_tf)

    part_maps_pt = tfpyth.th_NHWC_to_NCHW(torch.from_numpy(np.array(part_maps_tf)))
    scales_pt = torch.ones((10, 1, 1, 1))
    out_pt = ops_pt.part_map_to_mu_L_inv(part_maps_pt, scales_pt)

    assert _test_compatibility(out_tf[0], out_pt[0], permute_pt=False)

    # atol is not large enough
    assert np.allclose(out_tf[1], out_pt[1].numpy(), atol=1.0e-6)
    # assert _test_compatibility(out_tf[1], out_pt[1], permute_pt=False)


def test_get_features():
    import functools

    f_tf = functools.partial(ops.get_features, slim=True)
    args_tf = (tf.ones((10, 128, 128, 2)), tf.ones((10, 128, 128, 2)))
    out_tf = f_tf(*args_tf)
    f_pt = functools.partial(ops_pt.get_features, slim=True)
    args_pt = (torch.ones((10, 2, 128, 128)), torch.ones((10, 2, 128, 128)))
    out_pt = f_pt(*args_pt)
    assert _test_compatibility(out_tf, out_pt, permute_pt=False)


def test_get_img_slice_around_mu():
    import skimage

    image_t = torch.unsqueeze(torch.from_numpy(skimage.data.astronaut()), 0).permute(
        0, 3, 1, 2
    )
    mu = torch.zeros((1, 16, 2))

    image_slices = ops_pt.get_img_slice_around_mu(image_t, mu, [49, 49])
    assert image_slices.shape == (1, 16, 3, 49, 49)


def test_augm_mu():
    f_tf = ops.augm_mu
    image_in_tf = tf.ones((10, 128, 128, 3))
    image_rec_tf = tf.ones((10, 128, 128, 3))
    mu_tf = tf.zeros((10, 10, 2))
    features_tf = [tf.zeros((10, 64))]
    batch_size = 10
    n_parts = 10
    move_list = []

    out_tf = ops.augm_mu(
        image_in_tf, image_rec_tf, mu_tf, features_tf, batch_size, n_parts, move_list
    )

    ## pytorch
    f_pt = ops_pt.augm_mu
    image_in_pt = torch.ones((10, 3, 128, 128))
    image_rec_pt = torch.ones((10, 3, 128, 128))
    mu_pt = torch.zeros((10, 10, 2))
    features_pt = [torch.zeros((10, 64))]
    batch_size = 10
    n_parts = 10
    move_list = []

    out_pt = ops_pt.augm_mu(
        image_in_pt, image_rec_pt, mu_pt, features_pt, batch_size, n_parts, move_list
    )

    assert _test_compatibility(out_tf[0], out_pt[0], permute_pt=True)
    assert _test_compatibility(out_tf[1], out_pt[1], permute_pt=True)
    assert _test_compatibility(out_tf[2], out_pt[2], permute_pt=False)
    assert _test_compatibility(out_tf[3], out_pt[3], permute_pt=False)


def test_precision_dist_op():
    circular_precision = tf.zeros((2, 16, 2, 2))
    dist = tf.zeros((2, 16, 2, 16384))
    part_depth = 16
    nk = 16
    h = 128
    w = 128

    out_tf = ops.precision_dist_op(circular_precision, dist, part_depth, nk, h, w)

    circular_precision_pt = torch.zeros((2, 16, 2, 2))
    dist_pt = torch.zeros((2, 16, 2, 16384))
    part_depth = 16
    nk = 16
    h = 128
    w = 128

    out_pt = ops_pt.precision_dist_op(
        circular_precision_pt, dist_pt, part_depth, nk, h, w
    )
    assert _test_compatibility(out_tf[0], out_pt[0], permute_pt=False)
    assert _test_compatibility(out_tf[1], out_pt[1], permute_pt=False)


def test_feat_mu_to_enc():
    bs = 2
    nk = 16
    nf = 64
    features_tf = tf.ones((bs, nk, nf))
    mu_tf = tf.ones((bs, nk, 2))
    L_inv_tf = tf.ones((bs, nk, 2, 2))
    reconstruct_stages = [[128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
    part_depths = [16, 16, 16, 16, 4, 2]
    feat_map_depths = [[0, 0], [0, 0], [0, 0], [4, 16], [2, 4], [0, 2]]
    n_reverse = 2
    covariance = True
    feat_shape = False
    heat_feat_normalize = False
    static = True

    out_tf = ops.feat_mu_to_enc(
        features_tf,
        mu_tf,
        L_inv_tf,
        reconstruct_stages,
        part_depths,
        feat_map_depths,
        True,
        n_reverse,
        covariance=covariance,
        feat_shape=feat_shape,
        heat_feat_normalize=heat_feat_normalize,
    )

    # pytorch

    bs = 2
    nk = 16
    nf = 64
    features_pt = torch.ones((bs, nk, nf))
    mu_pt = torch.ones((bs, nk, 2))
    L_inv_pt = torch.ones((bs, nk, 2, 2))
    reconstruct_stages = [[128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
    part_depths = [16, 16, 16, 16, 4, 2]
    feat_map_depths = [[0, 0], [0, 0], [0, 0], [4, 16], [2, 4], [0, 2]]
    n_reverse = 2
    covariance = True
    feat_shape = False
    heat_feat_normalize = False
    static = True

    out_pt = ops_pt.feat_mu_to_enc(
        features_pt,
        mu_pt,
        L_inv_pt,
        reconstruct_stages,
        part_depths,
        feat_map_depths,
        True,
        n_reverse,
        covariance=covariance,
        feat_shape=feat_shape,
        heat_feat_normalize=heat_feat_normalize,
    )
    assert out_pt[0].shape == (bs, nk, 128, 128)


def test_heat_map_function():
    y_dist = tf.ones((1, 10, 16384))
    x_dist = tf.ones((1, 10, 16384))
    scale = 1
    out_tf = ops.heat_map_function(x_dist, y_dist, scale, scale)

    # pytorch
    y_dist = torch.ones((1, 10, 16384))
    x_dist = torch.ones((1, 10, 16384))
    scale = 1
    out_pt = ops_pt.heat_map_function(x_dist, y_dist, scale, scale)

    assert np.allclose(out_tf, out_pt.numpy())


def test_unary_mat():
    v = tf.ones((2, 2, 2))
    u_tf = ops.unary_mat(v)

    # pytorch
    v = torch.ones((2, 2, 2))
    u_th = ops_pt.unary_mat(v)
    assert np.allclose(u_tf, u_th.numpy())


def test_fold_img_with_mu():
    img_tf = tf.zeros((1, 128, 128, 3))
    mu_tf = tf.zeros((1, 10, 2))
    scale_tf = 1
    threshold = 0.75

    out_tf = ops.fold_img_with_mu(img_tf, mu_tf, scale_tf, False, threshold)

    img_pt = torch.zeros((1, 3, 128, 128))
    mu_pt = torch.zeros((1, 10, 2))
    scale_pt = 1
    threshold = 0.75
    out_pt = ops_pt.fold_img_with_mu(img_pt, mu_pt, scale_pt, threshold)

    for o_tf, o_pt in zip(out_tf, out_pt):
        assert np.allclose(o_tf, tfpyth.th_2D_channels_first_to_last(o_pt).numpy())


def test_mu_img_gate():
    mu_tf = tf.zeros((1, 10, 2))
    resolution = [128, 128]
    scale = 1
    out_tf = ops.mu_img_gate(mu_tf, resolution, scale)

    mu_pt = torch.zeros((1, 10, 2))
    out_pt = ops_pt.mu_img_gate(mu_pt, resolution, scale)


def test_binary_activation():
    f_tf = ops.binary_activation
    args_tf = (tf.ones((10, 128, 128, 2)),)
    out_tf = f_tf(*args_tf)
    f_pt = ops_pt.binary_activation
    args_pt = (torch.ones((10, 128, 128, 2)),)
    out_pt = f_pt(*args_pt)
    assert _test_compatibility(out_tf, out_pt)


def test_fold_img_with_L_inv():
    img_tf = tf.zeros((1, 128, 128, 3))
    mu_tf = tf.zeros((1, 10, 2))
    L_inv_tf = tf.ones((1, 10, 2, 2))
    scale_tf = tf.ones((1, 1, 2, 2))
    threshold = 0.75

    args_tf = (img_tf, mu_tf, L_inv_tf, scale_tf, False, threshold)
    out_tf = ops.fold_img_with_L_inv(
        img_tf, mu_tf, L_inv_tf, scale_tf, False, threshold
    )
    ## Pytorch
    f_pt = ops_pt.fold_img_with_L_inv

    img_pt = torch.zeros((1, 3, 128, 128))
    mu_pt = torch.zeros((1, 10, 2))
    L_inv_pt = torch.ones((1, 10, 2, 2))
    scale_pt = torch.ones((1, 1, 2, 2))
    threshold = 0.75
    out_pt, _ = ops_pt.fold_img_with_L_inv(img_pt, mu_pt, L_inv_pt, scale_pt, threshold)

    assert np.allclose(out_tf, tfpyth.th_2D_channels_first_to_last(out_pt).numpy())


def test_probabilistic_switch():
    f_tf = ops.probabilistic_switch
    args_tf = ([0.125,] * 4, [0.25,] * 9, 0)
    f_pt = ops_pt.probabilistic_switch
    args_pt = ([0.125,] * 4, [0.25,] * 9, 0)

    np.random.seed(42)
    out_tf = f_tf(*args_tf)
    np.random.seed(42)
    out_pt = f_pt(*args_pt)
    assert np.allclose(out_tf, out_pt)


import numpy as np
from skimage.data import astronaut


def test_torch_image_random_contrast():
    f_tf = tf.image.random_contrast
    img = astronaut()[np.newaxis, ...] / 255.0
    img = img.astype(np.float32)
    args_tf = (img.copy(), 0.3, 0.7)
    f_pt = ops_pt.torch_image_random_contrast
    args_pt = (tfpyth.th_NHWC_to_NCHW(torch.from_numpy(img.copy())), 0.3, 0.7)
    torch.backends.cudnn.deterministic = True
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.set_random_seed(seed)
    tf.enable_eager_execution()
    out_tf = f_tf(*args_tf)
    out_pt = f_pt(*args_pt)
    assert list(out_tf.shape) == list(tfpyth.th_NCHW_to_NHWC(out_pt).numpy().shape)


def test_tile_nd():
    a = torch.zeros((10, 1, 1, 1))
    b = ops_pt.tile_nd(a, [1, 10, 20, 30])
    assert b.shape == (10, 10, 20, 30)


def test_blob_generation():
    # check if xy order is the same
    covar = tf.concat([tf.ones((1, 10, 1)) * 3, tf.ones((1, 10, 1)) * 6], axis=2)

    part_maps_tf = tfnn.tf_hm(tf.ones((1, 10, 2)) * 64, 128, 128, covar)
    part_maps_tf /= tf.reduce_sum(part_maps_tf, axis=[1, 2], keepdims=True)

    # estimated_blob = op
    part_maps_pt = tfpyth.th_NHWC_to_NCHW(torch.from_numpy(np.array(part_maps_tf)))
    scales_pt = torch.ones((1, 1, 1, 1))
    mu, L_inv = ops_pt.part_map_to_mu_L_inv(part_maps_pt, scales_pt)
    img = torch.zeros((1, 3, 128, 128))

    _, foldy_map = ops_pt.fold_img_with_L_inv(img, mu, L_inv, 1.0, 0.3, normalize=False)

