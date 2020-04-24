import numpy as np


import torch
import torch.functional as F
import tensorflow as tf

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import tfpyth
import ops
import ops_pt


def AbsDetJacobian(batch_meshgrid):
    return tfpyth.wrap_torch_from_tensorflow(ops.AbsDetJacobian)(batch_meshgrid)


# return f(batch_meshgrid)
""" batch_meshgrid in shape [bn, 2, h, w] """

# b = channel_first_to_last(batch_meshgrid)
# b_ph = tf.compat.v1.placeholder(tf.float32, name="batch_meshgrid_ph")
# det = ops.AbsDetJacobian(b)
# f = tfpyth.torch_from_tensorflow(tf.compat.v1.Session(), [b_ph], det).apply
# out = f(b)
# return out

from torchvision.transforms import functional as TF


def torch_image_random_contrast(image, lower, upper, seed=None):
    contrast_factor = torch.distributions.uniform.Uniform(lower, upper).sample()
    return torch_image_adjust_contrast(image, contrast_factor)


def torch_image_adjust_contrast(image, contrast_factor):
    t_out = []
    for img in image:
        img_pil = TF.to_pil_image(img)
        img_transformed = TF.adjust_contrast(img_pil, contrast_factor)
        img_transformed = TF.to_tensor(img_transformed)
        t_out.append(img_transformed)
    t_out = torch.stack(t_out, axis=0)
    return t_out


def torch_image_random_brightness(image, max_delta, seed=None):
    delta = torch.distributions.uniform.Uniform(-max_delta, max_delta).sample()
    return torch_image_adjust_brightness(image, delta)


def torch_image_adjust_brightness(image, delta):
    t_out = []
    for img in image:
        img_pil = TF.to_pil_image(img)
        img_transformed = TF.adjust_brightness(img_pil, delta)
        img_transformed = TF.to_tensor(img_transformed)
        t_out.append(img_transformed)
    t_out = torch.stack(t_out, axis=0)
    return t_out


def torch_image_random_saturation(image, lower, upper, seed=None):
    saturation_factor = torch.distributions.uniform.Uniform(lower, upper).sample()
    return torch_image_adjust_saturation(image, saturation_factor)


def torch_image_adjust_saturation(image, saturation_factor):
    t_out = []
    for img in image:
        img_pil = TF.to_pil_image(img)
        img_transformed = TF.adjust_saturation(img_pil, saturation_factor)
        img_transformed = TF.to_tensor(img_transformed)
        t_out.append(img_transformed)
    t_out = torch.stack(t_out, axis=0)
    return t_out


def torch_image_random_hue(image, max_delta, seed=None):
    delta = torch.distributions.uniform.Uniform(-max_delta, max_delta).sample()
    return torch_image_adjust_saturation(image, delta)


def torch_image_adjust_hue(image, delta):
    t_out = []
    for img in image:
        img_pil = TF.to_pil_image(img)
        img_transformed = TF.adjust_hue(img_pil, delta)
        img_transformed = TF.to_tensor(img_transformed)
        t_out.append(img_transformed)
    t_out = torch.stack(t_out, axis=0)
    return t_out


def augm(t, contrast_var, brightness_var, saturation_var, hue_var, p_flip):
    t = torch_image_random_contrast(t, 1 - contrast_var, 1 + contrast_var)
    t = torch_image_random_brightness(t, brightness_var)
    t = torch_image_random_saturation(t, 1 - saturation_var, 1 + saturation_var)
    t = torch_image_random_hue(t, hue_var)

    random_tensor = 1 - p_flip + torch.empty([1], dtype=t.dtype).uniform_()
    binary_tensor = torch.floor(random_tensor)
    augmented = binary_tensor * t + (1 - binary_tensor) * (1 - t)
    return augmented


def torch_random_uniform(shape, lower, upper, dtype):
    t = torch.empty(list(shape), dtype=dtype).uniform_(lower, upper)
    return t


def torch_astype(x, dtype):
    return x.type(dtype)


def torch_stop_gradient(x):
    return x.detach()


def Parity(t_images, t_mesh, on=False):
    if on:
        bn = t_images.shape[0]
        P = torch_random_uniform(shape=[bn, 1, 1, 1], dtype=tf.float32) - 0.5
        P = torch_astype(P > 0.0, torch.float32)
        Pt_images = P * t_images[:, :, ::-1] + (1 - P) * t_images
        Pt_mesh = P * t_mesh[:, :, ::-1] + (1 - P) * t_mesh
    else:
        Pt_images = t_images
        Pt_mesh = t_mesh

    return Pt_images, Pt_mesh


def prepare_pairs(t_images, reconstr_dim, train, static):
    if train:
        bn, h, w, n_c = t_images.get_shape().as_list()
        if static:
            t_images = tf.concat(
                [
                    tf.expand_dims(t_images[: bn // 2], axis=1),
                    tf.expand_dims(t_images[bn // 2 :], axis=1),
                ],
                axis=1,
            )
        else:
            t_images = tf.reshape(t_images, shape=[bn // 2, 2, h, w, n_c])
        t_c_1_images = tf.map_fn(lambda x: ops.augm(x, arg), t_images)
        t_c_2_images = tf.map_fn(lambda x: ops.augm(x, arg), t_images)
        a, b = (
            tf.expand_dims(t_c_1_images[:, 0], axis=1),
            tf.expand_dims(t_c_1_images[:, 1], axis=1),
        )
        c, d = (
            tf.expand_dims(t_c_2_images[:, 0], axis=1),
            tf.expand_dims(t_c_2_images[:, 1], axis=1),
        )
        if static:
            t_input_images = tf.reshape(
                tf.concat([a, d], axis=0), shape=[bn, h, w, n_c]
            )
            t_reconstr_images = tf.reshape(
                tf.concat([c, b], axis=0), shape=[bn, h, w, n_c]
            )
        else:
            t_input_images = tf.reshape(
                tf.concat([a, d], axis=1), shape=[bn, h, w, n_c]
            )
            t_reconstr_images = tf.reshape(
                tf.concat([c, b], axis=1), shape=[bn, h, w, n_c]
            )

        t_input_images = tf.clip_by_value(t_input_images, 0.0, 1.0)
        t_reconstr_images = tf.image.resize_images(
            tf.clip_by_value(t_reconstr_images, 0.0, 1.0),
            size=(reconstr_dim, reconstr_dim),
        )

    else:
        t_input_images = tf.clip_by_value(t_images, 0.0, 1.0)
        t_reconstr_images = tf.image.resize_images(
            tf.clip_by_value(t_images, 0.0, 1.0), size=(reconstr_dim, reconstr_dim)
        )

    return t_input_images, t_reconstr_images


def torch_reshape(x, shape):
    return x.view(shape)


def reverse_batch(tensor, n_reverse):
    bn, *rest = list(tensor.shape)
    assert (bn / n_reverse).is_integer()
    tensor = torch_reshape(tensor, shape=[bn // n_reverse, n_reverse, *rest])
    idx = [i for i in range(n_reverse - 1, -1, -1)]
    idx = torch.LongTensor(idx)
    # tensor_rev = tensor[:, ::-1]
    tensor_rev = tensor.index_select(1, idx)
    tensor_rev = torch_reshape(tensor_rev, shape=[bn, *rest])
    return tensor_rev


def random_scal(bn, min_scal, max_scal):
    rand_scal = torch.zeros((bn // 2, 2), dtype=torch.float32).uniform_(
        min_scal, max_scal
    )
    rand_scal = tile(rand_scal, 2, 0)
    rand_scal = tile(rand_scal, 2, 1)
    rand_scal = rand_scal.view((2 * bn, 2))
    return rand_scal


def part_map_to_mu_L_inv(part_maps, scal):
    """
    Calculate mean for each channel of part_maps
    :param part_maps: tensor of part map activations [bn, h, w, n_part]
    :return: mean calculated on a grid of scale [-1, 1]
    """
    bn, nk, h, w = list(part_maps.shape)
    y_t = tile(torch.linspace(-1.0, 1.0, h).view([h, 1]), w, dim=1)
    x_t = tile(torch.linspace(-1.0, 1.0, w).view([1, w]), h, dim=0)
    y_t = torch.unsqueeze(y_t, dim=-1)
    x_t = torch.unsqueeze(x_t, dim=-1)
    meshgrid = torch.cat([y_t, x_t], dim=-1)

    mu = torch.einsum("ijl,akij->akl", meshgrid, part_maps)
    mu_out_prod = torch.einsum("akm,akn->akmn", mu, mu)

    mesh_out_prod = torch.einsum("ijm,ijn->ijmn", meshgrid, meshgrid)
    stddev = torch.einsum("ijmn,akij->akmn", mesh_out_prod, part_maps) - mu_out_prod

    a_sq = stddev[:, :, 0, 0]
    a_b = stddev[:, :, 0, 1]
    b_sq_add_c_sq = stddev[:, :, 1, 1]
    eps = 1e-12

    a = torch.sqrt(
        a_sq + eps
    )  # Σ = L L^T Prec = Σ^-1  = L^T^-1 * L^-1  ->looking for L^-1 but first L = [[a, 0], [b, c]
    b = a_b / (a + eps)
    c = torch.sqrt(b_sq_add_c_sq - b ** 2 + eps)
    z = torch.zeros_like(a)

    det = torch.unsqueeze(torch.unsqueeze(a * c, dim=-1), dim=-1)
    row_1 = torch.unsqueeze(
        torch.cat([torch.unsqueeze(c, dim=-1), torch.unsqueeze(z, dim=-1)], dim=-1),
        dim=-2,
    )
    row_2 = torch.unsqueeze(
        torch.cat([torch.unsqueeze(-b, dim=-1), torch.unsqueeze(a, dim=-1)], dim=-1),
        dim=-2,
    )

    L_inv = (
        scal / (det + eps) * torch.cat([row_1, row_2], dim=-2)
    )  # L^⁻1 = 1/(ac)* [[c, 0], [-b, a]
    return mu, L_inv


def tile(a, n_tile, dim):
    """Equivalent of numpy or tensorflow tile.
    
    References
    ----------
    [1] : https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/4
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(a, dim, order_index)


def tile_nd(a, n_tile_nd):
    for dim, nn_tile in enumerate(n_tile_nd):
        a = tile(a, nn_tile, dim)
    return a


def get_features(features, part_map, slim):
    if slim:
        features = torch.einsum("bfij,bkij->bkf", features, part_map)
    else:
        features = torch.einsum("bkij,bkij->bkf", features, part_map)
    return features


def augm_mu(image_in, image_rec, mu, features, batch_size, n_parts, move_list):
    image_in = tile_nd(torch.unsqueeze(image_in[0], dim=0), [batch_size, 1, 1, 1])
    image_rec = tile_nd(torch.unsqueeze(image_rec[0], dim=0), [batch_size, 1, 1, 1])
    mu = tile_nd(torch.unsqueeze(mu[0], dim=0), [batch_size, 1, 1])
    features = tile_nd(torch.unsqueeze(features[0], dim=0), [batch_size, 1, 1])
    batch_size = batch_size // 2
    ran = torch.arange(batch_size).view([batch_size, 1]) / batch_size - 0.5
    array = torch.cat(
        [
            torch.cat([ran, torch.zeros_like(ran)], dim=-1),
            torch.cat([torch.zeros_like(ran), ran], dim=1),
        ]
    )
    array = torch.unsqueeze(torch_astype(array, dtype=torch.float32), dim=1)
    for elem in move_list:
        part = torch.constant(elem, dtype=torch.int32, shape=([1, 1]))
        pad_part = torch.constant([[1, 1], [0, 0]])
        part_arr = torch.pad(torch.concat([part, -part]), pad_part, dim=-1)
        pad = torch.constant([[0, 0], [0, n_part - 1], [0, 0]]) + part_arr
        addy = torch.pad(array, pad)
        mu = mu + addy
    return image_in, image_rec, mu, features


def precision_dist_op(precision, dist, part_depth, nk, h, w):
    proj_precision = torch.einsum("bnik,bnkf->bnif", precision, dist) ** 2
    proj_precision = torch.sum(proj_precision, dim=-2)

    heat = 1 / (1 + proj_precision)
    heat = heat.view([-1, nk, h, w])
    part_heat = heat[:, :part_depth]
    part_heat = part_heat.permute([0, 2, 3, 1])
    return heat, part_heat


def feat_mu_to_enc(
    features,
    mu,
    L_inv,
    reconstruct_stages,
    part_depths,
    feat_map_depths,
    static,
    n_reverse,
    covariance=None,
    feat_shape=None,
    heat_feat_normalize=True,
    range_=10,
    # bs=2,
    # n_features=64,
    # n_keypoints=64
):
    # features_ph = tf.compat.v1.placeholder(tf.float32, name="features_ph")
    # mu_ph = tf.compat.v1.placeholder(tf.float32, name="mu_ph")
    # L_inv_ph = tf.compat.v1.placeholder(tf.float32, name="L_inv_ph")

    # encoding_list = ops.feat_mu_to_enc(
    #     features_ph,
    #     mu_ph,
    #     L_inv_ph,
    #     reconstruct_stages,
    #     part_depths,
    #     feat_map_depths,
    #     static,
    #     n_reverse,
    #     covariance,
    #     feat_shape,
    #     heat_feat_normalize,
    #     range=range_,
    # )
    # f = tfpyth.torch_from_tensorflow(
    #     tf.compat.v1.Session(), [features_ph, mu_ph, L_inv_ph], encoding_list
    # ).apply

    # TODO: define input
    # def func(features, mu, L_inv):
    #     out = ops.feat_mu_to_enc(features, mu, L_inv, reconstruct_stages, part_depths, feat_map_depths, static, n_reverse, covariance=covariance, feat_shape=feat_shape, heat_feat_normalize=heat_feat_normalize)
    #     return out
    # f = tfpyth.wrap_torch_from_tensorflow(func, ["features", "mu", "L_inv"], [(bs, n_keypoints, n_features), (bs, n_keypoints, 2), (bs, n_keypoints, 2, 2)])
    # return f(features, mu, L_inv)
    bn, nk, nf = list(features.shape)
    if static:
        reverse_features = torch.cat([features[bn // 2 :], features[: bn // 2]], dim=0)

    else:
        reverse_features = reverse_batch(features, n_reverse)

    encoding_list = []
    circular_precision = tile_nd(
        torch_reshape(
            torch.tensor([[range_, 0.0], [0, range_]], dtype=torch.float32),
            shape=[1, 1, 2, 2],
        ),
        [bn, nk, 1, 1],
    )
    for dims, part_depth, feat_slice in zip(reconstruct_stages, part_depths, feat_map_depths):
        h, w = dims[0], dims[1]

        y_t = torch.unsqueeze(
            tile_nd(torch_reshape(torch.linspace(-1.0, 1.0, h), [h, 1]), [1, w]), dim=-1
        )
        x_t = torch.unsqueeze(
            tile_nd(torch_reshape(torch.linspace(-1.0, 1.0, w), [1, w]), [h, 1]), dim=-1
        )

        y_t_flat = torch_reshape(y_t, (1, 1, 1, -1))
        x_t_flat = torch_reshape(x_t, (1, 1, 1, -1))

        mesh = torch.cat([y_t_flat, x_t_flat], dim=-2)
        dist = mesh - torch.unsqueeze(mu, dim=-1)

        if not covariance or not feat_shape:
            heat_circ, part_heat_circ = precision_dist_op(
                circular_precision, dist, part_depth, nk, h, w
            )

        if covariance or feat_shape:
            heat_shape, part_heat_shape = precision_dist_op(
                L_inv, dist, part_depth, nk, h, w
            )

        nkf = feat_slice[1] - feat_slice[0]

        if nkf != 0:
            feature_slice_rev = reverse_features[:, feat_slice[0] : feat_slice[1]]

            if feat_shape:
                heat_scal = heat_shape[:, feat_slice[0] : feat_slice[1]]

            else:
                heat_scal = heat_circ[:, feat_slice[0] : feat_slice[1]]

            if heat_feat_normalize:
                heat_scal_norm = torch.sum(heat_scal, dim=1, keepdims=True) + 1
                heat_scal = heat_scal / heat_scal_norm

            heat_feat_map = torch.einsum("bkij,bkn->bijn", heat_scal, feature_slice_rev)

            if covariance:
                encoding_list.append(
                    torch.cat([part_heat_shape, heat_feat_map], dim=-1)
                )

            else:
                encoding_list.append(
                    torch.cat([part_heat_circ, heat_feat_map], dim=-1)
                )

        else:
            if covariance:
                encoding_list.append(part_heat_shape)

            else:
                encoding_list.append(part_heat_circ)

    return encoding_list



def heat_map_function(y_dist, x_dist, y_scale, x_scale):
    # TODO: write test
    term_1 = (y_dist / (1e-6 + y_scale)) ** 2
    term_2 = (x_dist / (1e-6 + x_scale)) ** 2
    x = 1 / (term_1 + term_2 + 1)
    return x


def unary_mat(vector):
    # TODO: write test. Einsum probably needs adjustment
    b_1 = torch.unsqueeze(vector, dim=-2)
    reverse_vector = flip(vector, -1)
    b_2 = torch.unsqueeze(
        torch.einsum(
            "bkc,c->bkc",
            reverse_vector,
            torch.tensor(np.array([-1.0, 1]), dtype=torch.float32),
        ),
        dim=-2,
    )
    U_mat = torch.cat([b_1, b_2], dim=-2)
    return U_mat


def get_img_slice_around_mu(img, mu, slice_size, h, w, nk):
    # TODO: rewrite this in pytorch
    # import functools

    # def func(img, mu):
    #     return ops.get_img_slice_around_mu(img, mu, slice_size)

    # func = tfpyth.wrap_torch_from_tensorflow(
    #     func, ["img", "mu"], [(None, h, w, 3), (None, nk, 2)]
    # )
    # return func(img, mu)
    img = tfpyth.th_2D_channels_first_to_last(img)

    h, w, = slice_size
    bn, img_h, img_w, c = img.shape
    bn_2, nk, _ = mu.shape
    assert int(h / 2)
    assert int(w / 2)
    assert bn_2 == bn

    scal = torch.from_numpy(np.array([img_h, img_w], dtype=np.float32))
    mu = mu.detach()
    mu_no_grad = torch.einsum("bkj,j->bkj", (mu + 1) / 2.0, scal)
    mu_no_grad = torch_astype(mu_no_grad, torch.int32)
    mu_no_grad = torch_reshape(mu_no_grad, [-1, nk, 1, 1, 2])
    y = tile_nd(
        torch_reshape(torch.arange(-h // 2, h // 2), [1, 1, h, 1, 1]), [bn, nk, 1, w, 1]
    )
    x = tile_nd(
        torch_reshape(torch.arange(-w // 2, w // 2), [1, 1, 1, w, 1]), [bn, nk, h, 1, 1]
    )

    field = torch.cat([y, x], dim=-1) + mu_no_grad
    h1 = tile_nd(torch_reshape(torch.arange(bn), (bn, 1, 1, 1, 1)), [1, nk, h, w, 1])
    idx = torch.cat([h1, field], dim=-1)
    idx = torch_astype(idx, torch.int32)

    image_slices = gather_nd(img, idx, img.shape, idx.shape)
    image_slices = image_slices.permute(
        (0, 1, 4, 2, 3)
    )  # [N, nk, H, W, C] -> [N, nk, C, H, W]
    return image_slices


def gather_nd(params, indices, params_shape, indices_shape):
    """dirty wrapper for tf.gather_nd to use with pytorch"""

    def func(params, indices):
        return tf.gather_nd(params, indices)

    out = tfpyth.wrap_torch_from_tensorflow(
        func,
        ["params", "indices"],
        input_shapes=[params_shape, indices_shape],
        input_dtypes=[tf.float32, tf.int32],
    )(params, indices)
    return out


def fold_img_with_mu(img, mu, scale, threshold, normalize=True):
    # TODO: rewrite this in pytorch
    # pass
    bn, nc, h, w = list(img.shape)
    bn, nk, _ = list(mu.shape)
    py = torch.unsqueeze(mu[:, :, 0], 2).detach()
    px = torch.unsqueeze(mu[:, :, 1], 2).detach()

    y_t = tile_nd(torch_reshape(torch.linspace(-1, 1, h), [h, 1]), [1, w])
    x_t = tile_nd(torch_reshape(torch.linspace(-1, 1, w), [1, w]), [h, 1])

    x_t_flat = torch_reshape(x_t, (1, 1, -1))
    y_t_flat = torch_reshape(y_t, (1, 1, -1))

    y_dist = py - y_t_flat
    x_dist = px - x_t_flat

    heat_scal = heat_map_function(
        y_dist=y_dist, x_dist=x_dist, x_scale=scale, y_scale=scale
    )
    heat_scal = torch_reshape(heat_scal, [bn, nk, h, w])
    heat_scal = torch.einsum("bkij->bij", heat_scal)
    heat_scal = heat_scal.clamp(0.0, 1.0)

    heat_scal = torch.where(
        heat_scal > threshold,
        heat_scal,
        torch.zeros_like(heat_scal, dtype=heat_scal.dtype),
    )

    norm = torch.sum(heat_scal, dim=(1, 2), keepdims=True)
    if normalize:
        heat_scal_norm = heat_scal / norm
        folded_img = torch.einsum("bcij,bij->bcij", img, heat_scal_norm)
    else:
        folded_img = torch.einsum("bcij,bij->bcij", img, heat_scal)

    return folded_img, torch.unsqueeze(heat_scal, axis=1)


def mu_img_gate(mu, resolution, scale):
    # TODO: rewrite this in pytorch
    # pass
    bn, nk, _ = list(mu.shape)
    py = torch.unsqueeze(mu[:, :, 0], 2).detach()
    px = torch.unsqueeze(mu[:, :, 1], 2).detach()

    h, w = resolution
    y_t = tile_nd(torch_reshape(torch.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tile_nd(torch_reshape(torch.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    x_t_flat = torch_reshape(x_t, (1, 1, -1))
    y_t_flat = torch.reshape(y_t, (1, 1, -1))

    y_dist = py - y_t_flat
    x_dist = py - x_t_flat

    heat_scal = heat_map_function(
        y_dist=y_dist, x_dist=x_dist, x_scale=scale, y_scale=scale
    )
    heat_scal = torch_reshape(heat_scal, shape=[bn, nk, h, w])
    heat_scal = torch.einsum("bkij->bij", heat_scal)
    return heat_scal


def binary_activation(x):
    # TODO: rewrite this in pytorch
    cond = x < torch.zeros_like(x)
    out = torch.where(cond, torch.zeros_like(x), torch.ones_like(x))
    return out


def fold_img_with_L_inv(img, mu, L_inv, scale, threshold, normalize=True):
    bn, nc, h, w = list(img.shape)
    bn, nk, _ = list(mu.shape)

    mu_stop = mu.detach()

    y_t = tile_nd(torch_reshape(torch.linspace(-1.0, 1.0, h), [h, 1]), [1, w])
    x_t = tile_nd(torch_reshape(torch.linspace(-1.0, 1.0, w), [1, w]), [h, 1])
    x_t_flat = torch_reshape(x_t, (1, 1, -1))
    y_t_flat = torch_reshape(y_t, (1, 1, -1))

    mesh = torch.cat([y_t_flat, x_t_flat], dim=-2)
    dist = mesh - torch.unsqueeze(mu_stop, dim=-1)

    proj_precision = torch.einsum("bnik,bnkf->bnif", scale * L_inv, dist) ** 2
    proj_precision = torch.sum(proj_precision, dim=-2)

    heat = 1 / (1 + proj_precision)

    heat = torch.reshape(heat, shape=[bn, nk, h, w])
    heat = torch.einsum("bkij->bij", heat)
    heat_scal = heat.clamp(0.0, 1.0)
    heat_scal = torch.where(
        heat_scal > threshold, heat_scal, torch.zeros_like(heat_scal)
    )
    norm = torch.sum(
        heat_scal, dim=[1, 2], keepdims=True
    )  # sum over spatial dimensions
    if normalize:
        heat_scal = heat_scal / norm
    folded_img = torch.einsum("bcij,bij->bcij", img, heat_scal)
    return folded_img


def probabilistic_switch(handles, handle_probs, counter, scale=10000.0):
    t = counter / scale
    scheduled_probs = []
    for p_1, p_2 in zip(handle_probs[::2], handle_probs[1::2]):
        scheduled_prob = t * p_1 + (1 - t) * p_2
        scheduled_probs.append(scheduled_prob)

    handle = np.random.choice(handles, p=scheduled_probs)
    return handle


def spatial_softmax(logit_map):
    eps = 1.0e-12
    m, _ = torch.max(logit_map, dim=3, keepdim=True)
    m, _ = torch.max(m, dim=2, keepdim=True)
    exp = torch.exp(logit_map - m)
    norm = torch.sum(exp, dim=[1, 2], keepdims=True) + eps
    sm = exp / norm
    return sm


def flip(x, dim):
    """Reverse tensor order aling given dimension.
    Drop-in replacement for tf.reverse(x, axis)
    
    Parameters
    ----------
    x : torch.Tensor
        tensor to flip
    dim : int
        dimension along which to flip
    
    Returns
    -------
    torch.Tensor
        flipped tensor

    References
    ----------

    [1] : https://github.com/pytorch/pytorch/issues/229
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(
        x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device
    )
    return x[tuple(indices)]
