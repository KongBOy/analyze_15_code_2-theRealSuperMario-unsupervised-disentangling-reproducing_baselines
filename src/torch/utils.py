import torch
from matplotlib import cm
import numpy as np


def batch_colour_map(heat_map):
    c = list(heat_map.shape)[1]
    color = []
    for i in range(c):
        color.append(cm.hsv(float(i / c))[:3])
    color = torch.tensor(color).to(heat_map.device)
    color_map = torch.einsum("bkij,kl->blij", heat_map, color)
    return color_map


def np_batch_colour_map(heat_map):
    c = heat_map.shape[-1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])
    np_colour = np.array(colour)
    colour_map = np.einsum("bkij,kl->blij", heat_map, np_colour)
    return colour_map


def part_to_color_map(
    encoding_list, part_depths, size, square=True,
):
    part_maps = encoding_list[0][:, : part_depths[0], :, :]
    if square:
        part_maps = part_maps ** 4
    color_part_map = batch_colour_map(part_maps)
    color_part_map = torch.nn.functional.interpolate(color_part_map, size=(size, size))

    return color_part_map


def summary_feat_and_parts(
    encoding_list, part_depths, visualize_features=False, square=True
):
    part_outputs = []
    feat_outputs = []
    for n, enc in enumerate(encoding_list):
        part_maps, feat_maps = (
            enc[:, : part_depths[n], :, :],
            enc[:, part_depths[n] :, :, :],
        )
        if square:
            part_maps = part_maps ** 2
        color_part_map = batch_colour_map(part_maps)
        part_outputs.append(color_part_map)
        # with tf.variable_scope("parts"):
        #     tf.summary.image(
        #         name="parts" + str(n), tensor=color_part_map, max_outputs=4
        #     )

        if visualize_features:
            if list(feat_maps.shape)[1] > 0:
                if square:
                    feat_maps = feat_maps ** 2
                color_feat_map = batch_colour_map(
                    feat_maps / torch.sum(feat_maps, dim=[2, 3], keepdims=True)
                )
                feat_outputs.append(color_feat_map ** 2)
            # if feat_maps.get_shape().as_list()[-1] > 0:
            #     with tf.variable_scope("feature_maps"):
            #         if square:
            #             feat_maps = feat_maps ** 2
            #         color_feat_map = batch_colour_map(
            #             feat_maps / tf.reduce_sum(feat_maps, axis=[1, 2], keepdims=True)
            #         )
            #         tf.summary.image(
            #             name="feat_maps" + str(n),
            #             tensor=color_feat_map ** 2,
            #             max_outputs=4,
            #         )
    return part_outputs, feat_outputs
