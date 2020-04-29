import torch

from src.torch.ops_pt import (
    get_features,
    part_map_to_mu_L_inv,
    feat_mu_to_enc,
    prepare_pairs,
    AbsDetJacobian,
    get_img_slice_around_mu,
)

from supermariopy.ptutils.tps import ThinPlateSpline, make_input_tps_param
import tfpyth

# from torchvision.transforms import functional as F
from torch.nn import functional as F
from src.torch import architectures_pt, ops_pt, utils
from dotmap import DotMap


class Model(torch.nn.Module):
    def __init__(
        self,
        heat_dim,
        nFeat_1,
        nFeat_2,
        L_inv_scal,
        rec_stages,
        part_depths,
        feat_slices,
        covariance,
        average_features_mode,
        heat_feat_normalize,
        reconstr_dim,
        n_c,
        bn,
        n_parts,
        adversarial,
        patch_size,
        **kwargs
    ):
        super(Model, self).__init__()
        self.nFeat_1 = nFeat_1
        self.nFeat_2 = nFeat_2
        self.n_features = 64
        self.encoder = architectures_pt.SeperateHourglass_128(
            3, 16, self.n_features, nFeat_1, nFeat_2
        )
        self.img_decoder = architectures_pt.Decoder128([66, 68, 80, 16, 16, 16], n_c)
        self.heat_dim = heat_dim
        self.L_inv_scal = L_inv_scal
        self.rec_stages = rec_stages
        self.part_depths = part_depths
        self.feat_slices = feat_slices
        self.covariance = covariance
        self.average_features_mode = average_features_mode
        self.heat_feat_normalize = heat_feat_normalize
        self.reconstr_dim = reconstr_dim
        self.n_c = n_c
        self.bn = bn
        self.n_parts = n_parts
        self.adversarial = adversarial
        self.patch_size = patch_size
        if self.adversarial:
            self.discriminator = architectures_pt.Discriminator_Patch()

    def forward(self, image_in, image_rec=None, static=False):
        part_maps, raw_features = self.encoder(image_in)

        mu, L_inv = part_map_to_mu_L_inv(part_maps, self.L_inv_scal)
        features = get_features(raw_features, part_maps, True)

        encoding_same_id = feat_mu_to_enc(
            features,
            mu,
            L_inv,
            self.rec_stages,
            self.part_depths,
            self.feat_slices,
            n_reverse=2,
            covariance=self.covariance,
            feat_shape=self.average_features_mode,
            heat_feat_normalize=self.heat_feat_normalize,
            static=static,
        )
        encoding_same_id = [
            tfpyth.th_2D_channels_last_to_first(e) for e in encoding_same_id
        ]

        reconstruct_same_id = self.img_decoder(encoding_same_id)

        outputs = {
            "part_maps": part_maps,
            "mu": mu,
            "L_inv": L_inv,
            "features": features,
            "reconstruct_same_id": reconstruct_same_id,
            "encoding_same_id": encoding_same_id,
        }

        if self.adversarial:
            # TODO: test this
            flatten_dim = 2 * self.bn * self.n_parts
            part_map_last_layer = encoding_same_id[0][:, : self.part_depths[0], :, :]
            real_patches = get_img_slice_around_mu(
                torch.cat([image_rec, part_map_last_layer], dim=1),
                mu,
                self.patch_size,  # continue there
            )
            real_patches = ops_pt.torch_reshape(
                real_patches, [flatten_dim, -1, self.patch_size[0], self.patch_size[1]]
            )
            fake_patches_same_id = get_img_slice_around_mu(
                torch.cat([reconstruct_same_id, part_map_last_layer], dim=1),
                mu,
                self.patch_size,
            )
            fake_patches_same_id = ops_pt.torch_reshape(
                fake_patches_same_id,
                [flatten_dim, -1, self.patch_size[0], self.patch_size[1]],
            )
            patches = torch.cat([real_patches, fake_patches_same_id], dim=0)
            t_D, t_D_logits = self.discriminator(patches)

            additional_outputs = {
                "patches": patches,
                "t_D": t_D,  # real | fake
                "t_D_logits": t_D_logits,  # real | fake
            }
            outputs.update(additional_outputs)

        outputs = DotMap(outputs)

        return outputs


def make_visualizations(
    out,
    image_in,
    image_rec,
    heat_mask_l2,
    l_2_threshold,
    in_dim,
    part_depths,
    n_c,
    n_parts,
    bn,
    adversarial,
):
    visualizations = {}

    visualizations["g_reconstr"] = image_rec
    normal = utils.part_to_color_map(out.encoding_same_id, part_depths, size=in_dim)
    normal = normal / (1 + torch.sum(normal, dim=1, keepdim=True))
    vis_normal = torch.where(
        ops_pt.tile_nd(torch.sum(normal, dim=1, keepdim=True), [1, 3, 1, 1]) > 0.3,
        normal,
        image_in,
    )
    heat_mask_l2 = torch.nn.functional.interpolate(
        ops_pt.tile_nd(heat_mask_l2, [1, 3, 1, 1]), size=(in_dim, in_dim),
    )
    vis_normal = torch.where(heat_mask_l2 > l_2_threshold, vis_normal, 0.3 * vis_normal)
    visualizations["gt_t_1"] = vis_normal[:bn, ...]
    visualizations["gt_t_2"] = vis_normal[bn:, ...]
    visualizations["part_maps_t_1"] = utils.batch_colour_map(out.part_maps[:bn, ...])
    visualizations["part_maps_t_2"] = utils.batch_colour_map(out.part_maps[bn:, ...])

    if adversarial:
        f_dim = 2 * bn * n_parts
        visualizations["patch_real"] = out.patches[:f_dim, :n_c, :, :]
        visualizations["patch_fake"] = out.patches[
            f_dim : f_dim + f_dim // 2, :n_c, :, :
        ]
        visualizations["same_id_reconstruction"] = out.reconstruct_same_id
        visualizations["same_id_reconstruction"] = utils.summary_feat_and_parts(
            out.encoding_list, part_depths, False
        )

    # scale from range [0, 1] to [-1, 1]
    for k, v, in visualizations.items():
        visualizations[k] = (v - 0.5) * 2

    return visualizations
