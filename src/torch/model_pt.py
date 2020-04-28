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
        tps_par,
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
        static,
        reconstr_dim,
        n_c,
        bn,
        n_parts,
        adversarial,
        patch_size,
        in_dim,
        contrast_var,
        brightness_var,
        saturation_var,
        hue_var,
        p_flip,
        l_2_scal,
        l_2_threshold,
    ):
        super(Model, self).__init__()
        self.tps_par = tps_par
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
        self.static = static
        self.reconstr_dim = reconstr_dim
        self.n_c = n_c
        self.bn = bn
        self.n_parts = n_parts
        self.adversarial = adversarial
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.contrast_var = contrast_var
        self.brightness_var = brightness_var
        self.saturation_var = saturation_var
        self.hue_var = hue_var
        self.p_flip = p_flip
        if self.adversarial:
            self.discriminator = architectures_pt.Discriminator_Patch()
        self.l_2_threshold = l_2_threshold
        self.l_2_scal = l_2_scal

    def forward(self, x):
        image_orig = x
        coord, vector = make_input_tps_param(self.tps_par)
        t_images, t_mesh = ThinPlateSpline(
            image_orig, coord, vector, self.in_dim, self.n_c
        )
        image_in, image_rec = prepare_pairs(
            t_images,
            self.reconstr_dim,
            train=self.training,
            static=self.static,
            contrast_var=self.contrast_var,
            brightness_var=self.brightness_var,
            saturation_var=self.saturation_var,
            hue_var=self.hue_var,
            p_flip=self.p_flip,
        )
        transform_mesh = F.interpolate(t_mesh, (self.heat_dim, self.heat_dim))
        volume_mesh = AbsDetJacobian(transform_mesh)

        part_maps, raw_features = self.encoder(
            image_in
        )  # [2, 16, 64, 64] and [2, 64, 64, 64]

        mu, L_inv = part_map_to_mu_L_inv(part_maps, self.L_inv_scal)
        features = get_features(raw_features, part_maps, True)
        integrant = torch.squeeze(
            torch.unsqueeze(part_maps, dim=-1) * torch.unsqueeze(volume_mesh, dim=-1)
        )
        integrant = integrant / torch.sum(integrant, dim=[2, 3], keepdims=True)
        mu_t = torch.einsum("akij,alij->akl", integrant, transform_mesh)
        transform_mesh_out_prod = torch.einsum(
            "amij,anij->aijmn", transform_mesh, transform_mesh
        )
        mu_out_prod = torch.einsum("akm,akn->akmn", mu_t, mu_t)
        stddev_t = (
            torch.einsum("akij,aijmn->akmn", integrant, transform_mesh_out_prod)
            - mu_out_prod
        )  # [2, 16, 2, 2]

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
            static=self.static,
        )
        encoding_same_id = [
            tfpyth.th_2D_channels_last_to_first(e) for e in encoding_same_id
        ]

        reconstruct_same_id = self.img_decoder(encoding_same_id)

        outputs = {
            "image_in": image_in,
            "image_rec": image_rec,
            "transform_mesh": transform_mesh,
            "volume_mesh": volume_mesh,
            "part_maps": part_maps,
            "mu": mu,
            "L_inv": L_inv,
            "features": features,
            "integrant": integrant,
            "mu_t": mu_t,
            "stddev_t": stddev_t,
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


def make_visualizations(out, model, heat_mask_l2):
    visualizations = {}

    visualizations["g_reconstr"] = out.image_rec
    normal = utils.part_to_color_map(
        out.encoding_same_id, model.part_depths, size=model.in_dim
    )
    normal = normal / (1 + torch.sum(normal, dim=1, keepdim=True))
    vis_normal = torch.where(
        ops_pt.tile_nd(torch.sum(normal, dim=1, keepdim=True), [1, 3, 1, 1]) > 0.3,
        normal,
        out.image_in,
    )
    heat_mask_l2 = torch.nn.functional.interpolate(
        ops_pt.tile_nd(heat_mask_l2, [1, 3, 1, 1]), size=(model.in_dim, model.in_dim),
    )
    vis_normal = torch.where(
        heat_mask_l2 > model.l_2_threshold, vis_normal, 0.3 * vis_normal
    )
    visualizations["gt_t_1"] = vis_normal[: model.bn, ...]
    visualizations["gt_t_2"] = vis_normal[model.bn :, ...]
    visualizations["part_maps_t_1"] = utils.batch_colour_map(
        out.part_maps[: model.bn, ...]
    )
    visualizations["part_maps_t_2"] = utils.batch_colour_map(
        out.part_maps[model.bn :, ...]
    )

    if model.adversarial:
        f_dim = 2 * model.bn * model.n_parts
        visualizations["patch_real"] = out.patches[:f_dim, : model.n_c, :, :]
        visualizations["patch_fake"] = out.patches[
            f_dim : f_dim + f_dim // 2, : model.n_c, :, :
        ]
        visualizations["same_id_reconstruction"] = out.reconstruct_same_id
        visualizations["same_id_reconstruction"] = utils.summary_feat_and_parts(
            out.encoding_list, model.part_depths, False
        )
    return visualizations
