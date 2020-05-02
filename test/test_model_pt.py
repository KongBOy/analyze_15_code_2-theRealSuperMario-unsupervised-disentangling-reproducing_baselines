from src.tf.model import ModelArgs
from supermariopy.ptutils import tps
from src.torch import ops_pt, model_pt
import torch
import torch.nn.functional as F


class Test_model:
    def test_forward(self):
        N = 1
        H = 128
        W = 128
        C = 3
        args = ModelArgs(bn=1)
        image_batch = torch.zeros((N, C, H, W), dtype=torch.float32)
        image_batch_tiled = ops_pt.tile_nd(image_batch, [2, 1, 1, 1])
        arg = ModelArgs(bn=1)

        tps_params = tps.no_transformation_parameters(2 * N)
        tps_param_dic = tps.tps_parameters(**tps_params)
        from dotmap import DotMap

        tps_param_dic = DotMap(tps_param_dic)

        model = model_pt.Model(
            heat_dim=args.heat_dim,
            nFeat_1=args.nFeat1,
            nFeat_2=args.nFeat2,
            L_inv_scal=args.L_inv_scal,
            rec_stages=args.rec_stages,
            part_depths=args.part_depths,
            feat_slices=args.feat_slices,
            covariance=args.covariance,
            average_features_mode=args.average_features_mode,
            heat_feat_normalize=args.heat_feat_normalize,
            static=args.static,
            reconstr_dim=args.reconstr_dim,
            n_c=args.n_c,
            bn=args.bn,
            n_parts=args.n_parts,
            adversarial=False,
            patch_size=args.patch_size,
        )

        coord, vector = tps.make_input_tps_param(tps_param_dic)
        t_images, t_mesh = tps.ThinPlateSpline(
            image_batch_tiled, coord, vector, args.in_dim, args.n_c
        )
        image_in, image_rec = ops_pt.prepare_pairs(
            t_images,
            model.reconstr_dim,
            train=True,
            static=False,
            contrast_var=args.contrast_var,
            brightness_var=args.brightness_var,
            saturation_var=args.saturation_var,
            hue_var=args.hue_var,
            p_flip=args.p_flip,
        )
        model = model
        out = model(image_in)
        mu = out.mu
        part_maps = out.part_maps
        transform_mesh = F.interpolate(t_mesh, (args.heat_dim, args.heat_dim))
        volume_mesh = ops_pt.AbsDetJacobian(transform_mesh)

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

    # def test_adversarial(self):
    #     N = 1
    #     H = 128
    #     W = 128
    #     C = 3
    #     args = ModelArgs(bn=1, adversarial=True)
    #     image_batch = torch.zeros((N, C, H, W), dtype=torch.float32)
    #     image_batch_tiled = ops_pt.tile_nd(image_batch, [2, 1, 1, 1])

    #     tps_params = tps.no_transformation_parameters(2 * N)
    #     tps_param_dic = tps.tps_parameters(**tps_params)
    #     from dotmap import DotMap

    #     tps_param_dic = DotMap(tps_param_dic)

    #     model = model_pt.Model(
    #         heat_dim=args.heat_dim,
    #         nFeat_1=args.nFeat1,
    #         nFeat_2=args.nFeat2,
    #         L_inv_scal=args.L_inv_scal,
    #         rec_stages=args.rec_stages,
    #         part_depths=args.part_depths,
    #         feat_slices=args.feat_slices,
    #         covariance=args.covariance,
    #         average_features_mode=args.average_features_mode,
    #         heat_feat_normalize=args.heat_feat_normalize,
    #         static=args.static,
    #         reconstr_dim=args.reconstr_dim,
    #         n_c=args.n_c,
    #         bn=args.bn,
    #         n_parts=args.n_parts,
    #         adversarial=args.adversarial,
    #         patch_size=args.patch_size,
    #         in_dim=args.in_dim,
    #         l_2_scal=args.l_2_scal,
    #         l_2_threshold=args.l_2_threshold,
    #     )

    #     coord, vector = tps.make_input_tps_param(tps_param_dic)
    #     t_images, t_mesh = tps.ThinPlateSpline(
    #         image_batch_tiled, coord, vector, args.in_dim, args.n_c
    #     )
    #     image_in, image_rec = ops_pt.prepare_pairs(
    #         t_images,
    #         model.reconstr_dim,
    #         train=True,
    #         static=False,
    #         contrast_var=args.contrast_var,
    #         brightness_var=args.brightness_var,
    #         saturation_var=args.saturation_var,
    #         hue_var=args.hue_var,
    #         p_flip=args.p_flip,
    #     )
    #     model = model
    #     out = model(image_in, image_rec)
    #     mu = out.mu
    #     part_maps = out.part_maps
    #     transform_mesh = F.interpolate(t_mesh, (model.heat_dim, model.heat_dim))
    #     volume_mesh = ops_pt.AbsDetJacobian(transform_mesh)

    #     integrant = torch.squeeze(
    #         torch.unsqueeze(part_maps, dim=-1) * torch.unsqueeze(volume_mesh, dim=-1)
    #     )
    #     integrant = integrant / torch.sum(integrant, dim=[2, 3], keepdims=True)
    #     mu_t = torch.einsum("akij,alij->akl", integrant, transform_mesh)
    #     transform_mesh_out_prod = torch.einsum(
    #         "amij,anij->aijmn", transform_mesh, transform_mesh
    #     )
    #     mu_out_prod = torch.einsum("akm,akn->akmn", mu_t, mu_t)
    #     stddev_t = (
    #         torch.einsum("akij,aijmn->akmn", integrant, transform_mesh_out_prod)
    #         - mu_out_prod
    #     )  # [2, 16, 2, 2]
