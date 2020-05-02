from src.tf.model import ModelArgs
from supermariopy.ptutils import tps
from src.torch import ops_pt, losses_pt, model_pt
import torch


def split_batch(x):
    bs = list(x.shape)[0]
    return x[: (bs // 2), ...], x[(bs // 2) :, ...]


def test_without_adversarial():
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
        in_dim=args.in_dim,
        contrast_var=args.contrast_var,
        brightness_var=args.brightness_var,
        saturation_var=args.saturation_var,
        hue_var=args.hue_var,
        p_flip=args.p_flip,
    )

    out = model(image_batch_tiled)
    assert True


# def test_with_adversarial():
#     N = 1
#     H = 128
#     W = 128
#     C = 3
#     args = ModelArgs(bn=1)
#     image_batch = torch.zeros((N, C, H, W), dtype=torch.float32)
#     image_batch_tiled = ops_pt.tile_nd(image_batch, [2, 1, 1, 1])
#     arg = ModelArgs(bn=1)

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
#         adversarial=True,
#         patch_size=args.patch_size,
#         in_dim=args.in_dim,
#         contrast_var=args.contrast_var,
#         brightness_var=args.brightness_var,
#         saturation_var=args.saturation_var,
#         hue_var=args.hue_var,
#         p_flip=args.p_flip,
#     )

#     out = model(image_batch_tiled, image_batch_tiled)

#     logits_real, logits_fake = split_batch(out.t_D_logits)
#     adv_loss = losses_pt.adversarial_loss(logits_real, logits_fake)
#     assert True

