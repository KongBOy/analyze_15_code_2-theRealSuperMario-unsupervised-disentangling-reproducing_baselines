import pytest
from model import ModelArgs
from supermariopy.ptutils import tps
import tensorflow as tf
import numpy as np
import model_pt
import architectures_pt
import ops_pt
import torch


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
            tps_param_dic,
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
