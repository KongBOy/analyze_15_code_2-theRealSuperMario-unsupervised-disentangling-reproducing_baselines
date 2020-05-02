import numpy as np

import tensorflow as tf

tf.enable_eager_execution()

import sys

sys.path.insert(0, "/home/sandro/Projekte/github_projects/unsupervised-disentangling/")
# tf.enable_eager_execution()


class Test_Model:
    def test_model(self):
        from src.tf.model import Model, ModelArgs
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

    # def test_model_visualizations(self):
    #     from src.tf.model import Model, ModelArgs
    #     from supermariopy.tfutils import tps

    #     N = 1
    #     H = 128
    #     W = 128
    #     C = 3
    #     image_batch = np.zeros((N, H, W, C), dtype=np.float32)
    #     image_batch_tiled = tf.tile(image_batch, [2, 1, 1, 1])
    #     arg = ModelArgs(bn=1)

    #     tps_params = tps.no_transformation_parameters(2 * N)
    #     tps_param_dic = tps.tps_parameters(**tps_params)
    #     from dotmap import DotMap

    #     tps_param_dic = DotMap(tps_param_dic)

    #     model = Model(
    #         image_batch_tiled, arg, tps_param_dic, optimize=False, visualize=True
    #     )

