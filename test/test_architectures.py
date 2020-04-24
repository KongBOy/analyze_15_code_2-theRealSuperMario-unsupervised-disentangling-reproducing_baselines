import pytest
import numpy as np

import torch
import tensorflow as tf
from supermariopy.ptutils import nn as smptnn
from supermariopy.tfutils import nn as smtfnn
import sys

sys.path.insert(0, "/home/sandro/Projekte/github_projects/unsupervised-disentangling/")

# tf.enable_eager_execution()


class Test_Architectures:
    def test_disriminator_patch(self):
        import architectures

        image = tf.ones((1, 49, 49, 3), dtype=tf.float32)

        tf.random.set_random_seed(42)
        probs1, logits1 = architectures.discriminator_patch(image, True)
        tf.random.set_random_seed(42)
        probs2, logits2 = architectures.Disicriminator_Patch({"batch_size": 1})(
            image, True
        )
        assert np.allclose(probs1, probs2)
        assert np.allclose(logits1, logits2)

    def test_decoder(self):
        import architectures

        reconstr_dim = 128
        n_c = 3  # color dim
        encoding_list = [
            tf.zeros((1, 128 // (2 ** i), 128 // (2 ** i), 128)) for i in range(6)
        ]
        tf.random.set_random_seed(42)

        reconstruction = architectures.decoder(encoding_list, False, reconstr_dim, n_c)
        assert reconstruction.shape == (1, 128, 128, 3)

    def test_hourglass(self):
        N = 1
        H = 128
        W = 128
        C = 10
        x = tf.zeros((N, H, W, C))

        import architectures

        n_downsampling = 3
        n_out = 64
        y = architectures._hourglass(x, n_downsampling, n_out, False)
        assert smtfnn.shape_as_list(y) == [1, H, W, n_out]

    def test_seperate_hourglass(self):
        N = 1
        H = 128
        W = 128
        C = 10
        x = tf.zeros((N, H, W, C))

        import architectures

        n_landmark = 12
        n_features = 128
        nFeat_1 = 20
        nFeat_2 = 40
        y = architectures.seperate_hourglass(
            x, False, n_landmark, n_features, nFeat_1, nFeat_2
        )
        assert smtfnn.shape_as_list(y[0]) == [1, 64, 64, n_landmark]
        assert smtfnn.shape_as_list(y[1]) == [1, 64, 64, n_features]
