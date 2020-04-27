import torch
from supermariopy.ptutils import nn as smptnn

import sys

sys.path.insert(0, "/home/sandro/Projekte/github_projects/unsupervised-disentangling/")


class Test_Architectures_pt:
    def test_decoder128_pt(self):
        from src.torch import architectures_pt

        reconstr_dim = 128
        n_c = 3  # color dim
        encoding_list = [
            torch.zeros((1, 128, 128 // (2 ** i), 128 // (2 ** i))) for i in range(6)
        ]

        decoder = architectures_pt.Decoder128([128, ] * 6, n_c)
        reconstruction = decoder(encoding_list)
        assert smptnn.shape_as_list(reconstruction) == [1, 3, 128, 128]

    def test_hourglass_pt(self):
        N = 1
        H = 128
        W = 128
        C = 64
        x = torch.zeros((N, C, H, W))

        from src.torch import architectures_pt

        y = architectures_pt.Hourglass(C, 10, architectures_pt.Bottleneck, 1, 32, 3)(x)
        assert smptnn.shape_as_list(y) == [1, 10, 128, 128]

    def test_seperate_hourglass_pt(self):
        N = 1
        H = 128
        W = 128
        C = 10
        x = torch.zeros((N, C, H, W))

        from src.torch import architectures_pt

        n_landmark = 12
        n_features = 128
        nFeat_1 = 20
        nFeat_2 = 40
        y = architectures_pt.SeperateHourglass_128(
            C, n_landmark, n_features, nFeat_1, nFeat_2
        )(x)
        assert smptnn.shape_as_list(y[0]) == [1, n_landmark, 64, 64]
        assert smptnn.shape_as_list(y[1]) == [1, n_features, 64, 64]

    def test_discriminator_pt(self):
        N = 1
        H = 49
        W = 49
        C = 3
        x = torch.zeros((N, C, H, W))

        from src.torch import architectures_pt

        y_probs, y_logits = architectures_pt.Discriminator_Patch(in_channels=3)(x)
        assert y_probs.shape == (1, 1)
        assert y_logits.shape == (1, 1)
