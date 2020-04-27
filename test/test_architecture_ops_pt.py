import torch
import sys

sys.path.insert(0, "/home/sandro/Projekte/github_projects/unsupervised-disentangling/")
from src.tf import architecture_ops


class Test_ArchitectureOps_pt:
    def test_nccuc_pt(self):
        N = 1
        H = 128
        W = 128
        C = 10

        input_B = torch.ones((N, C, H, W))
        input_A = torch.ones((N, C, H // 2, W // 2))
        nccuc = architecture_ops.NCCUC(C, 30, 30)
        out = nccuc(input_A, input_B)

    def test_convbnrelu_pt(self):
        N = 1
        H = 128
        W = 128
        C = 10

        x = torch.ones((N, C, H, W))
        c_bn_relu = architecture_ops.ConvBnRelu(C, 256)(x)
        assert list(c_bn_relu.shape) == [1, 256, H, W]

    def test_conv_block_pt(self):
        N = 1
        H = 128
        W = 128
        C = 10
        C_out = 30

        x = torch.ones((N, C, H, W))
        c_bn_relu = architecture_ops.ConvBLock(C, C_out)(x)
        assert list(c_bn_relu.shape) == [1, C_out, H, W]
