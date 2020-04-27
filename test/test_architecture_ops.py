import tensorflow as tf
import sys

sys.path.insert(0, "/home/sandro/Projekte/github_projects/unsupervised-disentangling/")

# tf.enable_eager_execution()


class Test_ArchitectureOps:
    def test_nccuc(self):
        from src.tf import architecture_ops

        N = 1
        H = 128
        W = 128
        C = 10

        # n_filters has more than 1 element: upsample input_A recursively
        input_B = tf.ones((N, H, W, C))
        n_filters = [20, 20, 20]
        n_upsampling_steps = len(n_filters) - 2
        input_A = tf.ones(
            (N, H // (2 ** n_upsampling_steps), W // (2 ** n_upsampling_steps), C)
        )
        x1 = architecture_ops.nccuc(
            input_A, input_B, n_filters, padding="SAME", training=True, name="test"
        )
        assert list(x1.shape) == [1, 128, 128, 30]

    def test_conv_block(self):
        from src.tf import architecture_ops

        N = 1
        H = 128
        W = 128
        C = 10
        C_out = 30

        x = tf.ones((N, H, W, C))
        c_bn_relu = architecture_ops._conv_block(x, C_out, False)
        assert list(c_bn_relu.shape) == [1, 128, 128, 30]
