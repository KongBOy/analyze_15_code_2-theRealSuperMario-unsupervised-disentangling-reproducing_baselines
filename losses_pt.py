import torch
from ops_pt import fold_img_with_mu, fold_img_with_L_inv


def transform_loss(mu_t_1, mu_t_2):
    loss = torch.mean((mu_t_1 - mu_t_2) ** 2)
    return loss


def precision_loss(stddev_t_1, stddev_t_2):
    precision_sq = (stddev_t_1 - stddev_t_2) ** 2
    eps = 1e-6
    loss = torch.mean(torch.sqrt(torch.sum(precision_sq, dim=[2, 3]) + eps))
    return loss


def reconstruction_loss(
    reconstruct_same_id,
    image_rec,
    mu,
    L_inv,
    l_2_scal,
    l_2_threshold,
    use_l1=False,
    fold_with_shape=False,
):
    img_difference = reconstruct_same_id - image_rec

    if use_l1:
        distance_metric = torch.abs(img_difference)

    else:
        distance_metric = img_difference ** 2

    if fold_with_shape:
        fold_img_squared = fold_img_with_L_inv(
            distance_metric,
            mu.detach(),
            L_inv.detach(),
            l_2_scal,
            visualize=False,
            threshold=l_2_threshold,
            normalize=True,
        )
    else:
        fold_img_squared, heat_mask_l2 = fold_img_with_mu(
            distance_metric, mu, l_2_scal, threshold=l_2_threshold, normalize=True,
        )

    l2_loss = torch.mean(torch.sum(fold_img_squared, dim=[1, 2]))

    return l2_loss, heat_mask_l2


def adversarial_loss(real_logits, fake_logits):
    criterion = torch.nn.BCEWithLogitsLoss()
    # flatten_dim = 2 * self.arg.bn * self.arg.n_parts
    # D, D_ = self.t_D[:flatten_dim], self.t_D[flatten_dim:] # real | fake
    # D_logits, D_logits_ = (
    #     self.t_D_logits[:flatten_dim],
    #     self.t_D_logits[flatten_dim:],
    # )

    # d_loss_real = torch.mean(
    #     torch.sigmoid_cross_entropy_with_logits(
    #         logits=D_logits, labels=torch.ones_like(D)
    #     )
    # )
    # d_loss_fake = torch.mean(
    #     tf.nn.sigmoid_cross_entropy_with_logits(
    #         logits=D_logits_, labels=torch.zeros_like(D_)
    #     )
    # )
    # g_loss = torch.mean(
    #     torch.nn.sigmoid_cross_entropy_with_logits(
    #         logits=D_logits_, labels=torch.ones_like(D_)
    #     )
    # )
    d_loss_fake = criterion(fake_logits, torch.zeros_like(fake_logits))
    d_loss_real = criterion(real_logits, torch.ones_like(fake_logits))
    d_loss = d_loss_real + d_loss_fake

    g_loss = criterion(fake_logits, torch.ones_like(fake_logits))
    return d_loss, g_loss


# TOTAL loss
# transform_loss, precision_loss, l2_loss, d_loss, g_loss = self.loss

# total_loss = (
#     self.arg.c_l2 * l2_loss
#     + self.arg.c_trans * transform_loss
#     + self.arg.c_precision_trans * precision_loss
#     + self.arg.c_g * g_loss
# )
