import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf

import numpy as np
from edflow import TemplateIterator, get_logger
from src.torch import model_pt
from supermariopy.ptutils import losses as ptlosses
from src.torch import losses_pt
from supermariopy.ptutils import tps
from supermariopy.ptutils import compat as ptcompat
from dotmap import DotMap
import warnings

from supermariopy.ptutils import utils as ptu
from src.torch import ops_pt


from supermariopy.ptutils.tps import ThinPlateSpline, make_input_tps_param

# from torchvision.transforms import functional as F
from torch.nn import functional as F
from src.torch import architectures_pt, ops_pt, utils
from dotmap import DotMap


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.config = DotMap(config)
        self.model_args = DotMap(config["model_params"])

        self.spatial_size = config["spatial_size"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]

        self.net = model_pt.Model(**self.model_args)

    def forward(self, image_in, image_rec=None):
        out = self.net(image_in, image_rec)
        return out


def split_batch(x):
    bs = list(x.shape)[0]
    return x[: (bs // 2), ...], x[(bs // 2) :, ...]


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.config = self.model.config

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.disc_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.disc_lr
        )
        self.loss_params = self.model.config.loss_params

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.disc_optimizer.load_state_dict(state["disc_optimizer"])

    def step_op(self, model, **kwargs):
        x = kwargs["image_in"]
        if not self.config.model_params.static:
            x = np.reshape(
                x, (-1, self.config.spatial_size, self.config.spatial_size, 3)
            )
            x = to_torch(x, permute=True)
            image_batch_tiled = x
        else:
            x = to_torch(x, permute=True)
            image_batch_tiled = ptcompat.torch_tile_nd(x, [2, 1, 1, 1])

        def train_op():
            # compute losses and run optimization
            model.train()

            # prepare tps
            tps_params = self.config.tps_params
            tps_params["batch_size"] = 2 * self.config.batch_size
            tps_param_dic = tps.tps_parameters(**tps_params)
            coord, vector = tps.make_input_tps_param(tps_param_dic)
            t_images, t_mesh = tps.ThinPlateSpline(
                image_batch_tiled,
                coord,
                vector,
                self.config.model_params.in_dim,
                self.config.model_params.n_c,
            )
            image_in, image_rec = ops_pt.prepare_pairs(
                t_images,
                self.config.model_params.reconstr_dim,
                train=True,
                static=self.config.model_params.static,
                contrast_var=self.config.contrast_var,
                brightness_var=self.config.brightness_var,
                saturation_var=self.config.saturation_var,
                hue_var=self.config.hue_var,
                p_flip=self.config.p_flip,
            )
            out = model(image_in, image_rec)

            mu = out.mu
            part_maps = out.part_maps
            transform_mesh = F.interpolate(
                t_mesh,
                (self.config.model_params.heat_dim, self.config.model_params.heat_dim),
            )
            mu_t_1, mu_t_2, stddev_t_1, stddev_t_2 = propagate_mu_L(
                transform_mesh, part_maps
            )

            transform_loss_val = losses_pt.transform_loss(mu_t_1, mu_t_2)
            precision_loss_val = losses_pt.precision_loss(stddev_t_1, stddev_t_2)
            l2_loss, heat_mask_l2 = losses_pt.reconstruction_loss(
                out.reconstruct_same_id,
                image_rec,
                out.mu.detach(),
                out.L_inv.detach(),
                self.config.model_params.l_2_scal,
                self.config.model_params.l_2_threshold,
            )

            real_logits, fake_logits = split_batch(out.t_D_logits)
            d_loss, g_loss = losses_pt.adversarial_loss(real_logits, fake_logits)

            total_loss = (
                transform_loss_val * self.loss_params.lambda_t
                + precision_loss_val * self.loss_params.lambda_p
                + l2_loss * self.loss_params.lambda_r
                + g_loss * self.loss_params.lambda_g
            )

            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.optimizer.step()

            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()

        def log_op():
            with torch.no_grad():
                # calculate logs every now and then
                model.eval()

                # prepare tps
                tps_params = self.config.tps_params
                tps_params["batch_size"] = 2 * self.config.batch_size

                # TODO: replace with actual tps sampling
                tps_param_dic = tps.tps_parameters(**tps_params)
                coord, vector = tps.make_input_tps_param(tps_param_dic)
                t_images, t_mesh = tps.ThinPlateSpline(
                    image_batch_tiled,
                    coord,
                    vector,
                    self.config.model_params.in_dim,
                    self.config.model_params.n_c,
                )
                image_in, image_rec = ops_pt.prepare_pairs(
                    t_images,
                    self.config.model_params.reconstr_dim,
                    train=False,  # train applies augmentation transforms
                    static=self.config.model_params.static,
                    contrast_var=self.config.contrast_var,
                    brightness_var=self.config.brightness_var,
                    saturation_var=self.config.saturation_var,
                    hue_var=self.config.hue_var,
                    p_flip=self.config.p_flip,
                )
                out = model(image_in, image_rec)

                mu = out.mu
                part_maps = out.part_maps
                transform_mesh = F.interpolate(
                    t_mesh,
                    (
                        self.config.model_params.heat_dim,
                        self.config.model_params.heat_dim,
                    ),
                )
                mu_t_1, mu_t_2, stddev_t_1, stddev_t_2 = propagate_mu_L(
                    transform_mesh, part_maps
                )

                transform_loss_val = losses_pt.transform_loss(mu_t_1, mu_t_2)
                precision_loss_val = losses_pt.precision_loss(stddev_t_1, stddev_t_2)
                l2_loss, heat_mask_l2 = losses_pt.reconstruction_loss(
                    out.reconstruct_same_id,
                    image_rec,
                    out.mu.detach(),
                    out.L_inv.detach(),
                    self.config.model_params.l_2_scal,
                    self.config.model_params.l_2_threshold,
                )

                # todo: add adversarial logs

                total_loss = (
                    transform_loss_val * self.loss_params.lambda_t
                    + precision_loss_val * self.loss_params.lambda_p
                    + l2_loss * self.loss_params.lambda_r
                )

                logs = {"images": {}, "scalars": {}}  # images and scalars
                scalar_logs = {
                    "loss_transform": transform_loss_val,
                    "loss_precision": precision_loss_val,
                    "loss_l2": l2_loss,
                    "loss_total": total_loss,
                }

                visualizations = model_pt.make_visualizations(
                    out,
                    image_in,
                    image_rec,
                    heat_mask_l2,
                    transform_mesh,
                    self.config.model_params.l_2_threshold,
                    self.config.model_params.in_dim,
                    self.config.model_params.part_depths,
                    self.config.model_params.n_c,
                    self.config.model_params.n_parts,
                    self.config.model_params.bn,
                    self.config.model_params.adversarial,
                )

                logs["scalars"].update(scalar_logs)
                logs["images"].update(visualizations)

                import functools

                func = functools.partial(to_numpy, permute=True)
                logs["images"] = recursive_apply(logs["images"], func)
                logs["scalars"] = recursive_apply(logs["scalars"], to_numpy)
            return logs

        def eval_op():
            return {}

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}


def propagate_mu_L(transform_mesh, part_maps):
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

    mu_t_1, mu_t_2 = split_batch(mu_t)
    stddev_t_1, stddev_t_2 = split_batch(stddev_t)
    return mu_t_1, mu_t_2, stddev_t_1, stddev_t_2


def to_numpy(x, permute=False):
    """automatically detach and move to cpu if necessary."""
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().numpy()
    if isinstance(x, np.ndarray):
        if permute:
            x = np.transpose(x, (0, 2, 3, 1))  # NCHW --> NHWC
    return x


def to_torch(x, permute=False):
    """automatically convert numpy array to torch and permute channels from NHWC to NCHW"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = x.to(device)

    if permute:
        x = x.permute((0, 3, 1, 2))  # NHWC --> NCHW
    if x.dtype is torch.float64:
        x = x.type(torch.float32)
    return x


def recursive_apply(d: dict, func: callable):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = recursive_apply(v, func)
        else:
            d[k] = func(v)
    return d
