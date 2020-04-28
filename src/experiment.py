import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_args = DotMap(config["model_params"])

        self.spatial_size = config["spatial_size"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        tps_params = tps.no_transformation_parameters(2 * self.batch_size)
        tps_param_dic = tps.tps_parameters(**tps_params)

        self.net = model_pt.Model(tps_par=tps_param_dic, **self.model_args)

    def forward(self, x):
        out = self.net(x)
        return out

    # def transfer(self, x):
    #     # TODO
    #     return imgs


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
        # self.vgg_loss = ptlosses.VGGLossWithL1(0, l1_alpha=1.0e-3)
        # self.vgg_loss = self.vgg_loss.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr)
        self.loss_params = DotMap(self.model.config["loss_params"])

    @property
    def callbacks(self):
        # return {"eval_op": {"acc_callback": acc_callback}}
        pass

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def step_op(self, model, **kwargs):
        # get inputs

        image_in = kwargs["image_in"]
        image_rec = kwargs["image_rec"]
        image_in = to_torch(image_in, permute=True)
        image_in = ptcompat.torch_tile_nd(image_in, [2, 1, 1, 1])
        image_rec = to_torch(image_rec, permute=True)

        def train_op():
            # compute losses and run optimization
            model.train()
            out = model(image_in)
            mu_t_1, mu_t_2 = split_batch(out.mu)
            stddev_t_1, stddev_t_2 = split_batch(out.stddev_t)

            transform_loss_val = losses_pt.transform_loss(mu_t_1, mu_t_2)
            precision_loss_val = losses_pt.precision_loss(stddev_t_1, stddev_t_2)
            l2_loss, heat_mask_l2 = losses_pt.reconstruction_loss(
                out.reconstruct_same_id,
                out.image_rec,
                out.mu.detach(),
                out.L_inv.detach(),
                self.model.model_args.l_2_scal,
                self.model.model_args.l_2_threshold,
            )

            total_loss = (
                transform_loss_val * self.loss_params.lambda_t
                + precision_loss_val * self.loss_params.lambda_p
                + l2_loss * self.loss_params.lambda_r
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        def log_op():
            # calculate logs every now and then
            model.eval()
            out = model(image_in)
            mu_t_1, mu_t_2 = split_batch(out.mu)
            stddev_t_1, stddev_t_2 = split_batch(out.stddev_t)

            transform_loss_val = losses_pt.transform_loss(mu_t_1, mu_t_2)
            precision_loss_val = losses_pt.precision_loss(stddev_t_1, stddev_t_2)
            l2_loss, heat_mask_l2 = losses_pt.reconstruction_loss(
                out.reconstruct_same_id,
                out.image_rec,
                out.mu.detach(),
                out.L_inv.detach(),
                self.model.model_args.l_2_scal,
                self.model.model_args.l_2_threshold,
            )

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
            visualizations = model_pt.make_visualizations(out, model.net, heat_mask_l2)

            logs["scalars"].update(scalar_logs)
            logs["images"].update(visualizations)

            import functools

            func = functools.partial(to_numpy, permute=True)
            logs["images"] = recursive_apply(logs["images"], func)
            logs["scalars"] = recursive_apply(logs["scalars"], to_numpy)
            return logs

        def eval_op():
            model.eval()
            eval_logs = {"outputs": {}, "labels": {}}  # eval_logs
            return eval_logs

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}


# TODO: perceptual loss


def to_numpy(x):
    """automatically detach and move to cpu if necessary."""
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().numpy()
    return x


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
