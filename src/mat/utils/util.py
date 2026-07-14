import numpy as np
import math
import torch
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)

def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

# def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
#     """Decreases the learning rate linearly"""
#     lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def update_linear_schedule(optimizers, epoch, total_num_epochs, initial_lr):
    """
    Linearly decreases the learning rate from initial_lr → 0
    across total_num_epochs.
    """
    lr = initial_lr * (1 - epoch / float(total_num_epochs))
    lr = max(lr, 0.0)   # safety

    for opt in optimizers:
        for param_group in opt.param_groups:
            param_group["lr"] = lr


def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def quantile_huber_loss(
        current_quantiles: torch.Tensor,
        target_quantiles: torch.Tensor,
        cum_prob: Optional[torch.Tensor] = None,
        sum_over_quantiles: bool = True,
    ) -> torch.Tensor:
        """
        The quantile-regression loss, as described in the QR-DQN and TQC papers.
        Partially taken from https://github.com/bayesgroup/tqc_pytorch.

        :param current_quantiles: current estimate of quantiles, must be either
            (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
        :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
            (batch_size, n_quantiles)(batch_size, n_quantiles), or (batch_size, n_critics, n_target_quantiles)
        :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
            must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles).
            (if None, calculating unit quantiles)
        :param sum_over_quantiles: if summing over the quantile dimension or not
        :return: the loss
        """

        if current_quantiles.ndim != target_quantiles.ndim:
            raise ValueError(
                f"Error: The dimension of curremt_quantile ({current_quantiles.ndim}) needs to match "
                f"the dimension of target_quantiles ({target_quantiles.ndim})."
            )
        if current_quantiles.shape[0] != target_quantiles.shape[0]:
            raise ValueError(
                f"Error: The batch size of curremt_quantile ({current_quantiles.shape[0]}) needs to match "
                f"the batch size of target_quantiles ({target_quantiles.shape[0]})."
            )
        if current_quantiles.ndim not in (2, 3):
            raise ValueError(f"Error: The dimension of current_quantiles ({current_quantiles.ndim}) needs to be either 2 or 3.")

        if cum_prob is None:
            n_quantiles = current_quantiles.shape[-1]
            # Cumulative probabilities to calculate quantiles.
            cum_prob = (torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5) / n_quantiles
            if current_quantiles.ndim == 2:
                # For QR-DQN, current_quantiles have a shape (batch_size, n_quantiles), and make cum_prob
                # broadcastable to (batch_size, n_quantiles, n_target_quantiles)
                cum_prob = cum_prob.view(1, -1, 1)
            elif current_quantiles.ndim == 3:
                # For TQC, current_quantiles have a shape (batch_size, n_critics, n_quantiles), and make cum_prob
                # broadcastable to (batch_size, n_critics, n_quantiles, n_target_quantiles)
                cum_prob = cum_prob.view(1, 1, -1, 1)

        # QR-DQN
        # target_quantiles: (batch_size, n_target_quantiles) -> (batch_size, 1, n_target_quantiles)
        # current_quantiles: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
        # pairwise_delta: (batch_size, n_target_quantiles, n_quantiles)
        # TQC
        # target_quantiles: (batch_size, 1, n_target_quantiles) -> (batch_size, 1, 1, n_target_quantiles)
        # current_quantiles: (batch_size, n_critics, n_quantiles) -> (batch_size, n_critics, n_quantiles, 1)
        # pairwise_delta: (batch_size, n_critics, n_quantiles, n_target_quantiles)
        # Note: in both cases, the loss has the same shape as pairwise_delta
        pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
        loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss

        if sum_over_quantiles:
            loss = loss.sum(dim=-2).mean(dim=-1).mean(dim=0)
        else:
            loss = loss.mean(dim=-1).mean(dim=-1).mean(dim=0)

        return loss.unsqueeze(-1)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c