#import torch as th
from palette_diffusion.palette_diffusion import sum_flat
from guided_diffusion.nn import mean_flat
from guided_diffusion.Wavelet import multi_level_wavelet,wavelet_new
import torch


def load_guider(task):
    """
    Returns log p_0(c|x_0) based on the specified task.
    """
    # For inpainting, p_0(c|x_0) = exp(-||cond - mask * x_0||^2), where cond is 
    # the measurement, and mask = 1 in the measurement region and 0 elsewhere.
    if task == "inpainting":
        def get_guidance(x0, guider_kwargs):
            cond = guider_kwargs["cond"]  # [B, C, H, ...]
            mask = guider_kwargs["mask"]  # [B, C, H, ...]
            return -sum_flat(mask * (cond - x0) ** 2)  / sum_flat(mask)  # [B]

    # # For threshold_acc, p_0(c|x_0) = exp(-ReLU(threshold - max(acc(t_target)))), 
    # # where max is taken over the acceleration components at t_target, ensuring 
    # # that at least one component exceeds the threshold.
    # elif task == "threshold_acc":
    #     def get_guidance(x0, guider_kwargs):
    #         # Target time index for evaluating acceleration
    #         t_idx = guider_kwargs["t_target"]
    #         assert 0 <= t_idx < x0.shape[-1] - 1, "t_target is out of bounds."
    #         # Acceleration scale factor (used to normalize the threshold)
    #         std_acc = th.tensor(0.0040588385452668034, dtype=th.float32, device=x0.device)
    #         threshold = guider_kwargs["threshold"] * std_acc
    #         # Compute finite-difference acceleration at time t_idx
    #         acc = th.abs(x0[..., t_idx+1] - x0[..., t_idx])  # [B, C]
    #         # Log-Sum-Exp approximation of max(acc)
    #         beta = 10
    #         lse_max = (1/beta) * th.logsumexp(beta*acc, dim=1)  # [B]
    #         # Penalize if the max acceleration is below the threshold
    #         return -th.relu(threshold - lse_max)  # [B]

    # For moment_acc, p_0(c|x_0) = exp(-<acc^order>), which encourages a larger 
    # even-order moment of acceleration over a trajectory and thus preferentially 
    # samples extreme events.
    elif task == "moment_acc":
        def get_guidance(x0, guider_kwargs):
            order = guider_kwargs["moment_order"]
            assert order % 2 == 0, "moment order must be an even number."
            acc = x0[..., 1:] - x0[..., :-1]  # Compute acceleration, [B, C, H-1]
            return -mean_flat(acc ** order)  # [B]
        

    # elif task == "scale_reconstruction":
    #     def get_guidance(x0, guider_kwargs):
    #         level = 3
    #         x0 = wavelet_new(x0, level)
    #         cond = guider_kwargs["cond"]
    #         cond = wavelet_new(cond, level)
    #         return -sum_flat((cond - x0) ** 2)
            
    elif task.startswith("scale_reconstruction_level"):
        level = int(task.split("_")[-1].replace("level", ""))
        def get_guidance(x0, guider_kwargs):
            x0 = wavelet_new(x0, level)
            cond = guider_kwargs["cond"]
            cond = wavelet_new(cond, level)
            return -sum_flat((cond - x0) ** 2)
        

    elif task == "haar_sr":
        def get_guidance(x0, guider_kwargs):
            level = guider_kwargs["level"]
            reference = guider_kwargs["reference"]  # full-resolution signal [B, C, L]
            with torch.no_grad():
                ref_approx = compute_haar_approx(reference, level)  # [B, C, L // 2^level]
            x0_approx = compute_haar_approx(x0, level)  # [B, C, L // 2^level]
            return -sum_flat((ref_approx - x0_approx) ** 2)  # [B]

    else:
        raise NotImplementedError
    return get_guidance

def wavelet_new(x, level):
    batch_size, features, seq_len = x.size()
    group_size = 2**level
    num_groups = seq_len // group_size  
    x_mean = torch.mean(x.view(batch_size,features, num_groups, group_size), dim=3)
    x_mean = x_mean*group_size**(1/2)
    return x_mean

def compute_haar_approx(x, level):
    """
    Compute Haar approximation coefficients at given level.
    :param x: a Tensor of shape [B, C, L]
    :param level: int, Haar level
    :return: a Tensor of shape [B, C, L // 2^level]
    """
    B, C, L = x.shape
    assert L % (2**level) == 0
    x = x.view(B, C, -1, 2**level).mean(dim=-1)
    scale = x.new_tensor(2**(0.5 * level))
    return x * scale
