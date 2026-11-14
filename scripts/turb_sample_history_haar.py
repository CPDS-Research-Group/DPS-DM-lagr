"""
Generate a large batch of Lagrangian trajectories using training-free 
guidance (TFG) from a model and save their ENTIRE SAMPLING HISTORY 
(sample and pred_xstart) as large numpy arrays.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from palette_diffusion.palette_datasets import load_palette_data
from tfg_diffusion.script_util import (
    tfg_model_and_diffusion_defaults, 
    guidance_defaults,
    tfg_create_model_and_diffusion,
)
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating TFG model and diffusion...")
    model, diffusion = tfg_create_model_and_diffusion(
        **args_to_dict(args, tfg_model_and_diffusion_defaults().keys()),
        task=args.task, 
        guidance_kwargs=args_to_dict(args, guidance_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating data loader...")
    data = load_palette_data(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        deterministic=True,
    )

    logger.log("sampling with history...")
    seed = args.seed * 4 + int(os.environ.get("CUDA_VISIBLE_DEVICES", 0))
    th.manual_seed(seed)
    
    curr_batch, num_complete = 0, 0
    while num_complete < args.num_samples:
        model_kwargs = {}
        
        # 1. 从 data 迭代器获取并转移 guider_kwargs
        _, guider_kwargs = next(data)
        guider_kwargs = {k: v.to(dist_util.dev()) for k, v in guider_kwargs.items()}

        # 2. 依据条件数据的形状（cond）和设备初始化纯噪声
        shape = guider_kwargs["cond"].shape
        device = dist_util.dev()
        img = th.randn(*shape, device=device)
        
        # 3. 初始化用于保存历史轨迹的列表
        # 注意：sample_history 包括了最初始的纯噪声，所以长度会是 Timesteps + 1
        sample_history = [img.clone()]
        xstart_history = []  # 用于存储每一步的最优估计 (x0)
        
        indices = list(range(diffusion.num_timesteps))[::-1]

        # 4. 手动执行逐步引导和记录
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                # 调用我们在底层修改过、返回元组的 guide_step
                img, current_xstart = diffusion.guide_step(
                    model, img, t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    guider_kwargs=guider_kwargs,
                )
                sample_history.append(img.clone())
                xstart_history.append(current_xstart.clone())
        
        # 5. 堆叠张量 (增加时间步维度 dim=1)
        sample = th.stack(sample_history, dim=1)
        pred_xstart = th.stack(xstart_history, dim=1)
        
        # 对最终生成的样本步进行截断限制
        sample[:, -1] = sample[:, -1].clamp(-1, 1)

        # 6. 调用辅助函数进行分布式收集并保存到硬盘
        gather_and_save_samples(sample, curr_batch, "sample", args.seed)
        gather_and_save_samples(pred_xstart, curr_batch, "pred_xstart", args.seed)

        curr_batch += 1
        num_complete += dist.get_world_size() * args.batch_size
        logger.log(f"created {num_complete} samples")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        task="",
        dataset_path="",
        dataset_name="",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        seed=0, 
    )
    # 合并 TFG 和 Guidance 的参数默认值
    defaults.update(tfg_model_and_diffusion_defaults())
    defaults.update(guidance_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def gather_and_save_samples(tensor_data, curr_batch, sample_name, seed):
    """
    负责将传入的四维张量进行维度转换、跨GPU收集并保存为 npz 文件。
    """
    # tensor_data 形状预期为: (Batch, Timesteps, Channels, Length)
    # 将 Channels (dim=2) 和 Length (dim=3) 互换，变为 (Batch, Timesteps, Length, Channels)
    tensor_data = tensor_data.permute(0, 1, 3, 2)
    tensor_data = tensor_data.contiguous()

    # 分布式收集 (All-Gather)
    gathered_samples = [th.zeros_like(tensor_data) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_samples, tensor_data)  # gather not supported with NCCL
    all_images = [t.cpu().numpy() for t in gathered_samples]
    
    arr = np.concatenate(all_images, axis=0)
    
    # 在主节点(Rank 0)执行保存
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_dir = os.path.join(logger.get_dir(), f"samples_history-seed{int(seed):06d}")
        os.makedirs(out_dir, exist_ok=True)
        
        out_path = os.path.join(
            out_dir,
            f"batch{curr_batch:03d}-{sample_name}s_history_{shape_str}.npz"
        )
        logger.log(f"saving {sample_name} to {out_path}")
        np.savez(out_path, arr)


if __name__ == "__main__":
    main()