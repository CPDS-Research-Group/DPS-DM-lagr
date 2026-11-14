from guided_diffusion.script_util import create_model
from guided_diffusion import gaussian_diffusion as gd
from .tfg_diffusion import (
    BaseGuidanceDiffusion,
    DPSGuidanceDiffusion,
    TFGGuidanceDiffusion,
)#不同类的引导方法


def tfg_diffusion_defaults():
    return dict(
        diffusion_steps=1000,
        sigma_small=False,#False 8.1#是否使用小方差 √可以使用
        noise_schedule="linear",#线性噪声
        predict_xstart=False,
        rescale_timesteps=False,#是否重新缩放时间步√可以使用
    )


def tfg_model_and_diffusion_defaults():
    res = dict(
        dims=2,
        image_size=64,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,#8.20推理过程中一般设置为False
        use_scale_shift_norm=True,
        resblock_updown=False,#False 8.1
        use_fp16=False,
        use_new_attention_order=False,#False 8.1
    )
    res.update(tfg_diffusion_defaults())
    return res


def guidance_defaults():#引导方法
    return dict(
        guidance_name="",
        recur_steps=1,
        iter_steps=1,
        guidance_strength=0.0,
        rho=1.0,
        mu=1.0,
        sigma=0.01,
        eps_bsz=4,
        rho_schedule="increase",
        mu_schedule="increase",
        sigma_schedule="decrease",
    )


def tfg_create_model_and_diffusion(#创建扩散模型
    dims,
    image_size,
    in_channels,
    class_cond,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    sigma_small,
    noise_schedule,
    predict_xstart,
    rescale_timesteps,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    task,
    guidance_kwargs,
):
    model = create_model(
        dims,
        image_size,
        in_channels,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=False,#不可更改，否则与权重不符
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_tfg_diffusion(
        task=task,
        guidance_kwargs=guidance_kwargs,
        steps=diffusion_steps,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
    )
    return model, diffusion


def create_tfg_diffusion(
    *,
    task="",                     # 任务类型（如图像修复）
    guidance_kwargs=None,        # 引导策略的超参数
    steps=1000,                  # 扩散总步数
    sigma_small=False,#False 8.1 # 是否使用小方差模式
    noise_schedule="linear",     # 噪声调度策略
    predict_xstart=False,        # 是否直接预测原始数据 
    rescale_timesteps=False,     # 是否重缩放时间步
):
    # 生成噪声调度序列（beta系数）
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    
    # 设置模型预测目标：epsilon噪声 或 原始数据x_start
    model_mean_type = (
        gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
    )
    
    # 设置方差模型类型：固定大方差 或 固定小方差
    model_var_type = (
        gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL
    )
    
    # 使用MSE作为损失函数
    loss_type = gd.LossType.MSE
    
    # 根据引导策略名称选择对应的扩散实现类
    guidance_name = guidance_kwargs["guidance_name"]
    if guidance_name == "no":
        GuidanceDiffusion = BaseGuidanceDiffusion  # 无引导的基础扩散
    elif guidance_name == "dps":
        GuidanceDiffusion = DPSGuidanceDiffusion   # DPS引导策略
    elif guidance_name == "tfg":
        GuidanceDiffusion = TFGGuidanceDiffusion   # 本文提出的TFG策略
    else:
        raise NotImplementedError
    
    # 实例化选择的扩散类
    return GuidanceDiffusion(
        task=task,
        guidance_kwargs=guidance_kwargs,
        betas=betas,              # 预计算的噪声调度序列
        model_mean_type=model_mean_type,  # 预测目标类型
        model_var_type=model_var_type,    # 方差类型
        loss_type=loss_type,      # 损失函数类型
        rescale_timesteps=rescale_timesteps,  # 时间步缩放控制
    )
