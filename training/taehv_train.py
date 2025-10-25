#!/usr/bin/env python3
"""
TAEHV训练脚本
基于accelerate + deepspeed的分布式训练
"""

import argparse
import datetime
import json
import logging
import math
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# 过滤不重要的警告
warnings.filterwarnings("ignore", message=".*expandable_segments not supported.*")
warnings.filterwarnings("ignore", message=".*kernel version.*below.*recommended.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp")
warnings.filterwarnings("ignore", message=".*NCCL_BLOCKING_WAIT is deprecated.*")
warnings.filterwarnings("ignore", message=".*Setting OMP_NUM_THREADS.*")
warnings.filterwarnings("ignore", message=".*pymp-.*", category=UserWarning)

# 过滤CUDA分配器警告
warnings.filterwarnings("ignore", message=".*expandable_segments not supported on this platform.*")
warnings.filterwarnings("ignore", message=".*CUDAAllocatorConfig.*expandable_segments.*")

# 设置环境变量减少CUDA警告
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# 添加颜色支持
class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""
    
    # 颜色定义
    COLORS = {
        'DEBUG': '\033[36m',      # 青色
        'INFO': '\033[32m',       # 绿色
        'WARNING': '\033[33m',    # 黄色
        'ERROR': '\033[31m',      # 红色
        'CRITICAL': '\033[35m',   # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # 获取基础格式化内容
        log_message = super().format(record)
        
        # 添加颜色
        if record.levelname in self.COLORS:
            log_message = f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"
        
        return log_message

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedType

import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available

from tqdm.auto import tqdm
import transformers

# 验证相关库
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 检查版本
check_min_version("0.21.0")

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.taehv import TAEHV
from training.dataset import MiniDataset
from training.training_utils import get_ref_vae

# 配置彩色日志
logger = get_logger(__name__, log_level="INFO")

# 设置日志级别，减少DEBUG信息
logging.getLogger("accelerate.tracking").setLevel(logging.WARNING)  # 减少TensorBoard DEBUG信息
logging.getLogger("transformers").setLevel(logging.WARNING)  # 减少transformers的WARNING

# 为主logger添加颜色支持
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logging.getLogger().handlers.clear()  # 清除默认处理器
logging.getLogger().addHandler(console_handler)
logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Logging directory",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Report to which service.",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank"
    )

    args = parser.parse_args()
    return args


def compute_validation_metrics(
    model,
    val_dataloader,
    device,
    lpips_fn,
    num_samples=20,
    logger=None,
    use_amp=True
):
    """
    计算验证集上的评估指标
    
    Args:
        model: TAEHV模型
        val_dataloader: 验证数据加载器
        device: 计算设备
        lpips_fn: LPIPS损失函数
        num_samples: 评估样本数
        logger: 日志记录器
        use_amp: 是否使用自动混合精度（修复bf16类型不匹配问题）
    
    Returns:
        dict: {
            'val/psnr': float,
            'val/ssim': float,
            'val/lpips': float,
            'val/psnr_std': float,
            'val/ssim_std': float,
        }
    """
    if logger:
        logger.info(f"Computing validation metrics on {num_samples} samples...")
    
    model.eval()
    
    psnr_list = []
    ssim_list = []
    lpips_list = []
    
    with torch.no_grad():
        sample_count = 0
        for batch in val_dataloader:
            if sample_count >= num_samples:
                break
            
            # 获取视频数据
            if isinstance(batch, dict):
                videos = batch['video'].to(device)
            else:
                videos = batch.to(device)
            
            # 确保输入在 [0, 1] 范围（TAEHV 要求）
            if videos.min() < 0:
                videos = (videos + 1) / 2  # [-1, 1] -> [0, 1]
            
            try:
                # ✅ 使用自动混合精度包裹，解决bf16类型不匹配问题
                with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                    # 编码-解码
                    latents = model.encode_video(videos, parallel=True, show_progress_bar=False)
                    reconstructions = model.decode_video(latents, parallel=True, show_progress_bar=False)
                
                # ✅ 在autocast后，转换回float32（scikit-image不支持bfloat16）
                reconstructions = reconstructions.float()
                
                # 对齐帧数（TAEHV会裁剪帧）
                frames_to_trim = getattr(model, 'frames_to_trim', 0)
                if reconstructions.shape[1] < videos.shape[1] and frames_to_trim > 0:
                    # 裁剪原始视频，使其与重建视频的帧数匹配
                    videos_trimmed = videos[:, frames_to_trim:frames_to_trim + reconstructions.shape[1]]
                else:
                    videos_trimmed = videos
                
                # 确保帧数匹配
                if videos_trimmed.shape[1] != reconstructions.shape[1]:
                    if logger:
                        logger.warning(f"Frame mismatch: original {videos_trimmed.shape[1]} vs recon {reconstructions.shape[1]}, skipping batch")
                    continue
                
                # 转换为numpy计算PSNR/SSIM（确保float32）
                orig_np = videos_trimmed.cpu().float().numpy()
                recon_np = reconstructions.cpu().numpy()
                
                # 逐帧计算PSNR和SSIM
                batch_size, n_frames = orig_np.shape[0], orig_np.shape[1]
                for b in range(batch_size):
                    for t in range(n_frames):
                        orig_frame = orig_np[b, t].transpose(1, 2, 0)  # CHW -> HWC
                        recon_frame = recon_np[b, t].transpose(1, 2, 0)
                        
                        # PSNR
                        psnr_val = psnr(orig_frame, recon_frame, data_range=1.0)
                        psnr_list.append(psnr_val)
                        
                        # SSIM
                        ssim_val = ssim(
                            orig_frame, 
                            recon_frame, 
                            data_range=1.0, 
                            channel_axis=2, 
                            win_size=11
                        )
                        ssim_list.append(ssim_val)
                
                # 批量计算LPIPS（在GPU上更快）
                if lpips_fn is not None:
                    B, T, C, H, W = videos_trimmed.shape
                    # 确保两个输入都是float32（LPIPS不支持bf16）
                    orig_flat = videos_trimmed.float().reshape(B * T, C, H, W)
                    recon_flat = reconstructions.float().reshape(B * T, C, H, W)
                    
                    # LPIPS必须在float32下计算，不使用autocast
                    lpips_vals = lpips_fn(orig_flat, recon_flat)
                    lpips_list.extend(lpips_vals.cpu().numpy().flatten().tolist())
                
                sample_count += batch_size
                
            except Exception as e:
                if logger:
                    logger.warning(f"Error processing batch: {e}, skipping")
                continue
    
    model.train()
    
    # 计算统计信息
    results = {
        'val/psnr': float(np.mean(psnr_list)),
        'val/ssim': float(np.mean(ssim_list)),
        'val/psnr_std': float(np.std(psnr_list)),
        'val/ssim_std': float(np.std(ssim_list)),
    }
    
    if lpips_list:
        results['val/lpips'] = float(np.mean(lpips_list))
    
    if logger:
        logger.info(f"Validation completed: PSNR={results['val/psnr']:.2f}, SSIM={results['val/ssim']:.4f}")
    
    return results


def main():
    args = parse_args()

    # 导入配置文件
    config_path = Path(args.config)
    sys.path.append(str(config_path.parent))
    config_module = config_path.stem
    config = __import__(config_module).args

    # 初始化accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, logging_dir=args.logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision or config.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # 设置logging - 启用DEBUG以查看Seraena维度调试信息
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 设置随机种子
    if config.seed is not None:
        set_seed(config.seed)

    # 创建输出目录
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    # 加载模型
    logger.info("Loading TAEHV model...")
    model = TAEHV(
        checkpoint_path=config.pretrained_model_path,
        patch_size=config.patch_size,
        latent_channels=config.latent_channels
    )

    # 启用梯度检查点
    if config.gradient_checkpointing:
        model.encoder.requires_grad_(True)
        model.decoder.requires_grad_(True)

    # 准备优化器
    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
        eps=config.epsilon,
    )

    # 准备数据集
    logger.info("Loading dataset...")
    train_dataset = MiniDataset(
        annotation_file=config.annotation_file,
        data_dir=config.data_root,
        patch_hw=config.height,
        n_frames=config.n_frames,
        min_frame_delta=config.min_frame_delta,
        max_frame_delta=config.max_frame_delta,
        augmentation=True
    )

    # 构建DataLoader参数（处理单进程/多进程模式的兼容性）
    train_dataloader_kwargs = {
        'shuffle': True,
        'batch_size': config.train_batch_size,
        'num_workers': config.dataloader_num_workers,
        'pin_memory': config.pin_memory,
        'drop_last': True,
    }
    
    # prefetch_factor 仅在多进程模式下有效
    if config.dataloader_num_workers > 0 and hasattr(config, 'prefetch_factor'):
        train_dataloader_kwargs['prefetch_factor'] = config.prefetch_factor
    
    train_dataloader = DataLoader(train_dataset, **train_dataloader_kwargs)

    # 创建验证数据集（使用相同配置但不做增强）
    logger.info("Creating validation dataset...")
    val_dataset = MiniDataset(
        annotation_file=config.annotation_file,
        data_dir=config.data_root,
        patch_hw=config.height,
        n_frames=config.n_frames,
        min_frame_delta=config.min_frame_delta,
        max_frame_delta=config.max_frame_delta,
        augmentation=False,  # 验证时不做数据增强
        cache_videos=False
    )
    
    # 限制验证集大小（避免过慢）
    if len(val_dataset.annotations) > 100:
        val_dataset.annotations = val_dataset.annotations[:100]
        logger.info(f"Limited validation dataset to 100 samples")
    else:
        logger.info(f"Validation dataset has {len(val_dataset.annotations)} samples")
    
    # 验证DataLoader与训练保持一致的模式（单进程/多进程）
    val_num_workers = 0 if config.dataloader_num_workers == 0 else min(2, config.dataloader_num_workers)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,  # 验证不需要shuffle
        batch_size=config.validation_batch_size,
        num_workers=val_num_workers,  # 与训练配置一致
        pin_memory=True,
        drop_last=False  # 验证不需要drop_last
    )
    logger.info(f"✅ Validation dataloader created (num_workers={val_num_workers})")

    # 准备学习率调度器
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        # num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        # num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=config.max_train_steps,
        num_cycles=config.lr_num_cycles,
    )

    # 准备参考VAE（如果需要）
    vae_ref = None
    if config.use_ref_vae:
        logger.info("Loading reference VAE...")
        vae_ref = get_ref_vae(
            device="cpu", 
            dtype=getattr(torch, config.ref_vae_dtype),
            cogvideox_model_path=config.cogvideox_model_path
        )
        vae_ref.requires_grad_(False)

    # 准备Seraena（如果需要）
    seraena = None
    if config.use_seraena:
        logger.info("Loading Seraena adversarial trainer...")
        from models.seraena import Seraena
        # 使用4通道匹配TAEHV实际的latent通道数
        seraena = Seraena(3 * config.n_seraena_frames, 4)

    # 准备accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if vae_ref is not None:
        # 参考VAE不需要训练，手动移动到设备而不是通过DeepSpeed准备
        vae_ref = vae_ref.to(accelerator.device)
    
    # 注册checkpoint保存hooks - 确保模型状态正确保存
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # 保存主模型状态
            model_to_save = accelerator.unwrap_model(model)
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pth"))
            logger.info(f"Saved model state dict to {output_dir}/model.pth")

    def load_model_hook(models, input_dir):
        # 模型加载逻辑（如果需要）
        pass
    
    # 注册hooks
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    if seraena is not None:
        # Seraena不需要训练，手动移动到设备而不是通过DeepSpeed准备
        seraena = seraena.to(accelerator.device)

    # 初始化LPIPS感知损失（只在主进程）
    lpips_fn = None
    if accelerator.is_main_process:
        logger.info("Initializing LPIPS metric for validation...")
        lpips_fn = lpips.LPIPS(net='alex').to(accelerator.device)
        lpips_fn.eval()  # 设置为eval模式
        logger.info("✅ LPIPS initialized")

    # 初始化最佳模型跟踪变量
    best_val_psnr = 0.0
    best_val_step = 0
    patience_counter = 0
    patience_limit = getattr(config, 'early_stopping_patience', 5)  # 默认5次验证无改善则停止

    # 验证历史记录
    validation_history = {
        'steps': [],
        'psnr': [],
        'ssim': [],
        'lpips': []
    }
    logger.info(f"Early stopping patience: {patience_limit} validations")

    # 计算总训练步数
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    if config.max_train_steps is None:
        config.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

    # 我们需要重新计算我们的训练轮数作为我们的训练步数的函数
    config.num_train_epochs = math.ceil(config.max_train_steps / num_update_steps_per_epoch)

    # 初始化tracker
    if accelerator.is_main_process:
        # 过滤配置，只保留TensorBoard支持的类型 (int, float, str, bool, torch.Tensor)
        filtered_config = {}
        for key, value in vars(config).items():
            if isinstance(value, (int, float, str, bool)) or (hasattr(value, 'dtype') and hasattr(value, 'device')):
                filtered_config[key] = value
            elif isinstance(value, dict):
                # 将字典转换为字符串
                filtered_config[key] = str(value)
            else:
                # 将其他类型转换为字符串
                filtered_config[key] = str(value)
        
        accelerator.init_trackers(config.tracker_name, config=filtered_config)

    # 训练循环
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = "
        f"{config.train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps}"
    )
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # 如果从检查点恢复
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            # 检查是否为完整路径（包含路径分隔符）
            if os.path.sep in config.resume_from_checkpoint or '/' in config.resume_from_checkpoint:
                # 完整路径，直接使用
                checkpoint_path = config.resume_from_checkpoint
                path = os.path.basename(config.resume_from_checkpoint)  # 用于提取步数
            else:
                # 只是checkpoint名称，需要组合路径
                path = config.resume_from_checkpoint
                checkpoint_path = os.path.join(config.output_dir, path)
        else:
            # 获取最新的检查点
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
            checkpoint_path = os.path.join(config.output_dir, path) if path else None

        # 检查checkpoint是否存在
        if path is None or not os.path.exists(checkpoint_path):
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(checkpoint_path)
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # 确保所有进程的模型都处于训练模式
    model.train()
    
    # 同步所有进程
    accelerator.wait_for_everyone()
    
    # 提示GPU内存监控已启用
    if accelerator.is_main_process:
        logger.info("GPU memory monitoring enabled (every 50 steps)")

    # 训练主循环
    for epoch in range(first_epoch, config.num_train_epochs):
        model.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # 打印第一个batch的信息（仅一次）
            if global_step == 0 and step == 0 and accelerator.is_main_process:
                logger.info(f"First batch shape: {batch.shape}")
            
            with accelerator.accumulate(model):
                # 数据预处理 - MiniDataset已归一化到[0,1]
                frames = batch.float()  # N,T,C,H,W, [0,1] - no need to divide by 255
                # 转换到与模型参数相同的类型（bf16）
                frames = frames.to(accelerator.device, dtype=torch.bfloat16)
                
                # 前向传播
                with accelerator.autocast():
                    # 编码
                    encoded = model.encode_video(frames, parallel=True, show_progress_bar=False)
                    
                    # 解码
                    # 检查CogVideoX模式是否会跳过裁剪
                    # 逻辑与models/taehv.py:228保持一致
                    will_skip_trim = model.is_cogvideox and encoded.shape[1] % 2 == 0
                    if will_skip_trim:
                        # CogVideoX模式且编码帧数为偶数，解码时不裁剪，target也不裁剪
                        frames_target = frames
                    else:
                        # 正常模式，target需要裁剪
                        frames_target = frames[:, :-model.frames_to_trim] if model.frames_to_trim > 0 else frames
                    
                    decoded = model.decode_video(encoded, parallel=True, show_progress_bar=False)
                    
                    # 重建损失
                    recon_loss = F.mse_loss(decoded, frames_target)
                    
                    # 编码损失（如果有参考VAE）
                    total_loss = recon_loss
                    if vae_ref is not None:
                        try:
                            with torch.no_grad():
                                ref_latent = vae_ref.encode_video(frames)
                            encoder_loss = F.mse_loss(encoded, ref_latent)
                            total_loss = (
                                config.reconstruction_loss_weight * recon_loss + 
                                config.encoder_loss_weight * encoder_loss
                            )
                        except Exception as e:
                            logger.warning(f"VAE encoding failed: {e}, using only reconstruction loss")
                            total_loss = recon_loss
                    
                    # Seraena对抗损失（如果启用）
                    if seraena is not None:
                        try:
                            # 重组为seraena输入格式
                            def pad_and_group(x):
                                # x: N, T, C, H, W -> N*T//n_seraena_frames, n_seraena_frames*C, H, W
                                logger.debug(f"pad_and_group input shape: {x.shape}")
                                
                                frames_to_trim = getattr(model, 'frames_to_trim', 0)
                                logger.debug(f"frames_to_trim: {frames_to_trim}")
                                
                                if frames_to_trim > 0:
                                    x_padded = torch.cat([x, x[:, :frames_to_trim]], 1)
                                else:
                                    x_padded = x
                                    
                                n, t, c, h, w = x_padded.shape
                                logger.debug(f"After padding: {x_padded.shape}")
                                logger.debug(f"n_seraena_frames: {config.n_seraena_frames}")
                                
                                # 确保帧数能被n_seraena_frames整除
                                if t % config.n_seraena_frames != 0:
                                    logger.warning(f"Frame count {t} is not divisible by n_seraena_frames {config.n_seraena_frames}")
                                    # 裁剪到最接近的可整除数
                                    t_trimmed = (t // config.n_seraena_frames) * config.n_seraena_frames
                                    x_padded = x_padded[:, :t_trimmed]
                                    t = t_trimmed
                                    logger.debug(f"Trimmed to: {x_padded.shape}")
                                
                                result = x_padded.reshape(n * t//config.n_seraena_frames, config.n_seraena_frames*c, h, w)
                                logger.debug(f"pad_and_group output shape: {result.shape}")
                                return result
                            
                            # 修复Seraena参数维度不匹配问题
                            # 问题：pad_and_group内部会padding帧数，但第三个参数没有考虑这个padding
                            
                            actual_frames = frames_target.shape[1]  # 原始帧数（16）
                            frames_to_trim = getattr(model, 'frames_to_trim', 0)  # 3
                            padded_frames = actual_frames + (frames_to_trim if frames_to_trim > 0 else 0)  # 16+3=19
                            
                            # 计算基于padding后帧数的repeat_times
                            repeat_times_padded = padded_frames // config.n_seraena_frames  # 19÷1=19
                            
                            # 调试信息已移除 - 训练正常后不再需要
                            
                            # 计算三个参数并输出维度信息
                            param1 = pad_and_group(frames_target)
                            param2 = pad_and_group(decoded)
                            
                            # 修复第三个参数：确保维度匹配padding后的帧数
                            # encoded: [batch, channels, ...] -> 需要扩展到匹配padded_frames × batch
                            batch_size = frames_target.shape[0]
                            target_first_dim = batch_size * repeat_times_padded  # 2×19=38
                            
                            # 修复第三个参数：确保形状为4D [N, C, H, W]，匹配Seraena期望
                            # 对时间维度求平均，保留空间维度: [batch, channels, T, H, W] -> [batch, channels, H, W]
                            encoded_spatial = encoded.mean(dim=2)  # [batch, channels, H, W]
                            
                            # TAEHV使用4通道latent，直接使用不需要调整
                            param3 = encoded_spatial  # [batch, 4, H, W]
                            
                            seraena_target, seraena_debug = seraena.step_and_make_correction_targets(
                                param1,
                                param2,
                                param3
                            )
                            
                            # 转回原格式 - 重新设计为与pad_and_group对称
                            def ungroup_and_unpad(x, original_shape):
                                """
                                将pad_and_group的输出转换回原始格式
                                x: [n_groups, grouped_c, h, w] - pad_and_group的输出
                                original_shape: [N, T, C, H, W] - 原始张量的形状
                                """
                                n_groups, grouped_c, h, w = x.shape
                                original_n, original_t, original_c, original_h, original_w = original_shape
                                
                                logger.debug(f"ungroup_and_unpad input shape: {x.shape}")
                                logger.debug(f"original_shape: {original_shape}")
                                
                                c = grouped_c // config.n_seraena_frames
                                frames_to_trim = getattr(model, 'frames_to_trim', 0)
                                
                                # 计算pad后的帧数
                                padded_frames = original_t + (frames_to_trim if frames_to_trim > 0 else 0)
                                
                                # 检查是否被裁剪过（确保能被n_seraena_frames整除）
                                actual_frames = padded_frames
                                if padded_frames % config.n_seraena_frames != 0:
                                    actual_frames = (padded_frames // config.n_seraena_frames) * config.n_seraena_frames
                                
                                logger.debug(f"padded_frames: {padded_frames}, actual_frames: {actual_frames}")
                                logger.debug(f"Expected n_groups: {original_n * actual_frames // config.n_seraena_frames}")
                                
                                # 验证维度一致性
                                expected_n_groups = original_n * actual_frames // config.n_seraena_frames
                                if n_groups != expected_n_groups:
                                    logger.warning(f"Group count mismatch: got {n_groups}, expected {expected_n_groups}")
                                    return None
                                
                                # reshape回原始格式
                                x_ungrouped = x.reshape(original_n, actual_frames, c, h, w)
                                logger.debug(f"After ungrouping: {x_ungrouped.shape}")
                                
                                # 如果之前有padding，现在需要去掉padding
                                if frames_to_trim > 0 and actual_frames > original_t:
                                    # 去掉padding的帧
                                    trim_amount = min(frames_to_trim, actual_frames - original_t)
                                    x_ungrouped = x_ungrouped[:, :-trim_amount]
                                    logger.debug(f"After removing padding: {x_ungrouped.shape}")
                                
                                return x_ungrouped
                            
                            # 使用原始张量的形状信息进行反向转换
                            seraena_target = ungroup_and_unpad(seraena_target, frames_target.shape)
                            if seraena_target is not None:
                                # 确保形状匹配
                                if seraena_target.shape == decoded.shape:
                                    seraena_loss = F.mse_loss(decoded, seraena_target)
                                    total_loss = total_loss + config.seraena_loss_weight * seraena_loss
                                    logger.debug(f"✅ Seraena loss computed: {seraena_loss.item():.6f}")
                                else:
                                    logger.warning(f"Shape mismatch: decoded {decoded.shape} vs seraena_target {seraena_target.shape}")
                            else:
                                logger.warning("Skipping Seraena loss due to dimension mismatch")
                            
                        except Exception as e:
                            logger.warning(f"Seraena training failed: {e}, skipping adversarial loss")

                # 反向传播 - 添加错误检测和处理
                # 检查loss是否为NaN或无穷大
                if not torch.isfinite(total_loss):
                    logger.warning(f"Loss is not finite: {total_loss}, skipping step")
                    optimizer.zero_grad()
                    continue
                
                try:
                    accelerator.backward(total_loss)
                    
                    if accelerator.sync_gradients:
                        # 梯度裁剪 - 参考stage1代码的处理方式
                        if accelerator.distributed_type != DistributedType.DEEPSPEED:
                            # 非DeepSpeed情况下手动裁剪梯度
                            grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                            # 检查梯度范数是否有限（使用math.isfinite处理Python float）
                            if not math.isfinite(grad_norm):
                                logger.warning(f"Gradient norm is not finite: {grad_norm}, skipping step")
                                optimizer.zero_grad()
                                continue
                            
                            # 记录梯度范数用于调试
                            if global_step % 50 == 0:
                                logger.info(f"Gradient norm: {grad_norm:.4f}")
                        # else: DeepSpeed会自动处理梯度裁剪
                            
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"GPU OOM at step {global_step}, skipping batch")
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise e
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # 检查accumulate梯度
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                # 详细的TensorBoard日志记录
                log_dict = {
                    "train/loss": total_loss.detach().item(),
                    "train/reconstruction_loss": recon_loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                }
                
                # 编码器损失（如果有参考VAE）
                if vae_ref is not None and 'encoder_loss' in locals():
                    log_dict["train/encoder_loss"] = encoder_loss.item()
                
                # Seraena损失（如果启用）
                if seraena is not None and 'seraena_loss' in locals():
                    log_dict["train/seraena_loss"] = seraena_loss.item()
                
                # 梯度范数记录（如果计算了）
                if 'grad_norm' in locals() and math.isfinite(grad_norm):
                    log_dict["train/gradient_norm"] = grad_norm
                
                # GPU内存使用
                if torch.cuda.is_available():
                    memory_gb = torch.cuda.memory_allocated() / (1024**3)
                    log_dict["train/gpu_memory_gb"] = memory_gb
                
                accelerator.log(log_dict, step=global_step)
                train_loss += total_loss.detach().item()
                
                # 定期内存管理和通信优化（每100步）
                if global_step % 100 == 0:
                    torch.cuda.empty_cache()
                    if hasattr(accelerator.state, 'deepspeed_plugin') and accelerator.state.deepspeed_plugin is not None:
                        import gc
                        gc.collect()
                
                # 内存监控（每50步，静默执行）
                if global_step % 50 == 0:
                    if torch.cuda.is_available():
                        memory_allocated = torch.cuda.memory_allocated() / 1024**3
                        memory_reserved = torch.cuda.memory_reserved() / 1024**3
                        # 仅在内存异常时打印（超过35GB reserved）
                        if memory_reserved > 35.0:
                            logger.warning(f"Step {global_step}: High GPU memory usage - Reserved: {memory_reserved:.2f}GB")

                # ============================================================
                # 验证步骤（每validation_steps执行一次）
                # ============================================================
                if global_step % config.validation_steps == 0 and global_step > 0:
                    logger.info(f"🔍 Running validation at step {global_step}")
                    
                    # 同步所有进程
                    accelerator.wait_for_everyone()
                    
                    # 只在主进程运行验证（避免重复计算）
                    if accelerator.is_main_process:
                        try:
                            # 计算验证指标
                            val_metrics = compute_validation_metrics(
                                model=accelerator.unwrap_model(model),
                                val_dataloader=val_dataloader,
                                device=accelerator.device,
                                lpips_fn=lpips_fn,
                                num_samples=config.num_validation_samples,
                                logger=logger,
                                use_amp=(accelerator.mixed_precision != "no")
                            )
                            
                            # 记录到TensorBoard
                            accelerator.log(val_metrics, step=global_step)
                            
                            # 保存到历史记录
                            validation_history['steps'].append(global_step)
                            validation_history['psnr'].append(val_metrics['val/psnr'])
                            validation_history['ssim'].append(val_metrics['val/ssim'])
                            if 'val/lpips' in val_metrics:
                                validation_history['lpips'].append(val_metrics['val/lpips'])
                            
                            # 打印验证结果
                            logger.info(f"✅ Validation Results (Step {global_step}):")
                            logger.info(f"   PSNR:  {val_metrics['val/psnr']:.2f} ± {val_metrics['val/psnr_std']:.2f} dB")
                            logger.info(f"   SSIM:  {val_metrics['val/ssim']:.4f}")
                            if 'val/lpips' in val_metrics:
                                logger.info(f"   LPIPS: {val_metrics['val/lpips']:.4f}")
                            
                            # ============================================================
                            # 最佳模型保存逻辑
                            # ============================================================
                            current_psnr = val_metrics['val/psnr']
                            
                            if current_psnr > best_val_psnr:
                                # 发现新的最佳模型
                                improvement = current_psnr - best_val_psnr
                                best_val_psnr = current_psnr
                                best_val_step = global_step
                                patience_counter = 0  # 重置早停计数器
                                
                                logger.info(f"🏆 New Best Model! PSNR improved by {improvement:.2f} dB")
                                logger.info(f"   Best PSNR: {best_val_psnr:.2f} dB at step {best_val_step}")
                                
                                # 保存最佳模型
                                best_model_dir = os.path.join(config.output_dir, "best_model")
                                
                                # 删除旧的最佳模型
                                if os.path.exists(best_model_dir):
                                    shutil.rmtree(best_model_dir)
                                    logger.info(f"   Removed old best model")
                                
                                # 保存新的最佳模型
                                os.makedirs(best_model_dir, exist_ok=True)
                                
                                # 1. 保存模型权重
                                model_to_save = accelerator.unwrap_model(model)
                                torch.save(
                                    model_to_save.state_dict(),
                                    os.path.join(best_model_dir, "model.pth")
                                )
                                
                                # 2. 保存元信息
                                best_model_info = {
                                    'step': global_step,
                                    'psnr': float(current_psnr),
                                    'ssim': float(val_metrics['val/ssim']),
                                    'train_loss': float(total_loss.detach().item()) if 'total_loss' in locals() else 0.0,
                                    'timestamp': datetime.datetime.now().isoformat(),
                                }
                                if 'val/lpips' in val_metrics:
                                    best_model_info['lpips'] = float(val_metrics['val/lpips'])
                                
                                with open(os.path.join(best_model_dir, "model_info.json"), 'w') as f:
                                    json.dump(best_model_info, f, indent=2)
                                
                                logger.info(f"   ✅ Saved best model to {best_model_dir}")
                                
                            else:
                                # 没有改善
                                patience_counter += 1
                                logger.info(f"⚠️  No improvement for {patience_counter} validation(s)")
                                logger.info(f"   Current PSNR: {current_psnr:.2f} dB")
                                logger.info(f"   Best PSNR: {best_val_psnr:.2f} dB (at step {best_val_step})")
                                
                                # ============================================================
                                # 早停检查
                                # ============================================================
                                if patience_counter >= patience_limit:
                                    logger.info(f"🛑 Early Stopping Triggered!")
                                    logger.info(f"   No improvement for {patience_limit} validations ({patience_limit * config.validation_steps} steps)")
                                    logger.info(f"   Best model: step {best_val_step}, PSNR {best_val_psnr:.2f} dB")
                                    
                                    # 保存早停信息
                                    early_stop_info = {
                                        'stopped_at_step': global_step,
                                        'best_step': best_val_step,
                                        'best_psnr': float(best_val_psnr),
                                        'patience_limit': patience_limit,
                                        'reason': 'early_stopping'
                                    }
                                    
                                    with open(os.path.join(config.output_dir, "early_stop_info.json"), 'w') as f:
                                        json.dump(early_stop_info, f, indent=2)
                                    
                                    logger.info("   Exiting training loop...")
                                    # 设置标志，外层循环检查后退出
                                    should_stop_training = True
                            
                            # 定期保存验证历史
                            if global_step % (config.validation_steps * 5) == 0:
                                validation_history_path = os.path.join(config.output_dir, "validation_history.json")
                                with open(validation_history_path, 'w') as f:
                                    json.dump(validation_history, f, indent=2)
                                logger.info(f"   Saved validation history to {validation_history_path}")
                            
                        except Exception as e:
                            logger.error(f"❌ Validation failed: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # 同步所有进程（等待主进程完成验证）
                    accelerator.wait_for_everyone()
                    
                    # 清理显存
                    torch.cuda.empty_cache()
                    
                    # 检查是否应该早停
                    if 'should_stop_training' in locals() and should_stop_training:
                        logger.info("Early stopping: breaking out of training loop")
                        break  # 退出训练循环

                # 保存检查点
                if global_step % config.checkpointing_steps == 0:
                    # 保存前的内存和通信清理
                    torch.cuda.empty_cache()  # 清理GPU内存
                    if accelerator.is_main_process:
                        logger.info(f"Starting checkpoint save at step {global_step}")
                    
                    # 同步所有进程，确保状态一致
                    accelerator.wait_for_everyone()
                    
                    try:
                        # 保存检查点 - accelerator.save_state()内部处理多进程协调
                        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        if accelerator.is_main_process:
                            logger.info(f"✅ Successfully saved checkpoint to {save_path}")
                        
                        # 保存后额外清理
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.error(f"❌ Failed to save checkpoint at step {global_step}: {e}")
                        # 继续训练而不是崩溃
                    
                    # 再次同步，确保checkpoint保存完成
                    accelerator.wait_for_everyone()
                    
                    # 删除旧的检查点
                    if accelerator.is_main_process and config.checkpoints_total_limit is not None:
                        try:
                            checkpoints = os.listdir(config.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # 删除最老的检查点
                            if len(checkpoints) > config.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - config.checkpoints_total_limit
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(config.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        except Exception as e:
                            logger.warning(f"Failed to cleanup old checkpoints: {e}")
                            # 继续训练，不因为清理失败而中断

                # 记录日志
                if global_step % config.log_every == 0:
                    progress_bar.set_postfix(loss=total_loss.detach().item())

            logs = {"step_loss": total_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= config.max_train_steps:
                break

    # 创建pipeline并保存
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 保存最终模型
        model_unwrapped = accelerator.unwrap_model(model)
        torch.save(model_unwrapped.state_dict(), os.path.join(config.output_dir, "final_model.pth"))
        logger.info(f"Saved final model to {config.output_dir}/final_model.pth")

    accelerator.end_training()


if __name__ == "__main__":
    main()
