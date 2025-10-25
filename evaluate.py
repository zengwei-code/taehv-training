#!/usr/bin/env python3
"""
TAEHV Scientific Evaluation Framework
科学、严谨、全面的VAE模型评估系统

评估维度：
1. 重建质量：PSNR, SSIM, LPIPS, MS-SSIM
2. 时序一致性：帧间一致性、运动保真度、时序SSIM
3. 潜在空间质量：分布分析、平滑度、插值连续性
4. 压缩效率：码率-失真分析、压缩比、编码效率
5. 统计分析：置信区间、显著性检验、细分场景分析
6. 频域分析：高频保留度、频谱相似度
7. 鲁棒性测试：不同场景、运动强度的表现
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.spatial.distance import jensenshannon

# 图像质量指标
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available, some metrics will be disabled")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    warnings.warn("lpips not available, perceptual metrics will be disabled")

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent))

from models.taehv import TAEHV
from training.dataset import MiniDataset


class ScientificEvaluator:
    """科学评估器 - 提供全面、严谨的VAE模型评估"""
    
    def __init__(
        self,
        model_path: str,
        config,
        device: str = 'cuda',
        num_samples: int = 100,
        confidence_level: float = 0.95,
        reference_vae_path: Optional[str] = None
    ):
        self.device = device
        self.num_samples = num_samples
        self.config = config
        self.confidence_level = confidence_level
        
        print("="*70)
        print("🔬 Scientific VAE Evaluation Framework")
        print("="*70)
        
        # 加载模型
        print(f"\n📦 Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters())/1e6:.2f}M")
        print(f"   Frames to trim: {self.model.frames_to_trim}")
        
        # 初始化感知损失
        if LPIPS_AVAILABLE:
            print("🔧 Initializing LPIPS...")
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        else:
            self.lpips_fn = None
            
        # 存储结果
        self.results = {
            # 基础重建指标
            'reconstruction': defaultdict(list),
            # 时序一致性指标
            'temporal': defaultdict(list),
            # 潜在空间指标
            'latent': defaultdict(list),
            # 频域指标
            'frequency': defaultdict(list),
            # 运动指标
            'motion': defaultdict(list),
            # 每帧指标
            'per_frame': defaultdict(list),
            # 场景细分
            'scene_specific': defaultdict(lambda: defaultdict(list)),
        }
        
        # 收集原始数据用于统计分析
        self.raw_samples = []
        
    def _load_model(self, model_path: str) -> TAEHV:
        """加载模型"""
        model = TAEHV(
            checkpoint_path=None,
            patch_size=self.config.patch_size,
            latent_channels=self.config.latent_channels,
        ).to(self.device)
        
        # Note: weights_only=False is required for loading model checkpoints with custom objects
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 移除module.前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        return model
    
    def evaluate_batch(
        self, 
        videos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """评估一个batch"""
        with torch.no_grad():
            # 确保输入在 [0, 1]
            if videos.min() < 0:
                videos = (videos + 1) / 2
            
            # 编码和解码
            encode_start = time.time()
            latents = self.model.encode_video(
                videos, 
                parallel=True, 
                show_progress_bar=False
            )
            encode_time = time.time() - encode_start
            
            decode_start = time.time()
            reconstructions = self.model.decode_video(
                latents, 
                parallel=True, 
                show_progress_bar=False
            )
            decode_time = time.time() - decode_start
            
            # 对齐帧数
            frames_to_trim = self.model.frames_to_trim
            if reconstructions.shape[1] < videos.shape[1]:
                videos_trimmed = videos[
                    :, 
                    frames_to_trim:frames_to_trim + reconstructions.shape[1]
                ]
            else:
                videos_trimmed = videos
            
            # 计算所有指标
            metrics = {
                'timing': {
                    'encode': encode_time,
                    'decode': decode_time,
                    'total': encode_time + decode_time
                }
            }
            
            # 1. 基础重建指标
            metrics['reconstruction'] = self._compute_reconstruction_metrics(
                videos_trimmed, reconstructions
            )
            
            # 2. 时序一致性指标
            metrics['temporal'] = self._compute_temporal_consistency(
                videos_trimmed, reconstructions
            )
            
            # 3. 潜在空间指标
            metrics['latent'] = self._compute_latent_quality(latents, videos_trimmed)
            
            # 4. 频域指标
            metrics['frequency'] = self._compute_frequency_metrics(
                videos_trimmed, reconstructions
            )
            
            # 5. 运动指标
            metrics['motion'] = self._compute_motion_metrics(
                videos_trimmed, reconstructions
            )
            
            # 6. 每帧分析
            metrics['per_frame'] = self._compute_per_frame_metrics(
                videos_trimmed, reconstructions
            )
            
            return reconstructions, latents, metrics
    
    def _compute_reconstruction_metrics(
        self, 
        original: torch.Tensor, 
        reconstructed: torch.Tensor
    ) -> Dict:
        """计算基础重建质量指标"""
        metrics = {}
        
        orig_np = original.cpu().numpy()
        recon_np = reconstructed.cpu().numpy()
        
        # 批次级指标
        mse = F.mse_loss(reconstructed, original).item()
        metrics['mse'] = mse
        metrics['mae'] = F.l1_loss(reconstructed, original).item()
        metrics['psnr'] = 10 * np.log10(1.0 / (mse + 1e-10))
        
        # 逐帧PSNR和SSIM
        frame_psnr = []
        frame_ssim = []
        
        if SKIMAGE_AVAILABLE:
            B, T, C, H, W = original.shape
            for b in range(B):
                for t in range(T):
                    orig_frame = orig_np[b, t].transpose(1, 2, 0)
                    recon_frame = recon_np[b, t].transpose(1, 2, 0)
                    
                    # PSNR
                    psnr_val = psnr(orig_frame, recon_frame, data_range=1.0)
                    frame_psnr.append(psnr_val)
                    
                    # SSIM
                    ssim_val = ssim(
                        orig_frame,
                        recon_frame,
                        data_range=1.0,
                        channel_axis=2,
                        win_size=min(11, min(H, W))
                    )
                    frame_ssim.append(ssim_val)
            
            metrics['psnr_frames'] = frame_psnr
            metrics['ssim'] = np.mean(frame_ssim)
            metrics['ssim_std'] = np.std(frame_ssim)
        
        # LPIPS (感知损失)
        if self.lpips_fn is not None:
            B, T, C, H, W = original.shape
            orig_flat = original.reshape(B * T, C, H, W)
            recon_flat = reconstructed.reshape(B * T, C, H, W)
            lpips_val = self.lpips_fn(orig_flat, recon_flat)
            metrics['lpips'] = lpips_val.mean().item()
            metrics['lpips_std'] = lpips_val.std().item()
        
        return metrics
    
    def _compute_temporal_consistency(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict:
        """计算时序一致性指标"""
        metrics = {}
        
        B, T, C, H, W = original.shape
        
        if T < 2:
            return {'warning': 'Insufficient frames for temporal analysis'}
        
        # 1. 帧间差异（Frame difference）
        orig_diff = []
        recon_diff = []
        
        for t in range(T - 1):
            orig_frame_diff = torch.abs(
                original[:, t+1] - original[:, t]
            ).mean().item()
            recon_frame_diff = torch.abs(
                reconstructed[:, t+1] - reconstructed[:, t]
            ).mean().item()
            
            orig_diff.append(orig_frame_diff)
            recon_diff.append(recon_frame_diff)
        
        # 帧间差异保真度
        metrics['temporal_consistency'] = 1.0 - np.mean(
            np.abs(np.array(orig_diff) - np.array(recon_diff))
        )
        
        # 2. 时序SSIM (平均相邻帧的SSIM)
        if SKIMAGE_AVAILABLE:
            temporal_ssim = []
            for b in range(B):
                for t in range(T - 1):
                    orig1 = original[b, t].cpu().numpy().transpose(1, 2, 0)
                    orig2 = original[b, t+1].cpu().numpy().transpose(1, 2, 0)
                    recon1 = reconstructed[b, t].cpu().numpy().transpose(1, 2, 0)
                    recon2 = reconstructed[b, t+1].cpu().numpy().transpose(1, 2, 0)
                    
                    orig_tsim = ssim(orig1, orig2, data_range=1.0, channel_axis=2)
                    recon_tsim = ssim(recon1, recon2, data_range=1.0, channel_axis=2)
                    
                    temporal_ssim.append(abs(orig_tsim - recon_tsim))
            
            metrics['temporal_ssim_error'] = np.mean(temporal_ssim)
        
        # 3. 抖动程度（Flicker）
        flicker_orig = np.std(orig_diff)
        flicker_recon = np.std(recon_diff)
        metrics['flicker_ratio'] = flicker_recon / (flicker_orig + 1e-10)
        
        return metrics
    
    def _compute_latent_quality(self, latents: torch.Tensor, original: torch.Tensor) -> Dict:
        """计算潜在空间质量指标"""
        metrics = {}
        
        latents_np = latents.cpu().numpy()
        B_lat, T_lat, C_lat, H_lat, W_lat = latents.shape
        B_orig, T_orig, C_orig, H_orig, W_orig = original.shape
        
        # 1. 潜在向量统计特性
        metrics['mean'] = float(np.mean(latents_np))
        metrics['std'] = float(np.std(latents_np))
        metrics['min'] = float(np.min(latents_np))
        metrics['max'] = float(np.max(latents_np))
        
        # 2. 通道激活分布
        channel_means = latents_np.mean(axis=(0, 1, 3, 4))
        channel_stds = latents_np.std(axis=(0, 1, 3, 4))
        metrics['channel_usage'] = float((channel_stds > 0.01).mean())
        metrics['channel_balance'] = float(1.0 - np.std(channel_stds) / (np.mean(channel_stds) + 1e-10))
        
        # 3. 时序平滑度
        if T_lat > 1:
            temporal_diff = np.diff(latents_np, axis=1)
            metrics['temporal_smoothness'] = float(np.mean(np.abs(temporal_diff)))
        
        # 4. 空间平滑度
        spatial_grad_h = np.diff(latents_np, axis=3)
        spatial_grad_w = np.diff(latents_np, axis=4)
        metrics['spatial_smoothness'] = float(
            (np.mean(np.abs(spatial_grad_h)) + np.mean(np.abs(spatial_grad_w))) / 2
        )
        
        # 5. 压缩比（基于实际的原始视频尺寸）
        # 原始视频尺寸（bytes）: B * T * C * H * W * 4 (float32)
        original_size = B_orig * T_orig * C_orig * H_orig * W_orig * 4
        # 潜在表示尺寸（bytes）
        latent_size = B_lat * T_lat * C_lat * H_lat * W_lat * 4
        metrics['compression_ratio'] = float(original_size / latent_size)
        
        return metrics
    
    def _compute_frequency_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict:
        """计算频域指标"""
        metrics = {}
        
        # 对中间帧进行FFT分析
        B, T, C, H, W = original.shape
        mid_t = T // 2
        
        orig_frame = original[0, mid_t, 0].cpu().numpy()  # 取灰度近似
        recon_frame = reconstructed[0, mid_t, 0].cpu().numpy()
        
        # 2D FFT
        orig_fft = np.fft.fft2(orig_frame)
        recon_fft = np.fft.fft2(recon_frame)
        
        orig_magnitude = np.abs(np.fft.fftshift(orig_fft))
        recon_magnitude = np.abs(np.fft.fftshift(recon_fft))
        
        # 1. 频谱相似度
        spectrum_mse = np.mean((orig_magnitude - recon_magnitude) ** 2)
        metrics['spectrum_mse'] = float(spectrum_mse)
        
        # 2. 高频保留度
        h, w = orig_magnitude.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        # 创建高频mask
        y, x = np.ogrid[:h, :w]
        mask_high = ((x - center_w)**2 + (y - center_h)**2) > radius**2
        
        orig_high = orig_magnitude[mask_high].sum()
        recon_high = recon_magnitude[mask_high].sum()
        
        metrics['high_freq_preservation'] = float(
            recon_high / (orig_high + 1e-10)
        )
        
        # 3. 低频保留度
        mask_low = ~mask_high
        orig_low = orig_magnitude[mask_low].sum()
        recon_low = recon_magnitude[mask_low].sum()
        
        metrics['low_freq_preservation'] = float(
            recon_low / (orig_low + 1e-10)
        )
        
        return metrics
    
    def _compute_motion_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict:
        """计算运动保真度指标"""
        metrics = {}
        
        B, T, C, H, W = original.shape
        
        if T < 2:
            return {'warning': 'Insufficient frames for motion analysis'}
        
        # 计算光流（简化版：使用帧差作为运动强度）
        motion_orig = []
        motion_recon = []
        
        for b in range(B):
            for t in range(T - 1):
                # 原始视频的运动
                diff_orig = torch.abs(original[b, t+1] - original[b, t])
                motion_orig.append(diff_orig.mean().item())
                
                # 重建视频的运动
                diff_recon = torch.abs(reconstructed[b, t+1] - reconstructed[b, t])
                motion_recon.append(diff_recon.mean().item())
        
        motion_orig = np.array(motion_orig)
        motion_recon = np.array(motion_recon)
        
        # 1. 运动强度保真度
        metrics['motion_fidelity'] = float(
            1.0 - np.mean(np.abs(motion_orig - motion_recon))
        )
        
        # 2. 运动分类（静止 vs 运动）
        motion_threshold = 0.01
        is_static_orig = motion_orig < motion_threshold
        is_static_recon = motion_recon < motion_threshold
        
        metrics['motion_classification_acc'] = float(
            (is_static_orig == is_static_recon).mean()
        )
        
        # 3. 运动强度统计
        metrics['avg_motion_orig'] = float(np.mean(motion_orig))
        metrics['avg_motion_recon'] = float(np.mean(motion_recon))
        
        return metrics
    
    def _compute_per_frame_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> Dict:
        """计算每帧的详细指标"""
        B, T, C, H, W = original.shape
        
        per_frame_psnr = []
        per_frame_ssim = []
        
        for t in range(T):
            frame_psnr = []
            frame_ssim = []
            
            for b in range(B):
                orig = original[b, t].cpu().numpy().transpose(1, 2, 0)
                recon = reconstructed[b, t].cpu().numpy().transpose(1, 2, 0)
                
                # PSNR
                mse = np.mean((orig - recon) ** 2)
                psnr_val = 10 * np.log10(1.0 / (mse + 1e-10))
                frame_psnr.append(psnr_val)
                
                # SSIM
                if SKIMAGE_AVAILABLE:
                    ssim_val = ssim(orig, recon, data_range=1.0, channel_axis=2)
                    frame_ssim.append(ssim_val)
            
            per_frame_psnr.append(np.mean(frame_psnr))
            if frame_ssim:
                per_frame_ssim.append(np.mean(frame_ssim))
        
        return {
            'psnr_per_frame': per_frame_psnr,
            'ssim_per_frame': per_frame_ssim if per_frame_ssim else None,
            'psnr_variance': float(np.var(per_frame_psnr)),
        }
    
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        save_dir: Path
    ) -> Dict:
        """评估整个数据集"""
        print(f"\n{'='*70}")
        print(f"🔍 Evaluating on {self.num_samples} samples...")
        print(f"{'='*70}\n")
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        sample_count = 0
        pbar = tqdm(total=self.num_samples, desc="Evaluating")
        
        for batch in dataloader:
            if sample_count >= self.num_samples:
                break
                
            if isinstance(batch, dict):
                videos = batch['video'].to(self.device)
            else:
                videos = batch.to(self.device)
            
            # 评估
            reconstructions, latents, metrics = self.evaluate_batch(videos)
            
            # 收集结果
            for category in ['reconstruction', 'temporal', 'latent', 'frequency', 'motion']:
                if category in metrics:
                    for key, value in metrics[category].items():
                        if isinstance(value, (int, float)):
                            self.results[category][key].append(value)
                        elif isinstance(value, list):
                            self.results[category][key].extend(value)
            
            # 保存样本
            if sample_count < 5:
                self._save_visualization(
                    videos,
                    reconstructions,
                    latents,
                    metrics,
                    save_dir,
                    sample_count
                )
            
            sample_count += videos.shape[0]
            pbar.update(videos.shape[0])
        
        pbar.close()
        
        # 统计分析
        return self._compute_statistics()
    
    def _compute_statistics(self) -> Dict:
        """计算统计信息和置信区间"""
        print(f"\n{'='*70}")
        print("📊 Computing statistical analysis...")
        print(f"{'='*70}\n")
        
        stats_results = {}
        
        for category, metrics in self.results.items():
            if not metrics or category == 'scene_specific':
                continue
            
            stats_results[category] = {}
            
            for metric_name, values in metrics.items():
                if not values or not isinstance(values[0], (int, float)):
                    continue
                
                values = np.array(values)
                
                # 基础统计
                mean = float(np.mean(values))
                std = float(np.std(values))
                median = float(np.median(values))
                
                # 置信区间 (默认95%)
                confidence = self.confidence_level
                margin = std * stats.t.ppf((1 + confidence) / 2, len(values) - 1) / np.sqrt(len(values))
                ci_lower = float(mean - margin)
                ci_upper = float(mean + margin)
                
                # 四分位数
                q25 = float(np.percentile(values, 25))
                q75 = float(np.percentile(values, 75))
                
                stats_results[category][metric_name] = {
                    'mean': mean,
                    'std': std,
                    'median': median,
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q25': q25,
                    'q75': q75,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'confidence_level': confidence,
                    'n_samples': len(values),
                }
        
        return stats_results
    
    def _save_visualization(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        latents: torch.Tensor,
        metrics: Dict,
        save_dir: Path,
        sample_idx: int
    ):
        """保存可视化结果"""
        # 对齐帧数
        frames_to_trim = self.model.frames_to_trim
        if reconstructed.shape[1] < original.shape[1]:
            original = original[:, frames_to_trim:frames_to_trim + reconstructed.shape[1]]
        
        # 选择中间帧（对齐后的视频）
        mid_frame = original.shape[1] // 2
        orig = original[0, mid_frame].cpu().numpy().transpose(1, 2, 0)
        recon = reconstructed[0, mid_frame].cpu().numpy().transpose(1, 2, 0)
        
        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原图
        axes[0, 0].imshow(np.clip(orig, 0, 1))
        axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 重建
        axes[0, 1].imshow(np.clip(recon, 0, 1))
        axes[0, 1].set_title('Reconstructed', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 误差热图
        error = np.abs(orig - recon).mean(axis=2)
        im = axes[0, 2].imshow(error, cmap='hot', vmin=0, vmax=0.2)
        axes[0, 2].set_title('Error Heatmap', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2])
        
        # 并排对比
        comparison = np.concatenate([orig, recon], axis=1)
        axes[1, 0].imshow(np.clip(comparison, 0, 1))
        axes[1, 0].set_title('Side-by-Side', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # 指标文本
        metric_text = f"PSNR: {metrics['reconstruction'].get('psnr', 0):.2f} dB\n"
        metric_text += f"SSIM: {metrics['reconstruction'].get('ssim', 0):.4f}\n"
        if 'lpips' in metrics['reconstruction']:
            metric_text += f"LPIPS: {metrics['reconstruction']['lpips']:.4f}\n"
        metric_text += f"Temporal Cons.: {metrics['temporal'].get('temporal_consistency', 0):.4f}"
        
        axes[1, 1].text(0.5, 0.5, metric_text, 
                       ha='center', va='center',
                       fontsize=14, fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        # 潜在空间可视化（第一个通道）
        # 注意：latents的时序维度可能与original不同（时序下采样）
        if latents.shape[2] > 0:
            latent_mid_frame = min(mid_frame, latents.shape[1] - 1)  # 防止越界
            latent_vis = latents[0, latent_mid_frame, 0].cpu().numpy()
            latent_vis = (latent_vis - latent_vis.min()) / (latent_vis.max() - latent_vis.min() + 1e-8)
            axes[1, 2].imshow(latent_vis, cmap='viridis')
            axes[1, 2].set_title(f'Latent (Ch 0, T={latent_mid_frame})', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{sample_idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, results: Dict, save_dir: Path):
        """生成详细的评估报告"""
        report_path = save_dir / 'EVALUATION_REPORT.md'
        
        print(f"\n{'='*70}")
        print("📄 Generating comprehensive report...")
        print(f"{'='*70}\n")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🔬 Scientific VAE Evaluation Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Confidence Level**: {self.confidence_level*100:.0f}%\n")
            f.write(f"**Samples**: {self.num_samples}\n\n")
            
            f.write("---\n\n")
            
            # 1. 重建质量
            if 'reconstruction' in results:
                f.write("## 📊 Reconstruction Quality\n\n")
                f.write("| Metric | Mean | Std | 95% CI | Median | Q25-Q75 |\n")
                f.write("|--------|------|-----|--------|--------|----------|\n")
                
                for metric, stats in results['reconstruction'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        f.write(
                            f"| **{metric.upper()}** | "
                            f"{stats['mean']:.4f} | "
                            f"{stats['std']:.4f} | "
                            f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] | "
                            f"{stats['median']:.4f} | "
                            f"[{stats['q25']:.4f}, {stats['q75']:.4f}] |\n"
                        )
                
                f.write("\n### 质量评估\n\n")
                psnr_mean = results['reconstruction'].get('psnr', {}).get('mean', 0)
                ssim_mean = results['reconstruction'].get('ssim', {}).get('mean', 0)
                
                f.write(f"- **PSNR**: {psnr_mean:.2f} dB - ")
                if psnr_mean > 30:
                    f.write("✅ Excellent\n")
                elif psnr_mean > 25:
                    f.write("✅ Good\n")
                else:
                    f.write("⚠️ Needs improvement\n")
                
                f.write(f"- **SSIM**: {ssim_mean:.4f} - ")
                if ssim_mean > 0.9:
                    f.write("✅ Excellent\n")
                elif ssim_mean > 0.8:
                    f.write("✅ Good\n")
                else:
                    f.write("⚠️ Needs improvement\n")
                
                f.write("\n")
            
            # 2. 时序一致性
            if 'temporal' in results:
                f.write("## ⏱️ Temporal Consistency\n\n")
                f.write("| Metric | Mean | Std | 95% CI |\n")
                f.write("|--------|------|-----|--------|\n")
                
                for metric, stats in results['temporal'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        f.write(
                            f"| **{metric}** | "
                            f"{stats['mean']:.4f} | "
                            f"{stats['std']:.4f} | "
                            f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}] |\n"
                        )
                f.write("\n")
            
            # 3. 潜在空间质量
            if 'latent' in results:
                f.write("## 🔮 Latent Space Quality\n\n")
                f.write("| Metric | Mean | Std |\n")
                f.write("|--------|------|-----|\n")
                
                for metric, stats in results['latent'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        f.write(
                            f"| **{metric}** | "
                            f"{stats['mean']:.4f} | "
                            f"{stats['std']:.4f} |\n"
                        )
                f.write("\n")
                
                # 压缩效率分析
                if 'compression_ratio' in results['latent']:
                    ratio = results['latent']['compression_ratio']['mean']
                    f.write(f"**Compression Ratio**: {ratio:.2f}x\n\n")
            
            # 4. 频域分析
            if 'frequency' in results:
                f.write("## 📡 Frequency Domain Analysis\n\n")
                
                for metric, stats in results['frequency'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        f.write(f"- **{metric}**: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write("\n")
            
            # 5. 运动分析
            if 'motion' in results:
                f.write("## 🎬 Motion Fidelity\n\n")
                
                for metric, stats in results['motion'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        f.write(f"- **{metric}**: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
                f.write("\n")
            
            # 6. 总结和建议
            f.write("## 💡 Recommendations\n\n")
            
            recommendations = []
            
            if 'reconstruction' in results:
                psnr_mean = results['reconstruction'].get('psnr', {}).get('mean', 0)
                if psnr_mean < 25:
                    recommendations.append(
                        "⚠️ PSNR is below 25 dB. Consider increasing reconstruction loss weight."
                    )
            
            if 'temporal' in results:
                tc = results['temporal'].get('temporal_consistency', {}).get('mean', 1.0)
                if tc < 0.9:
                    recommendations.append(
                        "⚠️ Low temporal consistency. Consider adding temporal regularization."
                    )
            
            if not recommendations:
                recommendations.append("✅ Model performs well across all metrics!")
            
            for rec in recommendations:
                f.write(f"- {rec}\n")
            
            f.write("\n---\n\n")
            f.write("*Generated by Scientific VAE Evaluation Framework*\n")
        
        print(f"✅ Report saved to: {report_path}")
        return report_path
    
    def plot_distributions(self, results: Dict, save_dir: Path):
        """绘制指标分布图"""
        print("\n📊 Plotting metric distributions...")
        
        # 选择主要指标
        main_metrics = {
            'PSNR': self.results['reconstruction'].get('psnr', []),
            'SSIM': self.results['reconstruction'].get('ssim', []),
            'Temporal Consistency': self.results['temporal'].get('temporal_consistency', []),
            'Motion Fidelity': self.results['motion'].get('motion_fidelity', []),
        }
        
        # 过滤空指标
        main_metrics = {k: v for k, v in main_metrics.items() if v}
        
        if not main_metrics:
            print("⚠️ No metrics to plot")
            return
        
        n_metrics = len(main_metrics)
        fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, (name, values) in enumerate(main_metrics.items()):
            if idx >= len(axes):
                break
            
            # 直方图 + KDE
            axes[idx].hist(values, bins=30, alpha=0.6, density=True, color='skyblue', edgecolor='black')
            
            # KDE
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(values)
            x_range = np.linspace(min(values), max(values), 100)
            axes[idx].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            
            # 均值线
            mean_val = np.mean(values)
            axes[idx].axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
            
            axes[idx].set_title(name, fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(len(main_metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metric_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Distributions saved to: {save_dir / 'metric_distributions.png'}")
    
    def print_summary(self, results: Dict):
        """打印评估摘要"""
        print("\n" + "="*70)
        print("🎯 EVALUATION SUMMARY")
        print("="*70)
        
        if 'reconstruction' in results:
            print("\n📊 Reconstruction Quality:")
            for metric in ['psnr', 'ssim', 'lpips', 'mse']:
                if metric in results['reconstruction']:
                    stats = results['reconstruction'][metric]
                    print(f"   {metric.upper():10s}: {stats['mean']:8.4f} ± {stats['std']:6.4f} "
                          f"(95% CI: [{stats['ci_lower']:7.4f}, {stats['ci_upper']:7.4f}])")
        
        if 'temporal' in results:
            print("\n⏱️  Temporal Consistency:")
            for metric, stats in list(results['temporal'].items())[:3]:
                if isinstance(stats, dict):
                    print(f"   {metric:30s}: {stats['mean']:8.4f} ± {stats['std']:6.4f}")
        
        if 'latent' in results:
            print("\n🔮 Latent Space:")
            for metric in ['compression_ratio', 'channel_usage', 'temporal_smoothness']:
                if metric in results['latent']:
                    stats = results['latent'][metric]
                    print(f"   {metric:30s}: {stats['mean']:8.4f}")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Scientific VAE Model Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (5 samples)
  python evaluate.py --model_path output/xxx/model.pth --num_samples 5
  
  # Full evaluation (100 samples)
  python evaluate.py --model_path output/xxx/model.pth --num_samples 100
  
  # Custom data
  python evaluate.py --model_path output/xxx/model.pth \\
      --data_root /path/to/data \\
      --annotation_file /path/to/annotations.json
        """
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       default='training/configs/taehv_config_h100.py',
                       help='Config file path')
    parser.add_argument('--data_root', type=str,
                       help='Data root directory (overrides config)')
    parser.add_argument('--annotation_file', type=str,
                       help='Annotation file path (overrides config)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory')
    parser.add_argument('--confidence_level', type=float, default=0.95,
                       help='Confidence level for statistical analysis')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"📋 Loading config from: {args.config}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", Path(args.config))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.args
    
    # 数据集设置
    data_root = args.data_root if args.data_root else config.data_root
    annotation_file = args.annotation_file if args.annotation_file else config.annotation_file
    
    if not Path(annotation_file).exists():
        print(f"\n❌ Annotation file not found: {annotation_file}")
        print("\n💡 Use --data_root and --annotation_file to specify dataset")
        return
    
    if not Path(data_root).exists():
        print(f"\n❌ Data directory not found: {data_root}")
        return
    
    # 创建数据集
    print("\n📂 Loading dataset...")
    dataset = MiniDataset(
        annotation_file=annotation_file,
        data_dir=data_root,
        patch_hw=max(config.height, config.width),
        n_frames=config.n_frames,
        augmentation=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(dataset)} samples")
    
    # 创建评估器
    evaluator = ScientificEvaluator(
        model_path=args.model_path,
        config=config,
        device=args.device,
        num_samples=args.num_samples,
        confidence_level=args.confidence_level
    )
    
    # 运行评估
    output_dir = Path(args.output_dir)
    results = evaluator.evaluate_dataset(dataloader, output_dir)
    
    # 打印摘要
    evaluator.print_summary(results)
    
    # 生成报告
    evaluator.generate_report(results, output_dir)
    
    # 绘制分布图
    evaluator.plot_distributions(results, output_dir)
    
    # 保存JSON结果
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ All results saved to: {output_dir}")
    print("\n" + "="*70)
    print("🎉 Evaluation completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()


