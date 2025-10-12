"""
VAE Model Evaluation Script
科学评估 Tiny-VAE 模型的质量

评估维度：
1. 定量指标：PSNR, SSIM, LPIPS, MSE
2. 视觉质量：重建对比、误差热图
3. 潜在空间质量：压缩比、编码效率
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
import os

# Add models to path (parent directory contains models/ and training/)
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "training"))

from taehv import TAEHV
from dataset import MiniDataset


class VAEEvaluator:
    """VAE模型评估器"""
    
    def __init__(self, model_path, config, device='cuda', num_samples=100):
        self.device = device
        self.num_samples = num_samples
        self.config = config
        
        # 加载模型
        print(f"📦 Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 打印帧裁剪信息
        print(f"ℹ️  TAEHV will trim {self.model.frames_to_trim} frames from decoded videos")
        print(f"   (e.g., {config.n_frames} input frames -> {config.n_frames - self.model.frames_to_trim} output frames)")
        
        # 初始化LPIPS (感知损失)
        print("🔧 Initializing LPIPS metric...")
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
        # 存储结果
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'lpips': [],
            'mse': [],
            'mae': [],
        }
        
    def _load_model(self, model_path):
        """加载VAE模型"""
        # 初始化模型架构
        model = TAEHV(
            checkpoint_path=None,  # 不从预训练加载，后面会手动加载权重
            patch_size=self.config.patch_size,
            latent_channels=self.config.latent_channels,
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # 移除module.前缀（如果有）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Model loaded successfully")
        
        return model
    
    def evaluate_batch(self, videos):
        """
        评估一个batch的视频
        videos: [B, T, C, H, W] - 值范围应该是 [0, 1]
        """
        with torch.no_grad():
            # 确保输入在 [0, 1] 范围（TAEHV 要求）
            # 如果数据在 [-1, 1] 范围，需要转换
            if videos.min() < 0:
                videos = (videos + 1) / 2  # [-1, 1] -> [0, 1]
            
            # Encode - TAEHV 使用 encode_video
            latents = self.model.encode_video(videos, parallel=True, show_progress_bar=False)
            
            # Decode - TAEHV 使用 decode_video
            reconstructions = self.model.decode_video(latents, parallel=True, show_progress_bar=False)
            
            # 重要：TAEHV decode_video 会裁剪帧数！
            # 需要对原始视频进行相同的裁剪以对齐
            frames_to_trim = self.model.frames_to_trim
            if reconstructions.shape[1] < videos.shape[1]:
                # 裁剪原始视频，使其与重建视频的帧数匹配
                videos_trimmed = videos[:, frames_to_trim:frames_to_trim + reconstructions.shape[1]]
            else:
                videos_trimmed = videos
            
            # 计算指标
            batch_metrics = self._compute_metrics(videos_trimmed, reconstructions)
            
            return reconstructions, latents, batch_metrics
    
    def _compute_metrics(self, original, reconstructed):
        """计算各种评估指标"""
        # 转换到numpy用于某些指标
        orig_np = original.cpu().numpy()
        recon_np = reconstructed.cpu().numpy()
        
        batch_size = original.shape[0]
        batch_metrics = {k: [] for k in self.metrics.keys()}
        
        for b in range(batch_size):
            # 逐帧计算（对视频序列）
            for t in range(original.shape[1]):
                orig_frame = orig_np[b, t]
                recon_frame = recon_np[b, t]
                
                # 1. MSE (均方误差)
                mse = np.mean((orig_frame - recon_frame) ** 2)
                batch_metrics['mse'].append(mse)
                
                # 2. MAE (平均绝对误差)
                mae = np.mean(np.abs(orig_frame - recon_frame))
                batch_metrics['mae'].append(mae)
                
                # 3. PSNR (峰值信噪比)
                # 数据范围已经是 [0, 1]，直接使用
                # PSNR - 转置为 (H, W, C) 格式
                psnr_val = psnr(
                    orig_frame.transpose(1, 2, 0),
                    recon_frame.transpose(1, 2, 0),
                    data_range=1.0
                )
                batch_metrics['psnr'].append(psnr_val)
                
                # 4. SSIM (结构相似性)
                ssim_val = ssim(
                    orig_frame.transpose(1, 2, 0),
                    recon_frame.transpose(1, 2, 0),
                    data_range=1.0,
                    channel_axis=2,
                    win_size=11
                )
                batch_metrics['ssim'].append(ssim_val)
        
        # 5. LPIPS (感知相似性) - 在整个batch上计算
        with torch.no_grad():
            # Reshape: [B*T, C, H, W]
            B, T, C, H, W = original.shape
            orig_flat = original.reshape(B * T, C, H, W)
            recon_flat = reconstructed.reshape(B * T, C, H, W)
            
            lpips_val = self.lpips_fn(orig_flat, recon_flat)
            batch_metrics['lpips'].extend(lpips_val.cpu().numpy().flatten().tolist())
        
        return batch_metrics
    
    def evaluate_dataset(self, dataloader):
        """评估整个数据集"""
        print(f"\n🔍 Evaluating on {self.num_samples} samples...")
        
        sample_count = 0
        pbar = tqdm(total=self.num_samples, desc="Evaluating")
        
        for batch in dataloader:
            if sample_count >= self.num_samples:
                break
                
            # MiniDataset 返回的是 tensor，不是字典
            if isinstance(batch, dict):
                videos = batch['video'].to(self.device)
            else:
                videos = batch.to(self.device)
            
            # 评估
            reconstructions, latents, batch_metrics = self.evaluate_batch(videos)
            
            # 累积指标
            for key in self.metrics.keys():
                self.metrics[key].extend(batch_metrics[key])
            
            sample_count += videos.shape[0]
            pbar.update(videos.shape[0])
            
            # 保存一些可视化样本
            if sample_count <= 5:
                # 对齐帧数：裁剪原始视频以匹配重建视频
                frames_to_trim = self.model.frames_to_trim
                if reconstructions.shape[1] < videos.shape[1]:
                    videos_trimmed = videos[:, frames_to_trim:frames_to_trim + reconstructions.shape[1]]
                else:
                    videos_trimmed = videos
                
                self._save_visualization(
                    videos_trimmed, 
                    reconstructions, 
                    sample_idx=sample_count,
                    save_dir=Path("evaluation_results")
                )
        
        pbar.close()
        
        # 计算统计信息
        results = self._compute_statistics()
        
        return results
    
    def _compute_statistics(self):
        """计算统计信息"""
        results = {}
        
        for metric_name, values in self.metrics.items():
            results[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        
        return results
    
    def _save_visualization(self, original, reconstructed, sample_idx, save_dir):
        """保存可视化结果"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 取第一个样本的中间帧
        orig = original[0, original.shape[1]//2].cpu().numpy()
        recon = reconstructed[0, reconstructed.shape[1]//2].cpu().numpy()
        
        # 数据已经在 [0, 1] 范围，直接转换到 (H, W, C)
        orig = np.transpose(orig, (1, 2, 0))
        recon = np.transpose(recon, (1, 2, 0))
        
        # 计算误差图
        error = np.abs(orig - recon)
        
        # 创建图像
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].imshow(orig)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(recon)
        axes[0, 1].set_title('Reconstructed')
        axes[0, 1].axis('off')
        
        # 误差热图
        error_gray = np.mean(error, axis=2)
        im = axes[1, 0].imshow(error_gray, cmap='hot', vmin=0, vmax=0.3)
        axes[1, 0].set_title('Absolute Error Heatmap')
        axes[1, 0].axis('off')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 并排对比
        comparison = np.concatenate([orig, recon], axis=1)
        axes[1, 1].imshow(comparison)
        axes[1, 1].set_title('Original | Reconstructed')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "="*60)
        print("📊 VAE Model Evaluation Results")
        print("="*60)
        
        # 指标说明
        metric_descriptions = {
            'psnr': 'PSNR (Peak Signal-to-Noise Ratio) - 越高越好 (>30 excellent)',
            'ssim': 'SSIM (Structural Similarity) - 越高越好 (>0.9 excellent)',
            'lpips': 'LPIPS (Perceptual Loss) - 越低越好 (<0.1 excellent)',
            'mse': 'MSE (Mean Squared Error) - 越低越好',
            'mae': 'MAE (Mean Absolute Error) - 越低越好',
        }
        
        for metric_name, stats in results.items():
            print(f"\n📈 {metric_descriptions[metric_name]}")
            print(f"   Mean:   {stats['mean']:.4f}")
            print(f"   Std:    {stats['std']:.4f}")
            print(f"   Median: {stats['median']:.4f}")
            print(f"   Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # 综合评分
        print("\n" + "="*60)
        print("🎯 Overall Quality Assessment:")
        print("="*60)
        
        psnr_score = min(results['psnr']['mean'] / 35.0, 1.0)
        ssim_score = results['ssim']['mean']
        lpips_score = max(1.0 - results['lpips']['mean'] * 10, 0.0)
        
        overall_score = (psnr_score * 0.3 + ssim_score * 0.4 + lpips_score * 0.3) * 100
        
        print(f"   PSNR Score:    {psnr_score*100:.1f}/100")
        print(f"   SSIM Score:    {ssim_score*100:.1f}/100")
        print(f"   LPIPS Score:   {lpips_score*100:.1f}/100")
        print(f"   Overall Score: {overall_score:.1f}/100")
        
        if overall_score >= 85:
            quality = "🌟 Excellent"
        elif overall_score >= 70:
            quality = "✅ Good"
        elif overall_score >= 55:
            quality = "⚠️  Fair"
        else:
            quality = "❌ Poor"
        
        print(f"\n   Quality Level: {quality}")
        print("="*60)
    
    def save_results(self, results, save_path):
        """保存结果到JSON"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {save_path}")


def plot_metrics_distribution(evaluator, save_dir):
    """绘制指标分布图"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, (metric_name, values) in enumerate(evaluator.metrics.items()):
        if idx < len(axes):
            axes[idx].hist(values, bins=50, color=colors[idx], alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{metric_name.upper()} Distribution', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
            
            # 添加均值线
            mean_val = np.mean(values)
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            axes[idx].legend()
    
    # 隐藏多余的子图
    for idx in range(len(evaluator.metrics), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Metrics distribution plot saved to: {save_dir / 'metrics_distribution.png'}")


def compare_checkpoints(checkpoint_paths, config, dataloader, device='cuda'):
    """比较多个checkpoints的性能"""
    print("\n🔄 Comparing multiple checkpoints...")
    
    results_comparison = {}
    
    for ckpt_path in checkpoint_paths:
        ckpt_name = Path(ckpt_path).name
        print(f"\n⏳ Evaluating: {ckpt_name}")
        
        evaluator = VAEEvaluator(ckpt_path, config, device=device, num_samples=50)
        results = evaluator.evaluate_dataset(dataloader)
        results_comparison[ckpt_name] = results
    
    # 绘制对比图
    metrics_to_plot = ['psnr', 'ssim', 'lpips']
    
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics_to_plot):
        checkpoint_names = list(results_comparison.keys())
        means = [results_comparison[name][metric]['mean'] for name in checkpoint_names]
        stds = [results_comparison[name][metric]['std'] for name in checkpoint_names]
        
        axes[idx].bar(range(len(checkpoint_names)), means, yerr=stds, capsize=5)
        axes[idx].set_xticks(range(len(checkpoint_names)))
        axes[idx].set_xticklabels(checkpoint_names, rotation=45, ha='right')
        axes[idx].set_title(f'{metric.upper()} Comparison')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results/checkpoint_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n📊 Checkpoint comparison saved to: evaluation_results/checkpoint_comparison.png")
    
    return results_comparison


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAE Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='training/configs/taehv_config_h100.py', help='Config file')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--compare_checkpoints', action='store_true', help='Compare multiple checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory containing checkpoints to compare')
    parser.add_argument('--data_root', type=str, help='Override data root directory from config')
    parser.add_argument('--annotation_file', type=str, help='Override annotation file from config')
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"📋 Loading config from: {args.config}")
    config_path = Path(args.config)
    
    # 动态导入配置
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    config = config_module.args
    
    # 创建数据集 (使用验证集)
    print("\n📂 Loading validation dataset...")
    
    # 使用命令行参数覆盖配置
    data_root = args.data_root if args.data_root else config.data_root
    annotation_file = args.annotation_file if args.annotation_file else config.annotation_file
    
    # 检查数据集路径是否存在
    if not Path(annotation_file).exists():
        print(f"❌ Annotation file not found: {annotation_file}")
        print(f"\n💡 Tip: 如果你没有原始训练数据集，你可以：")
        print(f"   1. 使用 --annotation_file 和 --data_root 参数指定可用的数据集")
        print(f"   2. 创建一个小的测试数据集用于评估")
        print(f"\n   示例:")
        print(f"   python evaluate_vae.py \\")
        print(f"       --model_path ../output/xxx/final_model.pth \\")
        print(f"       --annotation_file /path/to/your/test_annotations.json \\")
        print(f"       --data_root /path/to/your/test_data")
        sys.exit(1)
    
    if not Path(data_root).exists():
        print(f"❌ Data directory not found: {data_root}")
        print(f"\n💡 Tip: 使用 --data_root 参数指定正确的数据目录")
        sys.exit(1)
    
    try:
        dataset = MiniDataset(
            annotation_file=annotation_file,
            data_dir=data_root,
            patch_hw=max(config.height, config.width),  # 使用较大的边作为patch大小
            n_frames=config.n_frames,
            augmentation=False  # 评估时不做数据增强
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"✅ Loaded {len(dataset)} validation samples")
        
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        print(f"\n💡 请检查数据集路径和格式是否正确")
        sys.exit(1)
    
    # 评估模式选择
    if args.compare_checkpoints and args.checkpoint_dir:
        # 比较多个checkpoints
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_paths = sorted(checkpoint_dir.glob('checkpoint-*/pytorch_model.bin'))
        
        if not checkpoint_paths:
            checkpoint_paths = sorted(checkpoint_dir.glob('*.pth'))
        
        if not checkpoint_paths:
            print("❌ No checkpoints found!")
            return
        
        print(f"Found {len(checkpoint_paths)} checkpoints to compare")
        results_comparison = compare_checkpoints(
            checkpoint_paths[:5],  # 最多比较5个
            config,
            dataloader,
            device='cuda'
        )
        
        # 保存对比结果
        with open('evaluation_results/checkpoint_comparison.json', 'w') as f:
            json.dump(results_comparison, f, indent=2)
        
    else:
        # 单个模型评估
        evaluator = VAEEvaluator(
            args.model_path,
            config,
            device='cuda',
            num_samples=args.num_samples
        )
        
        # 运行评估
        results = evaluator.evaluate_dataset(dataloader)
        
        # 打印结果
        evaluator.print_results(results)
        
        # 保存结果
        evaluator.save_results(results, 'evaluation_results/evaluation_results.json')
        
        # 绘制分布图
        plot_metrics_distribution(evaluator, 'evaluation_results')


if __name__ == '__main__':
    main()

