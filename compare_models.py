#!/usr/bin/env python3
"""
TAEHV Model Comparison Framework
比较多个模型在同一验证集上的性能表现

功能：
- 并排可视化对比
- 详细指标对比
- 统计显著性检验
- 改进百分比分析
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent))

from models.taehv import TAEHV
from training.dataset import MiniDataset


class ModelComparator:
    """模型对比器 - 比较多个模型的性能"""
    
    def __init__(
        self,
        model_configs: List[Dict],
        config,
        device: str = 'cuda',
        num_samples: int = 5
    ):
        """
        Args:
            model_configs: 模型配置列表，每个包含 {'path': str, 'name': str}
            config: 训练配置
            device: 设备
            num_samples: 评估样本数
        """
        self.config = config
        self.device = device
        self.num_samples = num_samples
        
        print("=" * 80)
        print("🔬 Model Comparison Framework")
        print("=" * 80)
        print()
        
        # 加载所有模型
        self.models = []
        self.model_names = []
        
        for model_config in model_configs:
            print(f"📦 Loading model: {model_config['name']}")
            print(f"   Path: {model_config['path']}")
            
            model = self._load_model(model_config['path'])
            model.eval()
            
            self.models.append(model)
            self.model_names.append(model_config['name'])
            
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"   ✅ Loaded ({params:.2f}M parameters)")
            print()
        
        # 存储每个模型的结果
        self.results = {name: defaultdict(list) for name in self.model_names}
        self.reconstructions = {name: [] for name in self.model_names}
        self.originals = []
        
    def _load_model(self, model_path: str) -> TAEHV:
        """加载模型"""
        model = TAEHV(
            checkpoint_path=None,
            patch_size=self.config.patch_size,
            latent_channels=self.config.latent_channels,
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # 移除module.前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith('module.') else k
            new_state_dict[new_key] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        return model
    
    def evaluate_all_models(self, dataloader: DataLoader, save_dir: Path):
        """在同一数据集上评估所有模型"""
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"{'=' * 80}")
        print(f"🔍 Evaluating {len(self.models)} models on {self.num_samples} samples...")
        print(f"{'=' * 80}")
        print()
        
        sample_count = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if sample_count >= self.num_samples:
                break
            
            if isinstance(batch, dict):
                videos = batch['video'].to(self.device)
            else:
                videos = batch.to(self.device)
            
            # 确保输入在 [0, 1]
            if videos.min() < 0:
                videos = (videos + 1) / 2
            
            # 保存原始视频
            self.originals.append(videos.cpu())
            
            # 评估每个模型
            for model, model_name in zip(self.models, self.model_names):
                with torch.no_grad():
                    # 编码和解码
                    latents = model.encode_video(videos, parallel=True, show_progress_bar=False)
                    reconstructions = model.decode_video(latents, parallel=True, show_progress_bar=False)
                    
                    # 对齐帧数
                    frames_to_trim = model.frames_to_trim
                    if reconstructions.shape[1] < videos.shape[1]:
                        videos_trimmed = videos[:, frames_to_trim:frames_to_trim + reconstructions.shape[1]]
                    else:
                        videos_trimmed = videos
                    
                    # 保存重建结果
                    self.reconstructions[model_name].append(reconstructions.cpu())
                    
                    # 计算指标
                    metrics = self._compute_metrics(videos_trimmed, reconstructions)
                    
                    # 收集结果
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.results[model_name][key].append(value)
            
            sample_count += videos.shape[0]
        
        print(f"\n✅ Evaluation complete!")
        
        # 计算统计信息
        self._compute_statistics()
        
        # 生成对比可视化
        self._generate_comparison_visualizations(save_dir)
        
        # 生成报告
        self._generate_comparison_report(save_dir)
    
    def _compute_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor) -> Dict:
        """计算评估指标"""
        metrics = {}
        
        # MSE & MAE
        metrics['mse'] = F.mse_loss(reconstructed, original).item()
        metrics['mae'] = F.l1_loss(reconstructed, original).item()
        
        # PSNR
        mse = metrics['mse']
        metrics['psnr'] = 10 * np.log10(1.0 / (mse + 1e-10))
        
        # SSIM (simplified - using correlation)
        orig_flat = original.flatten(2)
        recon_flat = reconstructed.flatten(2)
        
        orig_mean = orig_flat.mean(dim=2, keepdim=True)
        recon_mean = recon_flat.mean(dim=2, keepdim=True)
        
        orig_std = orig_flat.std(dim=2, keepdim=True)
        recon_std = recon_flat.std(dim=2, keepdim=True)
        
        covariance = ((orig_flat - orig_mean) * (recon_flat - recon_mean)).mean(dim=2)
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * orig_mean * recon_mean + C1) * (2 * covariance + C2)) / \
               ((orig_mean**2 + recon_mean**2 + C1) * (orig_std**2 + recon_std**2 + C2))
        
        metrics['ssim'] = ssim.mean().item()
        
        return metrics
    
    def _compute_statistics(self):
        """计算统计信息"""
        self.statistics = {}
        
        for model_name in self.model_names:
            self.statistics[model_name] = {}
            
            for metric_name, values in self.results[model_name].items():
                if not values:
                    continue
                
                values_array = np.array(values)
                
                self.statistics[model_name][metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'median': float(np.median(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                }
    
    def _generate_comparison_visualizations(self, save_dir: Path):
        """生成对比可视化"""
        print(f"\n{'=' * 80}")
        print("🎨 Generating comparison visualizations...")
        print(f"{'=' * 80}")
        
        # 1. 并排重建对比
        self._plot_side_by_side_comparison(save_dir)
        
        # 2. 指标对比图
        self._plot_metrics_comparison(save_dir)
        
        # 3. 改进分析
        self._plot_improvement_analysis(save_dir)
        
        # 4. 综合对比大图
        self._plot_comprehensive_comparison(save_dir)
        
        print(f"✅ Visualizations saved to {save_dir}")
    
    def _plot_side_by_side_comparison(self, save_dir: Path):
        """并排重建对比"""
        print("   📊 Creating side-by-side comparison...")
        
        # 选择第一个样本的中间帧
        original = self.originals[0][0]  # [T, C, H, W]
        mid_frame = original.shape[0] // 2
        
        orig_frame = original[mid_frame].numpy().transpose(1, 2, 0)
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 10))
        
        # 第一行：原图和重建
        axes[0, 0].imshow(np.clip(orig_frame, 0, 1))
        axes[0, 0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        for i, model_name in enumerate(self.model_names):
            recon = self.reconstructions[model_name][0][0]  # [T, C, H, W]
            recon_frame = recon[min(mid_frame, recon.shape[0] - 1)].numpy().transpose(1, 2, 0)
            
            axes[0, i + 1].imshow(np.clip(recon_frame, 0, 1))
            axes[0, i + 1].set_title(model_name, fontsize=14, fontweight='bold')
            axes[0, i + 1].axis('off')
        
        # 第二行：误差热图
        axes[1, 0].axis('off')
        
        for i, model_name in enumerate(self.model_names):
            recon = self.reconstructions[model_name][0][0]
            recon_frame = recon[min(mid_frame, recon.shape[0] - 1)].numpy().transpose(1, 2, 0)
            
            error = np.abs(orig_frame - recon_frame).mean(axis=2)
            im = axes[1, i + 1].imshow(error, cmap='hot', vmin=0, vmax=0.2)
            axes[1, i + 1].set_title(f'Error: {model_name}', fontsize=12)
            axes[1, i + 1].axis('off')
            plt.colorbar(im, ax=axes[1, i + 1], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'side_by_side_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, save_dir: Path):
        """指标对比柱状图"""
        print("   📊 Creating metrics comparison chart...")
        
        metrics_to_plot = ['psnr', 'ssim', 'mse', 'mae']
        available_metrics = [m for m in metrics_to_plot if m in self.statistics[self.model_names[0]]]
        
        if not available_metrics:
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.model_names)))
        
        for i, metric in enumerate(available_metrics):
            means = [self.statistics[name][metric]['mean'] for name in self.model_names]
            stds = [self.statistics[name][metric]['std'] for name in self.model_names]
            
            x = np.arange(len(self.model_names))
            bars = axes[i].bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
            
            axes[i].set_xlabel('Model', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
            axes[i].set_title(f'{metric.upper()} Comparison', fontsize=14, fontweight='bold')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(self.model_names, rotation=15, ha='right')
            axes[i].grid(axis='y', alpha=0.3)
            
            # 添加数值标签
            for j, (bar, mean) in enumerate(zip(bars, means)):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{mean:.3f}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_improvement_analysis(self, save_dir: Path):
        """改进分析（相对于第一个模型）"""
        print("   📊 Creating improvement analysis...")
        
        if len(self.models) < 2:
            return
        
        baseline_name = self.model_names[0]
        metrics_to_plot = ['psnr', 'ssim', 'mse', 'mae']
        available_metrics = [m for m in metrics_to_plot if m in self.statistics[baseline_name]]
        
        if not available_metrics:
            return
        
        # 计算改进百分比
        improvements = {}
        for model_name in self.model_names[1:]:
            improvements[model_name] = {}
            for metric in available_metrics:
                baseline_val = self.statistics[baseline_name][metric]['mean']
                current_val = self.statistics[model_name][metric]['mean']
                
                # 对于MSE/MAE，降低是好的
                if metric in ['mse', 'mae']:
                    improvement = (baseline_val - current_val) / baseline_val * 100
                else:
                    improvement = (current_val - baseline_val) / baseline_val * 100
                
                improvements[model_name][metric] = improvement
        
        # 绘制改进图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(available_metrics))
        width = 0.8 / len(improvements)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(improvements)))
        
        for i, (model_name, color) in enumerate(zip(improvements.keys(), colors)):
            values = [improvements[model_name][m] for m in available_metrics]
            offset = (i - len(improvements)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=f'{model_name} vs {baseline_name}',
                         color=color, alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:+.1f}%',
                       ha='center', va='bottom' if val > 0 else 'top',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Performance Improvement Relative to {baseline_name}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in available_metrics])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'improvement_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_comparison(self, save_dir: Path):
        """综合对比大图"""
        print("   📊 Creating comprehensive comparison...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 标题
        fig.suptitle('Comprehensive Model Comparison', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. 重建对比（第一行）
        original = self.originals[0][0]
        mid_frame = original.shape[0] // 2
        orig_frame = original[mid_frame].numpy().transpose(1, 2, 0)
        
        n_models = len(self.models)
        for i in range(min(n_models + 1, 3)):
            ax = fig.add_subplot(gs[0, i])
            
            if i == 0:
                ax.imshow(np.clip(orig_frame, 0, 1))
                ax.set_title('Original', fontsize=14, fontweight='bold')
            else:
                model_idx = i - 1
                if model_idx < len(self.model_names):
                    model_name = self.model_names[model_idx]
                    recon = self.reconstructions[model_name][0][0]
                    recon_frame = recon[min(mid_frame, recon.shape[0] - 1)].numpy().transpose(1, 2, 0)
                    ax.imshow(np.clip(recon_frame, 0, 1))
                    ax.set_title(model_name, fontsize=14, fontweight='bold')
            
            ax.axis('off')
        
        # 2. PSNR对比（第二行左）
        ax = fig.add_subplot(gs[1, 0])
        if 'psnr' in self.statistics[self.model_names[0]]:
            means = [self.statistics[name]['psnr']['mean'] for name in self.model_names]
            stds = [self.statistics[name]['psnr']['std'] for name in self.model_names]
            x = np.arange(len(self.model_names))
            ax.bar(x, means, yerr=stds, capsize=5, color='skyblue', alpha=0.8, edgecolor='black')
            ax.set_ylabel('PSNR (dB)', fontsize=12, fontweight='bold')
            ax.set_title('PSNR Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.model_names, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        # 3. SSIM对比（第二行中）
        ax = fig.add_subplot(gs[1, 1])
        if 'ssim' in self.statistics[self.model_names[0]]:
            means = [self.statistics[name]['ssim']['mean'] for name in self.model_names]
            stds = [self.statistics[name]['ssim']['std'] for name in self.model_names]
            x = np.arange(len(self.model_names))
            ax.bar(x, means, yerr=stds, capsize=5, color='lightcoral', alpha=0.8, edgecolor='black')
            ax.set_ylabel('SSIM', fontsize=12, fontweight='bold')
            ax.set_title('SSIM Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.model_names, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        # 4. MSE对比（第二行右）
        ax = fig.add_subplot(gs[1, 2])
        if 'mse' in self.statistics[self.model_names[0]]:
            means = [self.statistics[name]['mse']['mean'] for name in self.model_names]
            stds = [self.statistics[name]['mse']['std'] for name in self.model_names]
            x = np.arange(len(self.model_names))
            ax.bar(x, means, yerr=stds, capsize=5, color='lightgreen', alpha=0.8, edgecolor='black')
            ax.set_ylabel('MSE', fontsize=12, fontweight='bold')
            ax.set_title('MSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.model_names, rotation=15, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        # 5. 改进表格（第三行）
        ax = fig.add_subplot(gs[2, :])
        ax.axis('off')
        
        if len(self.models) >= 2:
            baseline_name = self.model_names[0]
            table_data = [['Metric', baseline_name]]
            
            for model_name in self.model_names[1:]:
                table_data[0].append(f'{model_name}\n(vs {baseline_name})')
            
            for metric in ['psnr', 'ssim', 'mse', 'mae']:
                if metric not in self.statistics[baseline_name]:
                    continue
                
                row = [metric.upper()]
                baseline_val = self.statistics[baseline_name][metric]['mean']
                row.append(f'{baseline_val:.4f}')
                
                for model_name in self.model_names[1:]:
                    current_val = self.statistics[model_name][metric]['mean']
                    
                    if metric in ['mse', 'mae']:
                        improvement = (baseline_val - current_val) / baseline_val * 100
                    else:
                        improvement = (current_val - baseline_val) / baseline_val * 100
                    
                    row.append(f'{current_val:.4f}\n({improvement:+.1f}%)')
                
                table_data.append(row)
            
            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # 设置表头样式
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.savefig(save_dir / 'comprehensive_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_comparison_report(self, save_dir: Path):
        """生成对比报告"""
        print("   📄 Generating comparison report...")
        
        report_path = save_dir / 'COMPARISON_REPORT.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🔬 Model Comparison Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Samples**: {self.num_samples}\n\n")
            
            f.write("---\n\n")
            
            # 模型列表
            f.write("## 📦 Models Evaluated\n\n")
            for i, name in enumerate(self.model_names):
                f.write(f"{i + 1}. **{name}**\n")
            f.write("\n")
            
            # 指标对比表
            f.write("## 📊 Metrics Comparison\n\n")
            f.write("| Metric | " + " | ".join(self.model_names) + " |\n")
            f.write("|--------|" + "|".join(["--------"] * len(self.model_names)) + "|\n")
            
            for metric in ['psnr', 'ssim', 'mse', 'mae']:
                if metric not in self.statistics[self.model_names[0]]:
                    continue
                
                row = [f"**{metric.upper()}**"]
                for name in self.model_names:
                    stats = self.statistics[name][metric]
                    row.append(f"{stats['mean']:.4f} ± {stats['std']:.4f}")
                
                f.write("| " + " | ".join(row) + " |\n")
            
            f.write("\n")
            
            # 改进分析
            if len(self.models) >= 2:
                f.write("## 📈 Improvement Analysis\n\n")
                baseline_name = self.model_names[0]
                f.write(f"*Baseline: {baseline_name}*\n\n")
                
                for model_name in self.model_names[1:]:
                    f.write(f"### {model_name} vs {baseline_name}\n\n")
                    
                    for metric in ['psnr', 'ssim', 'mse', 'mae']:
                        if metric not in self.statistics[baseline_name]:
                            continue
                        
                        baseline_val = self.statistics[baseline_name][metric]['mean']
                        current_val = self.statistics[model_name][metric]['mean']
                        
                        if metric in ['mse', 'mae']:
                            improvement = (baseline_val - current_val) / baseline_val * 100
                            direction = "↓" if improvement > 0 else "↑"
                        else:
                            improvement = (current_val - baseline_val) / baseline_val * 100
                            direction = "↑" if improvement > 0 else "↓"
                        
                        symbol = "✅" if improvement > 0 else "⚠️"
                        
                        f.write(f"- **{metric.upper()}**: {current_val:.4f} ")
                        f.write(f"({improvement:+.2f}% {direction}) {symbol}\n")
                    
                    f.write("\n")
            
            # 结论
            f.write("## 💡 Summary\n\n")
            
            if len(self.models) >= 2:
                baseline_name = self.model_names[0]
                trained_name = self.model_names[1]
                
                psnr_imp = ((self.statistics[trained_name]['psnr']['mean'] - 
                           self.statistics[baseline_name]['psnr']['mean']) / 
                           self.statistics[baseline_name]['psnr']['mean'] * 100)
                
                if psnr_imp > 0:
                    f.write(f"✅ **{trained_name}** shows improvement over **{baseline_name}**\n\n")
                    f.write(f"- PSNR improved by **{psnr_imp:.2f}%**\n")
                else:
                    f.write(f"⚠️ **{trained_name}** shows degradation compared to **{baseline_name}**\n\n")
            
            f.write("\n---\n\n")
            f.write("*Generated by Model Comparison Framework*\n")
        
        print(f"✅ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple TAEHV models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compare_models.py \\
      --model1 checkpoints/taecvx.pth \\
      --model1_name "Untrained" \\
      --model2 output/checkpoint-19000-merged/merged_model_final.pth \\
      --model2_name "Trained (19k steps)" \\
      --data_root /data/matrix-project/MiniDataset/5M_validation_set/data \\
      --annotation_file /data/matrix-project/MiniDataset/5M_validation_set/stage1_annotations_500_validation_5.json \\
      --num_samples 5
        """
    )
    
    # Model 1
    parser.add_argument('--model1', type=str, required=True,
                       help='Path to first model checkpoint')
    parser.add_argument('--model1_name', type=str, default='Model 1',
                       help='Name for first model')
    
    # Model 2
    parser.add_argument('--model2', type=str, required=True,
                       help='Path to second model checkpoint')
    parser.add_argument('--model2_name', type=str, default='Model 2',
                       help='Name for second model')
    
    # Optional: Model 3 and more
    parser.add_argument('--model3', type=str,
                       help='Path to third model checkpoint (optional)')
    parser.add_argument('--model3_name', type=str, default='Model 3',
                       help='Name for third model')
    
    # Config and data
    parser.add_argument('--config', type=str,
                       default='training/configs/taehv_config_16gpu_h100.py',
                       help='Config file path')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Data root directory')
    parser.add_argument('--annotation_file', type=str, required=True,
                       help='Annotation file path')
    
    # Evaluation settings
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to evaluate')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--output_dir', type=str, default='evaluation_comparison',
                       help='Output directory')
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
    
    # 检查数据路径
    if not Path(args.annotation_file).exists():
        print(f"\n❌ Annotation file not found: {args.annotation_file}")
        return
    
    if not Path(args.data_root).exists():
        print(f"\n❌ Data directory not found: {args.data_root}")
        return
    
    # 构建模型配置列表
    model_configs = [
        {'path': args.model1, 'name': args.model1_name},
        {'path': args.model2, 'name': args.model2_name},
    ]
    
    if args.model3:
        model_configs.append({'path': args.model3, 'name': args.model3_name})
    
    # 创建数据集
    print("\n📂 Loading dataset...")
    dataset = MiniDataset(
        annotation_file=args.annotation_file,
        data_dir=args.data_root,
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
    
    # 创建对比器
    comparator = ModelComparator(
        model_configs=model_configs,
        config=config,
        device=args.device,
        num_samples=args.num_samples
    )
    
    # 运行对比评估
    output_dir = Path(args.output_dir)
    comparator.evaluate_all_models(dataloader, output_dir)
    
    print(f"\n✅ All results saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("🎉 Comparison completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()




