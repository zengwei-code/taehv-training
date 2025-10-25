#!/usr/bin/env python3
"""
TAEHV 可视化工具 - 专注于结果可视化和报告生成
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


class TAEHVVisualizer:
    """TAEHV 结果可视化器"""
    
    @staticmethod
    def save_video_tensor(video_tensor: torch.Tensor, output_path: Path, fps: int = 8):
        """
        保存视频张量为MP4文件
        Args:
            video_tensor: [T,C,H,W] 视频张量，范围 [0,1]
            output_path: 输出路径
            fps: 帧率
        """
        frames = []
        for t in range(video_tensor.shape[0]):
            frame = video_tensor[t].cpu().numpy().transpose(1, 2, 0)
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame = np.ascontiguousarray(frame)  # 确保连续内存布局
            frames.append(frame)
        
        if len(frames) == 0:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"✅ Video saved to {output_path}")
    
    @staticmethod
    def create_comparison_video(
        original: torch.Tensor, 
        reconstructed: torch.Tensor, 
        output_path: Path, 
        fps: int = 8
    ):
        """
        创建左右对比视频
        Args:
            original: [T,C,H,W] 原始视频
            reconstructed: [T,C,H,W] 重建视频
            output_path: 输出路径
            fps: 帧率
        """
        frames = []
        for t in range(original.shape[0]):
            orig_frame = original[t].cpu().numpy().transpose(1, 2, 0)
            recon_frame = reconstructed[t].cpu().numpy().transpose(1, 2, 0)
            
            orig_frame = np.clip(orig_frame * 255, 0, 255).astype(np.uint8)
            recon_frame = np.clip(recon_frame * 255, 0, 255).astype(np.uint8)
            
            # 确保数组是连续的（OpenCV要求）
            orig_frame = np.ascontiguousarray(orig_frame)
            recon_frame = np.ascontiguousarray(recon_frame)
            
            # 添加标签
            cv2.putText(orig_frame, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(recon_frame, "Reconstructed", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            comparison = np.hstack([orig_frame, recon_frame])
            frames.append(comparison)
        
        if len(frames) == 0:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print(f"✅ Comparison video saved to {output_path}")
    
    @staticmethod
    def visualize_latent(
        latent: torch.Tensor, 
        output_path: Path,
        max_channels: int = 16
    ):
        """
        可视化潜在表示
        Args:
            latent: [T,C,H,W] 潜在张量
            output_path: 输出路径
            max_channels: 最多显示的通道数
        """
        T, C, H, W = latent.shape
        n_channels = min(C, max_channels)
        
        # 选择中间帧
        mid_t = T // 2
        latent_frame = latent[mid_t].cpu().numpy()
        
        # 创建网格
        cols = 4
        rows = (n_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        axes = axes.flatten() if n_channels > 1 else [axes]
        
        for i in range(n_channels):
            channel = latent_frame[i]
            # 归一化到 [0, 1]
            channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            
            axes[i].imshow(channel, cmap='viridis')
            axes[i].set_title(f'Channel {i}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Latent visualization saved to {output_path}")
    
    @staticmethod
    def create_error_heatmap(
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        output_path: Path,
        frame_idx: int = 0
    ):
        """
        创建误差热图
        Args:
            original: [T,C,H,W] 原始视频
            reconstructed: [T,C,H,W] 重建视频
            output_path: 输出路径
            frame_idx: 要可视化的帧索引
        """
        orig_frame = original[frame_idx].cpu().numpy().transpose(1, 2, 0)
        recon_frame = reconstructed[frame_idx].cpu().numpy().transpose(1, 2, 0)
        
        # 计算误差
        error = np.abs(orig_frame - recon_frame).mean(axis=2)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始帧
        axes[0].imshow(np.clip(orig_frame, 0, 1))
        axes[0].set_title('Original', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 重建帧
        axes[1].imshow(np.clip(recon_frame, 0, 1))
        axes[1].set_title('Reconstructed', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # 误差热图
        im = axes[2].imshow(error, cmap='hot', vmin=0, vmax=0.2)
        axes[2].set_title('Error Heatmap', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Error heatmap saved to {output_path}")
    
    @staticmethod
    def create_summary_grid(
        originals: torch.Tensor,
        reconstructions: torch.Tensor,
        output_path: Path,
        num_samples: int = 9
    ):
        """
        创建样本网格对比图
        Args:
            originals: [N,T,C,H,W] 原始视频批次
            reconstructions: [N,T,C,H,W] 重建视频批次
            output_path: 输出路径
            num_samples: 显示样本数
        """
        N = min(originals.shape[0], num_samples)
        cols = 3
        rows = (N + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols*2, figsize=(cols*6, rows*3))
        axes = axes.flatten() if N > 1 else [axes]
        
        for i in range(N):
            # 取每个视频的首帧
            orig = originals[i, 0].cpu().numpy().transpose(1, 2, 0)
            recon = reconstructions[i, 0].cpu().numpy().transpose(1, 2, 0)
            
            # 原始帧
            axes[i*2].imshow(np.clip(orig, 0, 1))
            axes[i*2].set_title(f'Sample {i} - Original')
            axes[i*2].axis('off')
            
            # 重建帧
            axes[i*2+1].imshow(np.clip(recon, 0, 1))
            axes[i*2+1].set_title(f'Sample {i} - Recon')
            axes[i*2+1].axis('off')
        
        # 隐藏多余的子图
        for i in range(N*2, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Summary grid saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="TAEHV Visualization Tool")
    parser.add_argument("--originals", type=str, required=True, 
                       help="Path to originals.pt")
    parser.add_argument("--reconstructions", type=str, required=True, 
                       help="Path to reconstructions.pt")
    parser.add_argument("--latents", type=str, 
                       help="Path to latents.pt (optional)")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Sample index for detailed visualization")
    
    args = parser.parse_args()
    
    # 加载数据
    print("📂 Loading data...")
    originals = torch.load(args.originals, weights_only=False)
    reconstructions = torch.load(args.reconstructions, weights_only=False)
    print(f"✅ Loaded {originals.shape[0]} samples")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = TAEHVVisualizer()
    
    # 1. 创建对比视频
    print("\n🎬 Creating comparison videos...")
    visualizer.create_comparison_video(
        originals[args.sample_idx],
        reconstructions[args.sample_idx],
        output_dir / f"sample_{args.sample_idx:03d}_comparison.mp4"
    )
    
    # 2. 创建误差热图
    print("\n🔥 Creating error heatmap...")
    visualizer.create_error_heatmap(
        originals[args.sample_idx],
        reconstructions[args.sample_idx],
        output_dir / f"sample_{args.sample_idx:03d}_error.png",
        frame_idx=0
    )
    
    # 3. 创建样本网格
    print("\n🖼️  Creating summary grid...")
    visualizer.create_summary_grid(
        originals,
        reconstructions,
        output_dir / "summary_grid.png",
        num_samples=min(9, originals.shape[0])
    )
    
    # 4. 可视化潜在表示（如果有）
    if args.latents:
        print("\n🔮 Visualizing latent space...")
        latents = torch.load(args.latents, weights_only=False)
        visualizer.visualize_latent(
            latents[args.sample_idx],
            output_dir / f"sample_{args.sample_idx:03d}_latent.png"
        )
    
    print(f"\n✅ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()

