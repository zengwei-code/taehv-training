#!/usr/bin/env python3
"""
TAEHV 简洁推理脚本 - 只负责推理功能
专注于视频编码和解码，不包含评估和可视化
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

from models.taehv import TAEHV
from training.dataset import MiniDataset


class SimpleInference:
    """简洁的TAEHV推理器 - 单一职责：推理"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.model = self._load_model(model_path)
        self.model.eval()
        
    def _load_model(self, model_path: str) -> TAEHV:
        """加载模型"""
        print(f"Loading model from {model_path}")
        model = TAEHV(checkpoint_path=None)
        
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model loaded: {param_count:.2f}M parameters")
        return model
    
    @torch.no_grad()
    def encode(self, video: torch.Tensor, parallel: bool = True) -> torch.Tensor:
        """
        编码视频到潜在空间
        Args:
            video: [N,T,C,H,W] 视频张量，范围 [0,1]
            parallel: 是否并行处理
        Returns:
            latent: [N,T',C',H',W'] 潜在表示
        """
        video = video.to(self.device)
        latent = self.model.encode_video(video, parallel=parallel, show_progress_bar=False)
        return latent
    
    @torch.no_grad()
    def decode(self, latent: torch.Tensor, parallel: bool = True) -> torch.Tensor:
        """
        从潜在空间解码视频
        Args:
            latent: [N,T,C,H,W] 潜在表示
            parallel: 是否并行处理
        Returns:
            video: [N,T',C',H',W'] 重建视频，范围 [0,1]
        """
        latent = latent.to(self.device)
        video = self.model.decode_video(latent, parallel=parallel, show_progress_bar=False)
        return video
    
    @torch.no_grad()
    def reconstruct(self, video: torch.Tensor, parallel: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整的重建流程：编码 -> 解码
        Args:
            video: [N,T,C,H,W] 输入视频
            parallel: 是否并行处理
        Returns:
            reconstructed: 重建的视频
            latent: 潜在表示
        """
        latent = self.encode(video, parallel=parallel)
        reconstructed = self.decode(latent, parallel=parallel)
        return reconstructed, latent
    
    def benchmark(self, video: torch.Tensor, num_runs: int = 10) -> dict:
        """
        性能基准测试
        Args:
            video: 测试视频
            num_runs: 运行次数
        Returns:
            性能统计字典
        """
        video = video.to(self.device)
        
        # 预热
        _ = self.reconstruct(video)
        
        encode_times = []
        decode_times = []
        
        for _ in range(num_runs):
            # 编码
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start = time.time()
            latent = self.encode(video)
            torch.cuda.synchronize() if self.device == 'cuda' else None
            encode_times.append(time.time() - start)
            
            # 解码
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start = time.time()
            _ = self.decode(latent)
            torch.cuda.synchronize() if self.device == 'cuda' else None
            decode_times.append(time.time() - start)
        
        return {
            'encode_mean': np.mean(encode_times),
            'encode_std': np.std(encode_times),
            'decode_mean': np.mean(decode_times),
            'decode_std': np.std(decode_times),
            'total_mean': np.mean(encode_times) + np.mean(decode_times),
        }


def main():
    parser = argparse.ArgumentParser(description="TAEHV Simple Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="/data/matrix-project/MiniDataset/data")
    parser.add_argument("--annotation_file", type=str, 
                       default="/data/matrix-project/MiniDataset/stage1_annotations_500.json")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_latents", action="store_true", help="Save latent representations")
    parser.add_argument("--output_dir", type=str, default="inference_output")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = SimpleInference(args.model_path, args.device)
    
    # 加载数据
    print(f"\n📂 Loading dataset...")
    dataset = MiniDataset(
        annotation_file=args.annotation_file,
        data_dir=args.data_root,
        patch_hw=128,
        n_frames=12
    )
    print(f"✅ Loaded {len(dataset)} samples")
    
    # 推理
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Running inference on {args.num_samples} samples...")
    
    all_latents = []
    all_reconstructions = []
    all_originals = []
    
    for i in tqdm(range(min(args.num_samples, len(dataset)))):
        video = dataset[i].unsqueeze(0).float().to(args.device)
        
        # 推理
        reconstructed, latent = inferencer.reconstruct(video)
        
        # 对齐帧数
        if inferencer.model.frames_to_trim > 0:
            video_aligned = video[:, inferencer.model.frames_to_trim:]
        else:
            video_aligned = video
        
        if args.save_latents:
            all_latents.append(latent.cpu())
        all_reconstructions.append(reconstructed.cpu())
        all_originals.append(video_aligned.cpu())
    
    # 保存结果
    if args.save_latents:
        latents_path = output_dir / "latents.pt"
        torch.save(torch.cat(all_latents, dim=0), latents_path)
        print(f"✅ Latents saved to {latents_path}")
    
    recon_path = output_dir / "reconstructions.pt"
    torch.save(torch.cat(all_reconstructions, dim=0), recon_path)
    print(f"✅ Reconstructions saved to {recon_path}")
    
    orig_path = output_dir / "originals.pt"
    torch.save(torch.cat(all_originals, dim=0), orig_path)
    print(f"✅ Originals saved to {orig_path}")
    
    # 性能基准测试
    if args.benchmark:
        print("\n⏱️  Running performance benchmark...")
        test_video = dataset[0].unsqueeze(0).float()
        stats = inferencer.benchmark(test_video, num_runs=20)
        print(f"\n📊 Performance Results:")
        print(f"   Encoding: {stats['encode_mean']*1000:.2f} ± {stats['encode_std']*1000:.2f} ms")
        print(f"   Decoding: {stats['decode_mean']*1000:.2f} ± {stats['decode_std']*1000:.2f} ms")
        print(f"   Total:    {stats['total_mean']*1000:.2f} ms")
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()

