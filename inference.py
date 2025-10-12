#!/usr/bin/env python3
"""
TAEHV完整推理测试脚本
整合所有功能：SSIM指标 + 结构化输出 + 潜在表示可视化 + HTML报告 + 文档格式输出
"""

import argparse
import json
import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 尝试导入高级指标
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("⚠️  torchmetrics not available, SSIM will be skipped")

try:
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

# 添加模型路径
import sys
sys.path.append(str(Path(__file__).parent))

from models.taehv import TAEHV
from training.dataset import MiniDataset
from training.utils import get_ref_vae


class TAEHVInference:
    """完整的TAEHV推理器"""
    
    def __init__(self, model_path: str, device: str = "cuda", use_ref_vae: bool = False, cogvideox_path: str = None):
        self.device = device
        self.model = self.load_model(model_path)
        
        # 初始化指标计算器
        if SSIM_AVAILABLE:
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        else:
            self.ssim = None
            
        if LPIPS_AVAILABLE:
            try:
                self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
                self.lpips_available = True
            except Exception:
                self.lpips_available = False
        else:
            self.lpips_available = False
        
        # 加载参考VAE（可选）
        self.ref_vae = None
        if use_ref_vae:
            try:
                self.ref_vae = get_ref_vae(device=device, cogvideox_model_path=cogvideox_path)
                self.ref_vae.eval()
                print("✅ Reference VAE loaded for comparison")
            except Exception as e:
                print(f"⚠️  Failed to load reference VAE: {e}")
    
    def load_model(self, model_path: str):
        """加载训练后的模型"""
        print(f"Loading model from {model_path}")
        model = TAEHV(checkpoint_path=None)
        
        if model_path.endswith('.pth'):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
        
        model = model.to(self.device).eval()
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ Model loaded on {self.device}")
        print(f"✅ Model has {param_count:.2f}M parameters")
        return model
    
    def calculate_comprehensive_metrics(self, original: torch.Tensor, reconstructed: torch.Tensor):
        """计算全面的质量指标"""
        original = torch.clamp(original, 0, 1)
        reconstructed = torch.clamp(reconstructed, 0, 1)
        
        metrics = {}
        
        # MSE
        mse = F.mse_loss(reconstructed, original).item()
        metrics['mse'] = mse
        
        # PSNR
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = float('inf')
        metrics['psnr'] = psnr
        
        # SSIM
        if self.ssim is not None:
            if original.dim() == 5:  # N,T,C,H,W
                ssim_scores = []
                for t in range(original.shape[1]):
                    ssim_score = self.ssim(reconstructed[:, t], original[:, t]).item()
                    ssim_scores.append(ssim_score)
                metrics['ssim'] = np.mean(ssim_scores)
            else:
                metrics['ssim'] = self.ssim(reconstructed, original).item()
        else:
            metrics['ssim'] = None
        
        # LPIPS
        if self.lpips_available:
            try:
                if original.dim() == 5:
                    lpips_scores = []
                    for t in range(original.shape[1]):
                        lpips_score = self.lpips(reconstructed[:, t], original[:, t]).item()
                        lpips_scores.append(lpips_score)
                    metrics['lpips'] = np.mean(lpips_scores)
                else:
                    metrics['lpips'] = self.lpips(reconstructed, original).item()
            except Exception:
                metrics['lpips'] = None
        else:
            metrics['lpips'] = None
        
        return metrics
    
    def evaluate_reference_vae(self, original: torch.Tensor):
        """使用参考VAE评估"""
        if self.ref_vae is None:
            return None
        
        try:
            with torch.no_grad():
                start_time = time.time()
                ref_encoded = self.ref_vae.encode_video(original)
                ref_decoded = self.ref_vae.decode_video(ref_encoded)
                ref_time = time.time() - start_time
                
                ref_metrics = self.calculate_comprehensive_metrics(original, ref_decoded)
                ref_metrics['inference_time'] = ref_time
                return ref_metrics
        except Exception as e:
            print(f"⚠️  Reference VAE evaluation failed: {e}")
            return None
    
    def save_video_tensor(self, video_tensor: torch.Tensor, output_path: Path, fps: int = 8):
        """保存视频张量为MP4文件"""
        frames = []
        for t in range(video_tensor.shape[0]):
            frame = video_tensor[t].cpu().numpy().transpose(1, 2, 0)
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
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
    
    def create_comparison_video(self, original: torch.Tensor, reconstructed: torch.Tensor, output_path: Path, fps: int = 8):
        """创建对比视频"""
        frames = []
        for t in range(original.shape[0]):
            orig_frame = original[t].cpu().numpy().transpose(1, 2, 0)
            recon_frame = reconstructed[t].cpu().numpy().transpose(1, 2, 0)
            
            orig_frame = np.clip(orig_frame * 255, 0, 255).astype(np.uint8)
            recon_frame = np.clip(recon_frame * 255, 0, 255).astype(np.uint8)
            
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
    
    def visualize_latents(self, encoded: torch.Tensor, output_path: str):
        """可视化潜在表示"""
        if encoded.dim() != 5:
            return
        
        latents = encoded[0].cpu()  # T,C,H,W
        n_frames, n_channels, h, w = latents.shape
        
        fig, axes = plt.subplots(2, min(6, n_frames), figsize=(min(6, n_frames)*3, 6))
        if n_frames == 1:
            axes = axes.reshape(2, 1)
        
        for t in range(min(6, n_frames)):
            # RGB通道
            if n_channels >= 3:
                rgb_latent = latents[t, :3]
                rgb_latent = (rgb_latent - rgb_latent.min()) / (rgb_latent.max() - rgb_latent.min() + 1e-8)
                axes[0, t].imshow(rgb_latent.permute(1, 2, 0))
                axes[0, t].set_title(f'Frame {t} (RGB channels)')
                axes[0, t].axis('off')
            
            # 通道均值
            mean_latent = latents[t].mean(0)
            mean_latent = (mean_latent - mean_latent.min()) / (mean_latent.max() - mean_latent.min() + 1e-8)
            axes[1, t].imshow(mean_latent.numpy(), cmap='viridis')
            axes[1, t].set_title(f'Frame {t} (Channel mean)')
            axes[1, t].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_structured_output(self, output_dir: Path, sample_idx: int, original_video: torch.Tensor, 
                                reconstructed_video: torch.Tensor, encoded: torch.Tensor, metrics: dict, timing: dict):
        """创建结构化输出"""
        comparison_dir = output_dir / "comparison_videos"
        latent_dir = output_dir / "latent_visualizations"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        latent_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存视频文件
        self.save_video_tensor(original_video[0], comparison_dir / f"sample_{sample_idx:03d}_original.mp4")
        self.save_video_tensor(reconstructed_video[0], comparison_dir / f"sample_{sample_idx:03d}_reconstructed.mp4")
        self.create_comparison_video(original_video[0], reconstructed_video[0], 
                                   comparison_dir / f"sample_{sample_idx:03d}_comparison.mp4")
        
        # 可视化潜在表示
        self.visualize_latents(encoded, latent_dir / f"sample_{sample_idx:03d}_latents.png")
        
        return {
            "sample_id": sample_idx,
            "metrics": metrics,
            "timing": timing,
            "files": {
                "original": str(comparison_dir / f"sample_{sample_idx:03d}_original.mp4"),
                "reconstructed": str(comparison_dir / f"sample_{sample_idx:03d}_reconstructed.mp4"),
                "comparison": str(comparison_dir / f"sample_{sample_idx:03d}_comparison.mp4"),
                "latents": str(latent_dir / f"sample_{sample_idx:03d}_latents.png")
            }
        }
    
    def generate_html_report(self, results: dict, output_path: Path):
        """生成HTML汇总报告"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TAEHV Inference Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .samples {{ margin-top: 30px; }}
        .sample {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .good {{ background-color: #e8f5e8; }}
        .warning {{ background-color: #fff3cd; }}
        .error {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>TAEHV推理质量报告</h1>
        <p><strong>模型:</strong> {results['model_path']}</p>
        <p><strong>测试时间:</strong> {results['timestamp']}</p>
        <p><strong>样本数量:</strong> {results['total_samples']}</p>
    </div>
    
    <div class="metrics">
        <div class="metric-box">
            <h3>平均MSE</h3>
            <h2>{results['average_metrics']['mse']:.6f}</h2>
        </div>
        <div class="metric-box">
            <h3>平均PSNR</h3>
            <h2>{results['average_metrics']['psnr']:.2f} dB</h2>
        </div>"""
        
        if results['average_metrics'].get('ssim') is not None:
            html_content += f"""
        <div class="metric-box">
            <h3>平均SSIM</h3>
            <h2>{results['average_metrics']['ssim']:.4f}</h2>
        </div>"""
        
        html_content += f"""
        <div class="metric-box">
            <h3>平均推理时间</h3>
            <h2>{results['average_timing']['total']:.3f}s</h2>
        </div>
    </div>
    
    <h2>质量分析</h2>
    <ul>
        <li><strong>重建质量:</strong> {'优秀' if results['average_metrics']['psnr'] > 30 else '良好' if results['average_metrics']['psnr'] > 25 else '需改进'}</li>"""
        
        if results['average_metrics'].get('ssim') is not None:
            html_content += f"""
        <li><strong>结构保持:</strong> {'优秀' if results['average_metrics']['ssim'] > 0.9 else '良好' if results['average_metrics']['ssim'] > 0.8 else '需改进'}</li>"""
        
        html_content += f"""
        <li><strong>推理效率:</strong> {'快' if results['average_timing']['total'] < 0.2 else '中等' if results['average_timing']['total'] < 0.5 else '慢'}</li>
    </ul>
    
    <div class="samples">
        <h2>样本详情 (前10个)</h2>"""
        
        for sample in results['samples'][:10]:
            status_class = ('good' if sample['metrics']['psnr'] > 30 else 'warning' if sample['metrics']['psnr'] > 25 else 'error')
            html_content += f"""
        <div class="sample {status_class}">
            <h4>样本 {sample['sample_id']}</h4>
            <p>MSE: {sample['metrics']['mse']:.6f}, PSNR: {sample['metrics']['psnr']:.2f}dB"""
            if sample['metrics'].get('ssim') is not None:
                html_content += f""", SSIM: {sample['metrics']['ssim']:.4f}"""
            html_content += f"""</p>
            <p>推理时间: {sample['timing']['total']:.3f}s</p>
        </div>"""
        
        html_content += """
    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def print_formatted_results(self, taehv_avg, taehv_std, timing_avg, ref_avg, num_samples):
        """按照文档中4.2.2的期望格式打印结果"""
        print("\n" + "=" * 60)
        print("Final Evaluation Results:")
        print("=" * 60)
        
        # Reconstruction Quality
        print("├─ Reconstruction Quality")
        mse_status = "✅" if taehv_avg['mse'] < 0.005 else "⚠️" if taehv_avg['mse'] < 0.01 else "❌"
        print(f"│   ├─ MSE: {taehv_avg['mse']:.6f} ± {taehv_std['mse']:.6f}  {mse_status} ({'< 0.005' if mse_status == '✅' else '< 0.01' if mse_status == '⚠️' else '≥ 0.01'})")
        
        psnr_status = "✅" if taehv_avg['psnr'] > 25.0 else "⚠️" if taehv_avg['psnr'] > 20.0 else "❌"
        print(f"│   ├─ PSNR: {taehv_avg['psnr']:.2f} ± {taehv_std['psnr']:.2f} dB   {psnr_status} ({'> 25.0' if psnr_status == '✅' else '> 20.0' if psnr_status == '⚠️' else '≤ 20.0'})")
        
        if taehv_avg.get('ssim') is not None:
            ssim_status = "✅" if taehv_avg['ssim'] > 0.85 else "⚠️" if taehv_avg['ssim'] > 0.80 else "❌"
            print(f"│   └─ SSIM: {taehv_avg['ssim']:.4f} ± {taehv_std['ssim']:.4f}     {ssim_status} ({'> 0.85' if ssim_status == '✅' else '> 0.80' if ssim_status == '⚠️' else '≤ 0.80'})")
        else:
            print("│   └─ SSIM: Not available (install torchmetrics)")
        
        # Performance
        print("├─ Performance")
        encoding_status = "✅" if timing_avg['encoding'] < 0.15 else "⚠️" if timing_avg['encoding'] < 0.30 else "❌"
        print(f"│   ├─ Encoding: {timing_avg['encoding']:.3f}s       {encoding_status} ({'< 0.15s' if encoding_status == '✅' else '< 0.30s' if encoding_status == '⚠️' else '≥ 0.30s'})")
        
        decoding_status = "✅" if timing_avg['decoding'] < 0.18 else "⚠️" if timing_avg['decoding'] < 0.35 else "❌"
        print(f"│   └─ Decoding: {timing_avg['decoding']:.3f}s       {decoding_status} ({'< 0.18s' if decoding_status == '✅' else '< 0.35s' if decoding_status == '⚠️' else '≥ 0.35s'})")
        
        # Reference VAE comparison
        if ref_avg and ref_avg.get('mse') and ref_avg.get('inference_time'):
            print("└─ vs Reference VAE")
            
            # Quality Gap
            if ref_avg['mse'] > 0:
                mse_improvement = (ref_avg['mse'] - taehv_avg['mse']) / ref_avg['mse'] * 100
                gap_status = "✅" if mse_improvement > -5 else "⚠️" if mse_improvement > -15 else "❌"
                print(f"    ├─ Quality Gap: {mse_improvement:+.1f}%     {gap_status} ({'>-5%' if gap_status == '✅' else '>-15%' if gap_status == '⚠️' else '≤-15%'})")
            
            # Speed Gain
            if ref_avg['inference_time'] > 0:
                speed_gain = (ref_avg['inference_time'] - timing_avg['total']) / ref_avg['inference_time'] * 100
                speed_status = "✅" if speed_gain > 200 else "⚠️" if speed_gain > 100 else "❌"
                print(f"    └─ Speed Gain: +{speed_gain:.0f}%      {speed_status} ({'>+200%' if speed_status == '✅' else '>+100%' if speed_status == '⚠️' else '≤+100%'})")
        else:
            print("└─ vs Reference VAE: Not available")
        
        print("\n" + "=" * 60)
        print(f"Evaluation completed on {num_samples} samples")
        
        # Overall assessment
        overall_scores = []
        if taehv_avg['mse'] < 0.005: overall_scores.append(1)
        elif taehv_avg['mse'] < 0.01: overall_scores.append(0.5)
        else: overall_scores.append(0)
        
        if taehv_avg['psnr'] > 25: overall_scores.append(1)
        elif taehv_avg['psnr'] > 20: overall_scores.append(0.5)
        else: overall_scores.append(0)
        
        if taehv_avg.get('ssim'):
            if taehv_avg['ssim'] > 0.85: overall_scores.append(1)
            elif taehv_avg['ssim'] > 0.80: overall_scores.append(0.5)
            else: overall_scores.append(0)
        
        avg_score = np.mean(overall_scores)
        if avg_score >= 0.8:
            print("🏆 Overall Assessment: EXCELLENT - Model exceeds expectations!")
        elif avg_score >= 0.6:
            print("✅ Overall Assessment: GOOD - Model meets most requirements")
        elif avg_score >= 0.4:
            print("⚠️  Overall Assessment: ACCEPTABLE - Model needs improvement")
        else:
            print("❌ Overall Assessment: POOR - Significant improvement needed")
        
        print("=" * 60)
    
    def run_inference(self, dataset_path: str, annotation_file: str, output_dir: str, 
                     num_samples: int = 10, output_format: str = "enhanced"):
        """运行完整推理测试"""
        
        print("=" * 60)
        print("TAEHV Inference Testing")  
        print("=" * 60)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集
        print("Loading test dataset...")
        dataset = MiniDataset(annotation_file=annotation_file, data_dir=dataset_path, patch_hw=128, n_frames=12)
        print(f"✅ Dataset loaded with {len(dataset)} samples")
        
        # 推理结果收集
        all_results = []
        all_metrics = {'mse': [], 'psnr': [], 'ssim': [], 'lpips': []}
        all_timings = {'encoding': [], 'decoding': [], 'total': []}
        ref_vae_metrics = {'mse': [], 'psnr': [], 'ssim': [], 'inference_time': []} if self.ref_vae else None
        
        # 批量推理
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Testing"):
            sample = dataset[i]
            frames = sample.unsqueeze(0).float() / 255.0  # N,T,C,H,W, [0,1]
            frames = frames.to(self.device)
            
            with torch.no_grad():
                # TAEHV推理
                start_time = time.time()
                
                # 编码
                encode_start = time.time()
                encoded = self.model.encode_video(frames, parallel=True, show_progress_bar=False)
                encode_time = time.time() - encode_start
                
                # 解码  
                decode_start = time.time()
                frames_target = frames[:, :-self.model.frames_to_trim] if self.model.frames_to_trim > 0 else frames
                decoded = self.model.decode_video(encoded, parallel=True, show_progress_bar=False)
                decode_time = time.time() - decode_start
                
                total_time = time.time() - start_time
            
            # TAEHV指标计算
            metrics = self.calculate_comprehensive_metrics(frames_target, decoded)
            timing = {'encoding': encode_time, 'decoding': decode_time, 'total': total_time}
            
            # 收集统计信息
            for key in all_metrics:
                if metrics.get(key) is not None:
                    all_metrics[key].append(metrics[key])
            for key in all_timings:
                all_timings[key].append(timing[key])
            
            # 参考VAE评估
            if self.ref_vae and ref_vae_metrics:
                ref_result = self.evaluate_reference_vae(frames_target)
                if ref_result:
                    for key in ref_vae_metrics:
                        if key in ref_result:
                            ref_vae_metrics[key].append(ref_result[key])
            
            # 创建输出
            if output_format in ["enhanced", "complete"]:
                sample_result = self.create_structured_output(output_dir, i, frames_target, decoded, encoded, metrics, timing)
                all_results.append(sample_result)
            elif output_format == "simple" and i < 5:
                # 简单格式：前5个样本
                self.save_video_tensor(frames_target[0], output_dir / f"sample_{i:03d}_original.mp4")
                self.save_video_tensor(decoded[0], output_dir / f"sample_{i:03d}_reconstructed.mp4") 
                self.create_comparison_video(frames_target[0], decoded[0], output_dir / f"sample_{i:03d}_comparison.mp4")
        
        # 计算平均指标
        taehv_avg = {}
        taehv_std = {}
        for key, values in all_metrics.items():
            if values:
                taehv_avg[key] = float(np.mean(values))
                taehv_std[key] = float(np.std(values))
            else:
                taehv_avg[key] = None
                taehv_std[key] = None
        
        taehv_timing_avg = {key: float(np.mean(values)) for key, values in all_timings.items()}
        
        # 参考VAE平均指标
        ref_avg = {}
        if ref_vae_metrics:
            for key, values in ref_vae_metrics.items():
                if values:
                    ref_avg[key] = float(np.mean(values))
        
        # 输出结果
        if output_format == "complete":
            # 文档格式输出
            self.print_formatted_results(taehv_avg, taehv_std, taehv_timing_avg, ref_avg, num_samples)
        else:
            # 标准输出
            print("\n" + "=" * 60)
            print("INFERENCE RESULTS")
            print("=" * 60)
            print(f"Tested samples: {len(all_metrics['mse'])}")
            print(f"Average MSE: {taehv_avg['mse']:.6f} ± {taehv_std['mse']:.6f}")
            print(f"Average PSNR: {taehv_avg['psnr']:.2f} ± {taehv_std['psnr']:.2f} dB") 
            if taehv_avg.get('ssim') is not None:
                print(f"Average SSIM: {taehv_avg['ssim']:.4f} ± {taehv_std['ssim']:.4f}")
            if taehv_avg.get('lpips') is not None:
                print(f"Average LPIPS: {taehv_avg['lpips']:.4f} ± {taehv_std['lpips']:.4f}")
            print(f"Average inference time: {taehv_timing_avg['total']:.3f}s")
            print(f"Results saved to: {output_dir}")
        
        # 保存结果
        if output_format in ["enhanced", "complete"]:
            # JSON结果
            final_results = {
                "model_path": str(self.model),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_samples": len(all_results) if all_results else len(all_metrics['mse']),
                "average_metrics": taehv_avg,
                "average_timing": taehv_timing_avg,
                "samples": all_results if all_results else []
            }
            
            with open(output_dir / "metrics.json", 'w') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            if output_format == "enhanced":
                self.generate_html_report(final_results, output_dir / "summary_report.html")
                print(f"✓ JSON results: {output_dir}/metrics.json")
                print(f"✓ HTML report: {output_dir}/summary_report.html")
        else:
            # 文本结果
            results_file = output_dir / "results.txt"
            with open(results_file, "w") as f:
                f.write("TAEHV Inference Results\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Tested samples: {len(all_metrics['mse'])}\n\n")
                f.write(f"Average MSE: {taehv_avg['mse']:.6f} ± {taehv_std['mse']:.6f}\n")
                f.write(f"Average PSNR: {taehv_avg['psnr']:.2f} ± {taehv_std['psnr']:.2f} dB\n")
                if taehv_avg.get('ssim') is not None:
                    f.write(f"Average SSIM: {taehv_avg['ssim']:.4f} ± {taehv_std['ssim']:.4f}\n")
                f.write("\nPer-sample results:\n")
                for i, (mse, psnr) in enumerate(zip(all_metrics['mse'], all_metrics['psnr'])):
                    f.write(f"Sample {i:3d}: MSE={mse:.6f}, PSNR={psnr:.2f}dB")
                    if all_metrics['ssim'] and i < len(all_metrics['ssim']):
                        f.write(f", SSIM={all_metrics['ssim'][i]:.4f}")
                    f.write("\n")
            print(f"✓ Detailed results saved to {results_file}")
        
        return {
            'taehv': {'metrics': taehv_avg, 'std': taehv_std, 'timing': taehv_timing_avg},
            'reference_vae': ref_avg if ref_avg else None,
            'num_samples': len(all_metrics['mse'])
        }


def main():
    parser = argparse.ArgumentParser(description="TAEHV Complete Inference Testing")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_root", type=str, default="/data/matrix-project/MiniDataset/data", help="Path to test data")
    parser.add_argument("--annotation_file", type=str, default="/data/matrix-project/MiniDataset/stage1_annotations_500.json", help="Path to annotation file")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--use_ref_vae", action="store_true", help="Use reference VAE for comparison")
    parser.add_argument("--cogvideox_model_path", type=str, default="CogVideoX-2b", help="Path to CogVideoX-2b model")
    parser.add_argument("--output_format", type=str, default="enhanced", choices=["simple", "enhanced", "complete"], help="Output format")
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = TAEHVInference(
        model_path=args.model_path,
        device=args.device,
        use_ref_vae=args.use_ref_vae,
        cogvideox_path=args.cogvideox_model_path
    )
    
    # 运行推理
    inferencer.run_inference(
        dataset_path=args.data_root,
        annotation_file=args.annotation_file,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        output_format=args.output_format
    )


if __name__ == "__main__":
    main()
