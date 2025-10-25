#!/usr/bin/env python3
"""
独立的数据范围和模型配置检查脚本

用于事后分析checkpoint，诊断PSNR过低等问题。

使用方法:
    python scripts/check_model_data_range.py \
        --model_path output/xxx/checkpoint-1000/model.pth \
        --config training/configs/taehv_config_a800.py \
        --data_root /path/to/data \
        --annotation_file /path/to/annotations.json \
        --output_dir ./check_results
"""

import argparse
import sys
import os
from pathlib import Path
import json
from typing import Dict, Any, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
import logging

from models.taehv import TAEHV
from training.dataset import MiniDataset


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Check model data range and configuration")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="checkpoints/taecvx.pth",
        help="Path to pretrained base model (default: checkpoints/taecvx.pth)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Data directory (overrides config)"
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        help="Annotation file (overrides config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./check_results",
        help="Output directory for check results"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to check"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for checking"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    return parser.parse_args()


def load_config(config_path: str):
    """加载配置文件"""
    config_path = Path(config_path)
    sys.path.append(str(config_path.parent))
    
    # 动态导入配置
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    return config_module.args


def check_model_config(config, logger) -> Dict[str, Any]:
    """检查模型配置"""
    logger.info("="*79)
    logger.info("⚙️  Model Configuration Check")
    logger.info("="*79)
    
    config_info = {}
    
    # 数据配置
    logger.info("\n📂 Data Configuration:")
    data_config = {}
    for key in ['height', 'width', 'n_frames', 'batch_size', 'augmentation', 'data_root', 'annotation_file']:
        if hasattr(config, key):
            value = getattr(config, key)
            data_config[key] = value
            logger.info(f"  {key}: {value}")
    config_info['data'] = data_config
    
    # 模型配置
    logger.info("\n🏗️  Model Configuration:")
    model_config = {}
    for key in ['model_type', 'use_seraena']:
        if hasattr(config, key):
            value = getattr(config, key)
            model_config[key] = value
            logger.info(f"  {key}: {value}")
    config_info['model'] = model_config
    
    # 训练配置
    logger.info("\n🎯 Training Configuration:")
    train_config = {}
    for key in ['learning_rate', 'lr_scheduler', 'max_train_steps', 'gradient_accumulation_steps']:
        if hasattr(config, key):
            value = getattr(config, key)
            train_config[key] = value
            logger.info(f"  {key}: {value}")
    config_info['training'] = train_config
    
    logger.info("\n" + "="*79)
    
    return config_info


def check_data_range(input_videos: torch.Tensor, reconstructions: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
    """检查数据范围
    
    Args:
        input_videos: 输入视频 tensor [B, T, C, H, W]
        reconstructions: 重建视频 tensor [B, T, C, H, W]
        batch_idx: 批次索引
        
    Returns:
        检查结果字典
    """
    result = {
        "batch_idx": batch_idx,
        "input": {},
        "reconstruction": {},
        "errors": {},
        "warnings": []
    }
    
    # 输入数据统计
    result["input"]["shape"] = list(input_videos.shape)
    result["input"]["dtype"] = str(input_videos.dtype)
    result["input"]["min"] = float(input_videos.min())
    result["input"]["max"] = float(input_videos.max())
    result["input"]["mean"] = float(input_videos.mean())
    result["input"]["std"] = float(input_videos.std())
    
    # 重建数据统计
    result["reconstruction"]["shape"] = list(reconstructions.shape)
    result["reconstruction"]["dtype"] = str(reconstructions.dtype)
    result["reconstruction"]["min"] = float(reconstructions.min())
    result["reconstruction"]["max"] = float(reconstructions.max())
    result["reconstruction"]["mean"] = float(reconstructions.mean())
    result["reconstruction"]["std"] = float(reconstructions.std())
    
    # 计算误差（调整形状以匹配）
    # 如果帧数不同，截取相同的帧数
    min_frames = min(input_videos.shape[1], reconstructions.shape[1])
    input_aligned = input_videos[:, :min_frames]
    recon_aligned = reconstructions[:, :min_frames]
    
    mse = float(torch.nn.functional.mse_loss(input_aligned, recon_aligned))
    mae = float(torch.nn.functional.l1_loss(input_aligned, recon_aligned))
    max_diff = float((input_aligned - recon_aligned).abs().max())
    
    result["errors"]["mse"] = mse
    result["errors"]["mae"] = mae
    result["errors"]["rmse"] = mse ** 0.5
    result["errors"]["max_diff"] = max_diff
    
    # 异常检测
    warnings = []
    
    # 检查1: 输入范围
    if result["input"]["min"] < -0.1 or result["input"]["max"] > 1.1:
        warnings.append({
            "type": "input_range",
            "severity": "high",
            "message": f"Input range [{result['input']['min']:.4f}, {result['input']['max']:.4f}] outside expected [0, 1]"
        })
    
    # 检查2: 重建范围
    if result["reconstruction"]["min"] < -0.1 or result["reconstruction"]["max"] > 1.1:
        warnings.append({
            "type": "recon_range",
            "severity": "high",
            "message": f"Reconstruction range [{result['reconstruction']['min']:.4f}, {result['reconstruction']['max']:.4f}] outside expected [0, 1]"
        })
    
    # 检查3: 高MSE
    if mse > 0.1:
        warnings.append({
            "type": "high_mse",
            "severity": "medium",
            "message": f"High MSE: {mse:.6f} (explains low PSNR)"
        })
    
    # 检查4: 范围不匹配
    input_range = result["input"]["max"] - result["input"]["min"]
    recon_range = result["reconstruction"]["max"] - result["reconstruction"]["min"]
    if abs(input_range - recon_range) > 0.3:
        warnings.append({
            "type": "range_mismatch",
            "severity": "medium",
            "message": f"Range mismatch - Input: {input_range:.4f}, Recon: {recon_range:.4f}"
        })
    
    result["warnings"] = warnings
    
    return result


def print_check_result(result: Dict[str, Any], logger, detailed: bool = True):
    """打印检查结果"""
    logger.info("\n" + "="*79)
    logger.info(f"🔍 Data Range Check (Batch {result['batch_idx']})")
    logger.info("="*79)
    
    # 输入数据
    logger.info("\n📥 Input Videos:")
    logger.info(f"  Shape: {result['input']['shape']}")
    logger.info(f"  Range: [{result['input']['min']:.4f}, {result['input']['max']:.4f}]  (Expected: [0, 1])")
    logger.info(f"  Mean: {result['input']['mean']:.4f} ± {result['input']['std']:.4f}")
    logger.info(f"  Dtype: {result['input']['dtype']}")
    
    # 重建数据
    logger.info("\n📤 Reconstructions:")
    logger.info(f"  Shape: {result['reconstruction']['shape']}")
    logger.info(f"  Range: [{result['reconstruction']['min']:.4f}, {result['reconstruction']['max']:.4f}]")
    logger.info(f"  Mean: {result['reconstruction']['mean']:.4f} ± {result['reconstruction']['std']:.4f}")
    logger.info(f"  Dtype: {result['reconstruction']['dtype']}")
    
    # 误差
    logger.info("\n📊 Reconstruction Errors:")
    logger.info(f"  MSE:  {result['errors']['mse']:.6f}")
    logger.info(f"  RMSE: {result['errors']['rmse']:.6f}")
    logger.info(f"  MAE:  {result['errors']['mae']:.6f}")
    logger.info(f"  Max Diff: {result['errors']['max_diff']:.6f}")
    
    # 估算PSNR
    if result['errors']['mse'] > 0:
        psnr = -10 * torch.log10(torch.tensor(result['errors']['mse'])).item()
        logger.info(f"  Estimated PSNR: {psnr:.2f} dB")
    
    # 警告
    if result["warnings"]:
        logger.info(f"\n⚠️  Warnings ({len(result['warnings'])}):")
        for warn in result["warnings"]:
            severity_emoji = "🔴" if warn["severity"] == "high" else "🟡"
            logger.info(f"  {severity_emoji} [{warn['type']}] {warn['message']}")
    else:
        logger.info("\n✅ No warnings - Data ranges look correct!")
    
    logger.info("="*79)


def main():
    logger = setup_logging()
    args = parse_args()
    
    logger.info("="*79)
    logger.info("🔍 Model Data Range and Configuration Checker")
    logger.info("="*79)
    
    # 1. 加载配置
    logger.info(f"\n📂 Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # 覆盖数据路径（如果提供）
    if args.data_root:
        config.data_root = args.data_root
        logger.info(f"   Overriding data_root: {args.data_root}")
    if args.annotation_file:
        config.annotation_file = args.annotation_file
        logger.info(f"   Overriding annotation_file: {args.annotation_file}")
    
    # 检查配置
    logger.info("\n⚙️  Checking model configuration...")
    config_info = check_model_config(config, logger)
    
    # 2. 加载模型
    logger.info(f"\n🏗️  Loading model...")
    
    # 首先尝试加载预训练基础模型
    if Path(args.pretrained_path).exists():
        logger.info(f"   Loading pretrained base model from {args.pretrained_path}")
        try:
            model = TAEHV(checkpoint_path=args.pretrained_path)
            logger.info("   ✅ Pretrained base model loaded successfully")
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load pretrained model: {e}")
            logger.info("   Creating model without pretrained weights...")
            model = TAEHV(checkpoint_path=None)
    else:
        logger.warning(f"   ⚠️  Pretrained model not found at {args.pretrained_path}")
        logger.info("   Creating model without pretrained weights...")
        model = TAEHV(checkpoint_path=None)
    
    # 然后加载训练好的 checkpoint（如果提供）
    if args.model_path and Path(args.model_path).exists():
        logger.info(f"\n🔄 Loading trained checkpoint from {args.model_path}")
        try:
            state_dict = torch.load(args.model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info("   ✅ Trained checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"   ❌ Error loading trained checkpoint: {e}")
            logger.info("   Continuing with pretrained weights only...")
    elif args.model_path:
        logger.error(f"   ❌ Trained checkpoint not found: {args.model_path}")
        logger.info("   Continuing with pretrained weights only...")
    
    model = model.to(args.device)
    model.eval()
    
    # 3. 加载数据集
    logger.info(f"\n📊 Loading dataset...")
    logger.info(f"   Data root: {config.data_root}")
    logger.info(f"   Annotation file: {config.annotation_file}")
    
    try:
        dataset = MiniDataset(
            annotation_file=config.annotation_file,
            data_dir=config.data_root,
            patch_hw=getattr(config, 'height', 480),
            n_frames=getattr(config, 'n_frames', 12),
            augmentation=False
        )
        
        # 限制样本数量
        if len(dataset.annotations) > args.num_samples:
            dataset.annotations = dataset.annotations[:args.num_samples]
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"   ✅ Loaded {len(dataset)} samples")
        
    except Exception as e:
        logger.error(f"   ❌ Failed to load dataset: {e}")
        logger.error(f"   Please check data_root and annotation_file paths")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. 执行检查
    logger.info(f"\n🔍 Performing data range checks on {args.num_samples} samples...")
    logger.info("="*79)
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            logger.info(f"\nProcessing batch {batch_idx + 1}/{len(dataloader)}...")
            
            # 预处理 - MiniDataset输出uint8 [0,255]，需要归一化
            videos = batch.float() / 255.0  # 归一化到 [0,1]
            videos = videos.to(args.device)
            
            # 模型推理
            try:
                encoded = model.encode_video(videos, parallel=True, show_progress_bar=False)
                decoded = model.decode_video(encoded, parallel=True, show_progress_bar=False)
                
                # 执行检查
                result = check_data_range(videos, decoded, batch_idx)
                
                # 打印结果（第一个batch打印详细信息）
                print_check_result(result, logger, detailed=(batch_idx == 0))
                
                # 保存结果
                result_path = output_dir / f"batch_{batch_idx}_result.json"
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                all_results.append(result)
                
            except Exception as e:
                logger.error(f"   ❌ Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
    
    # 6. 汇总结果
    logger.info("\n" + "="*79)
    logger.info("📊 Summary of All Checks")
    logger.info("="*79)
    
    if all_results:
        # 统计警告
        total_warnings = sum(len(r["warnings"]) for r in all_results)
        high_severity_warnings = sum(
            len([w for w in r["warnings"] if w["severity"] == "high"])
            for r in all_results
        )
        
        logger.info(f"\nTotal batches checked: {len(all_results)}")
        logger.info(f"Total warnings: {total_warnings}")
        logger.info(f"High severity warnings: {high_severity_warnings}")
        
        # 计算平均统计
        avg_input_range = sum(r["input"]["max"] - r["input"]["min"] for r in all_results) / len(all_results)
        avg_recon_range = sum(r["reconstruction"]["max"] - r["reconstruction"]["min"] for r in all_results) / len(all_results)
        
        logger.info(f"\nAverage input range: {avg_input_range:.4f}")
        logger.info(f"Average reconstruction range: {avg_recon_range:.4f}")
        
        # 误差统计
        avg_mse = sum(r["errors"]["mse"] for r in all_results) / len(all_results)
        avg_mae = sum(r["errors"]["mae"] for r in all_results) / len(all_results)
        
        logger.info(f"\nAverage MSE: {avg_mse:.6f}")
        logger.info(f"Average MAE: {avg_mae:.6f}")
        
        # 估算PSNR
        if avg_mse > 0:
            estimated_psnr = -10 * torch.log10(torch.tensor(avg_mse)).item()
            logger.info(f"Estimated PSNR: {estimated_psnr:.2f} dB")
        
        # 打印所有高严重性警告
        if high_severity_warnings > 0:
            logger.info(f"\n⚠️  High Severity Warnings:")
            for idx, result in enumerate(all_results):
                high_warns = [w for w in result["warnings"] if w["severity"] == "high"]
                for warn in high_warns:
                    logger.info(f"   Batch {idx}: [{warn['type']}] {warn['message']}")
        
        # 保存汇总
        summary = {
            "config": config_info,
            "num_batches": len(all_results),
            "total_warnings": total_warnings,
            "high_severity_warnings": high_severity_warnings,
            "avg_input_range": float(avg_input_range),
            "avg_recon_range": float(avg_recon_range),
            "avg_mse": float(avg_mse),
            "avg_mae": float(avg_mae),
            "estimated_psnr": float(estimated_psnr) if avg_mse > 0 else None
        }
        
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\n💾 Saved summary to {summary_path}")
        
        # 保存完整结果
        full_results_path = output_dir / "full_check_results.json"
        with open(full_results_path, 'w') as f:
            json.dump({
                "summary": summary,
                "batch_results": all_results
            }, f, indent=2)
        logger.info(f"💾 Saved full results to {full_results_path}")
    
    logger.info("\n" + "="*79)
    logger.info("✅ Check completed!")
    logger.info(f"📁 Results saved to: {output_dir}")
    logger.info("="*79)


if __name__ == "__main__":
    main()
