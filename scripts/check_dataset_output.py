#!/usr/bin/env python3
"""
检查MiniDataset输出
用于诊断数据范围异常问题

Usage:
    python scripts/check_dataset_output.py \
        --annotation_file /path/to/annotations.json \
        --data_root /path/to/videos \
        --num_samples 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from training.dataset import MiniDataset


def check_dataset_sample(dataset, idx):
    """检查单个dataset样本
    
    Args:
        dataset: MiniDataset实例
        idx: 样本索引
    
    Returns:
        dict: 检查结果
    """
    result = {
        'index': idx,
        'success': False,
        'shape': None,
        'dtype': None,
        'device': None,
        'range': None,
        'mean': None,
        'std': None,
        'min': None,
        'max': None,
        'error': None
    }
    
    try:
        # 获取样本
        video = dataset[idx]
        
        result['success'] = True
        result['shape'] = list(video.shape)
        result['dtype'] = str(video.dtype)
        result['device'] = str(video.device)
        
        # 统计信息
        result['min'] = float(video.min())
        result['max'] = float(video.max())
        result['mean'] = float(video.float().mean())
        result['std'] = float(video.float().std())
        result['range'] = [result['min'], result['max']]
        
    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()
    
    return result


def print_result(result, annotation=None):
    """打印检查结果"""
    print(f"\n{'='*80}")
    print(f"📦 样本 #{result['index']}")
    if annotation:
        print(f"   文件: {annotation.get('path', 'unknown')}")
    print(f"{'='*80}")
    
    if result['error']:
        print(f"❌ 错误: {result['error']}")
        if result.get('traceback'):
            print(f"\n详细错误:")
            print(result['traceback'])
        return
    
    if not result['success']:
        print(f"❌ 读取失败")
        return
    
    # 基本信息
    print(f"✅ 读取成功")
    print(f"\n📊 Tensor信息:")
    print(f"  • Shape: {result['shape']}")
    print(f"  • Dtype: {result['dtype']}")
    print(f"  • Device: {result['device']}")
    
    # 统计信息
    print(f"\n🎨 数据统计:")
    print(f"  • 范围: [{result['min']:.6f}, {result['max']:.6f}]")
    print(f"  • 均值: {result['mean']:.6f}")
    print(f"  • 标准差: {result['std']:.6f}")
    
    # 诊断
    print(f"\n✓ 诊断:")
    
    issues = []
    
    # 检查dtype
    if 'uint8' in result['dtype']:
        print(f"  ✅ Dtype正确: uint8 (原始像素值)")
        expected_range = [0, 255]
    elif 'float' in result['dtype']:
        print(f"  ℹ️  Dtype: float (可能已归一化)")
        if result['max'] <= 1.1:
            expected_range = [0, 1]
        else:
            expected_range = [0, 255]
    else:
        issues.append(f"⚠️  意外的dtype: {result['dtype']}")
        expected_range = None
    
    # 检查范围
    min_val, max_val = result['min'], result['max']
    
    if max_val < 0.01:
        issues.append("🔴 严重: 最大值 < 0.01 - 数据几乎全是0！")
        issues.append("   → 可能是视频读取失败或全黑帧")
    elif 'uint8' in result['dtype'] and max_val <= 255:
        print(f"  ✅ 范围正确: [0, 255] (uint8)")
        if max_val < 10:
            issues.append("⚠️  警告: 最大值 < 10 - 数据异常低（uint8类型）")
            issues.append("   → 可能是暗场景或读取问题")
    elif 'float' in result['dtype'] and max_val <= 1.1 and max_val >= 0.9:
        print(f"  ✅ 范围正确: [0, 1] (已归一化)")
    elif 'float' in result['dtype'] and max_val > 200:
        print(f"  ⚠️  Float类型但范围像uint8: [{min_val:.1f}, {max_val:.1f}]")
        issues.append("   → 可能需要归一化 (/ 255.0)")
    else:
        issues.append(f"⚠️  范围异常: [{min_val:.6f}, {max_val:.6f}]")
    
    # 检查均值
    mean_val = result['mean']
    if 'uint8' in result['dtype']:
        if mean_val < 10:
            issues.append("⚠️  均值过低 (< 10) - 图像过暗")
        elif mean_val > 200:
            issues.append("⚠️  均值过高 (> 200) - 图像过亮")
        else:
            print(f"  ✅ 均值正常: {mean_val:.2f} (uint8)")
    else:  # float
        if result['max'] <= 1.1:  # 归一化后
            if mean_val < 0.1:
                issues.append("⚠️  均值过低 (< 0.1) - 图像过暗")
            elif mean_val > 0.9:
                issues.append("⚠️  均值过高 (> 0.9) - 图像过亮")
            else:
                print(f"  ✅ 均值正常: {mean_val:.4f} (归一化)")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"  ✅ 所有检查通过")


def main():
    parser = argparse.ArgumentParser(description='检查MiniDataset输出')
    parser.add_argument('--annotation_file', type=str, required=True,
                      help='Annotation JSON文件路径')
    parser.add_argument('--data_root', type=str, required=True,
                      help='视频文件根目录')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='检查的样本数量')
    parser.add_argument('--height', type=int, default=480,
                      help='目标高度')
    parser.add_argument('--width', type=int, default=720,
                      help='目标宽度')
    parser.add_argument('--n_frames', type=int, default=12,
                      help='帧数')
    parser.add_argument('--output', type=str, default=None,
                      help='保存结果的JSON文件路径')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print(f"🔍 检查MiniDataset输出")
    print(f"{'='*80}\n")
    
    # 创建Dataset
    print(f"📖 创建MiniDataset...")
    print(f"  • annotation_file: {args.annotation_file}")
    print(f"  • data_root: {args.data_root}")
    print(f"  • 请求尺寸: {args.height}x{args.width}")
    
    # MiniDataset使用patch_hw参数（正方形patch）
    # 取height和width的较小值作为patch_hw
    patch_hw = min(args.height, args.width)
    print(f"  • 实际patch尺寸: {patch_hw}x{patch_hw}")
    print(f"  • 帧数: {args.n_frames}")
    
    try:
        
        dataset = MiniDataset(
            annotation_file=args.annotation_file,
            data_dir=args.data_root,
            patch_hw=patch_hw,
            n_frames=args.n_frames,
            augmentation=False  # 不使用增强，方便检查
        )
    except Exception as e:
        print(f"\n❌ 创建Dataset失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"✅ Dataset创建成功，包含 {len(dataset)} 个样本\n")
    
    # 读取annotations用于显示文件名
    with open(args.annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # 选择要检查的样本
    num_samples = min(args.num_samples, len(dataset))
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(dataset)), num_samples)
    
    print(f"🔍 检查 {num_samples} 个随机样本...\n")
    
    # 检查每个样本
    results = []
    normal_count = 0
    abnormal_count = 0
    
    for idx in sample_indices:
        ann = annotations[idx] if idx < len(annotations) else None
        result = check_dataset_sample(dataset, idx)
        results.append(result)
        
        print_result(result, annotation=ann)
        
        # 统计
        if result['success']:
            max_val = result['max']
            if max_val < 0.01:
                abnormal_count += 1
            elif max_val > 0.9:  # 正常范围（归一化或原始）
                normal_count += 1
            else:
                abnormal_count += 1
    
    # 总结
    print(f"\n{'='*80}")
    print(f"📊 检查总结")
    print(f"{'='*80}")
    print(f"总样本数: {num_samples}")
    print(f"正常: {normal_count} ({normal_count/num_samples*100:.1f}%)")
    print(f"异常: {abnormal_count} ({abnormal_count/num_samples*100:.1f}%)")
    
    # 分析结果
    if results and results[0]['success']:
        first_max = results[0]['max']
        first_dtype = results[0]['dtype']
        
        print(f"\n📝 Dataset输出特征:")
        print(f"  • Dtype: {first_dtype}")
        print(f"  • 典型范围: [0, {first_max:.4f}]")
        
        if 'uint8' in first_dtype and first_max <= 255:
            print(f"\n✅ Dataset输出未归一化 [0, 255]")
            print(f"   → 训练脚本需要归一化: batch.float() / 255.0")
        elif 'float' in first_dtype and first_max <= 1.1:
            print(f"\n✅ Dataset输出已归一化 [0, 1]")
            print(f"   → 训练脚本不应再除以255!")
            print(f"   → 检查 training/taehv_train.py 中的归一化代码")
        elif first_max < 0.01:
            print(f"\n🔴 严重问题: Dataset输出几乎全是0!")
            print(f"   可能原因:")
            print(f"   1. 视频读取失败")
            print(f"   2. MiniDataset实现有bug")
            print(f"   3. 数据集文件损坏")
        else:
            print(f"\n⚠️  Dataset输出范围异常")
            print(f"   需要检查 MiniDataset 的实现")
    
    # 保存结果
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'summary': {
                    'total': num_samples,
                    'normal': normal_count,
                    'abnormal': abnormal_count,
                },
                'results': results
            }, f, indent=2)
        
        print(f"\n💾 结果已保存到: {output_path}")
    
    # 下一步建议
    print(f"\n{'='*80}")
    print(f"下一步:")
    
    if abnormal_count == 0 and 'uint8' in first_dtype:
        print(f"  ✅ Dataset正常输出 uint8 [0, 255]")
        print(f"  → 检查训练脚本是否正确归一化")
        print(f"  → 查看 training/taehv_train.py 搜索 'batch.float()'")
    elif abnormal_count == 0 and 'float' in first_dtype and first_max <= 1.1:
        print(f"  ⚠️  Dataset已归一化到 [0, 1]")
        print(f"  → 训练脚本可能重复归一化！")
        print(f"  → 检查并修改: batch.float() / 255.0 → batch.float()")
    elif first_max < 0.01:
        print(f"  🔴 Dataset输出异常 (几乎全是0)")
        print(f"  → 检查 training/data/video_dataset.py 的实现")
        print(f"  → 可能需要修改视频读取逻辑")
    else:
        print(f"  ⚠️  需要进一步诊断")
        print(f"  → 运行 scripts/trace_data_pipeline.py")
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

