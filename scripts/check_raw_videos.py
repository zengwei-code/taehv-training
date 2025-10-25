#!/usr/bin/env python3
"""
检查原始视频文件
用于诊断数据范围异常问题

Usage:
    python scripts/check_raw_videos.py \
        --annotation_file /path/to/annotations.json \
        --data_root /path/to/videos \
        --num_samples 5
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import decord
from decord import VideoReader, cpu
import cv2

# 设置decord使用CPU
decord.bridge.set_bridge('torch')


def check_video_file(video_path, method='decord'):
    """检查单个视频文件
    
    Args:
        video_path: 视频文件路径
        method: 读取方法 'decord' 或 'opencv'
    
    Returns:
        dict: 检查结果
    """
    result = {
        'path': str(video_path),
        'exists': False,
        'readable': False,
        'num_frames': 0,
        'resolution': None,
        'fps': 0,
        'pixel_range': None,
        'pixel_mean': None,
        'pixel_std': None,
        'dtype': None,
        'error': None
    }
    
    # 检查文件是否存在
    if not os.path.exists(video_path):
        result['error'] = 'File not found'
        return result
    
    result['exists'] = True
    
    try:
        if method == 'decord':
            # 使用decord读取
            vr = VideoReader(str(video_path), ctx=cpu(0))
            result['num_frames'] = len(vr)
            result['resolution'] = f"{vr[0].shape[1]}x{vr[0].shape[0]}"
            result['fps'] = vr.get_avg_fps()
            
            # 读取前10帧检查像素值
            num_frames_to_check = min(10, len(vr))
            frames = []
            for i in range(num_frames_to_check):
                frame = vr[i].numpy()  # [H, W, C]
                frames.append(frame)
            
            frames = np.stack(frames, axis=0)  # [T, H, W, C]
            
        elif method == 'opencv':
            # 使用opencv读取
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                result['error'] = 'Cannot open with OpenCV'
                return result
            
            result['num_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result['fps'] = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result['resolution'] = f"{width}x{height}"
            
            # 读取前10帧
            frames = []
            for i in range(min(10, result['num_frames'])):
                ret, frame = cap.read()
                if not ret:
                    break
                # OpenCV读取的是BGR，转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            
            if not frames:
                result['error'] = 'Cannot read frames'
                return result
            
            frames = np.stack(frames, axis=0)  # [T, H, W, C]
        
        else:
            result['error'] = f'Unknown method: {method}'
            return result
        
        # 分析像素值
        result['readable'] = True
        result['dtype'] = str(frames.dtype)
        result['pixel_range'] = [float(frames.min()), float(frames.max())]
        result['pixel_mean'] = float(frames.mean())
        result['pixel_std'] = float(frames.std())
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def print_result(result, detailed=False):
    """打印检查结果"""
    print(f"\n{'='*80}")
    print(f"📹 视频: {Path(result['path']).name}")
    print(f"{'='*80}")
    
    if not result['exists']:
        print(f"❌ 文件不存在: {result['path']}")
        return
    
    if result['error']:
        print(f"❌ 错误: {result['error']}")
        return
    
    if not result['readable']:
        print(f"❌ 无法读取")
        return
    
    # 基本信息
    print(f"✅ 文件存在且可读")
    print(f"\n📊 基本信息:")
    print(f"  • 帧数: {result['num_frames']}")
    print(f"  • 分辨率: {result['resolution']}")
    print(f"  • 帧率: {result['fps']:.2f} fps")
    
    # 像素值统计
    print(f"\n🎨 像素值统计:")
    print(f"  • 数据类型: {result['dtype']}")
    print(f"  • 范围: [{result['pixel_range'][0]:.4f}, {result['pixel_range'][1]:.4f}]")
    print(f"  • 均值: {result['pixel_mean']:.4f}")
    print(f"  • 标准差: {result['pixel_std']:.4f}")
    
    # 判断是否正常
    min_val, max_val = result['pixel_range']
    mean_val = result['pixel_mean']
    
    print(f"\n✓ 诊断:")
    
    issues = []
    
    if max_val < 10:
        issues.append("⚠️  像素值异常低（< 10）- 可能是黑色视频或读取错误")
    elif max_val > 250 and max_val <= 255:
        print(f"  ✅ 像素范围正常 [0, 255]")
    elif max_val > 0.9 and max_val <= 1.0:
        print(f"  ✅ 像素范围正常 [0, 1] (已归一化)")
    else:
        issues.append(f"⚠️  像素范围异常 [{min_val:.4f}, {max_val:.4f}]")
    
    if mean_val < 10 and max_val > 200:
        issues.append("⚠️  均值过低 - 可能是大部分为黑色")
    elif mean_val > 200:
        issues.append("⚠️  均值过高 - 可能是大部分为白色")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"  ✅ 所有检查通过")


def main():
    parser = argparse.ArgumentParser(description='检查原始视频文件')
    parser.add_argument('--annotation_file', type=str, required=True,
                      help='Annotation JSON文件路径')
    parser.add_argument('--data_root', type=str, required=True,
                      help='视频文件根目录')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='检查的样本数量')
    parser.add_argument('--method', type=str, default='decord',
                      choices=['decord', 'opencv'],
                      help='视频读取方法')
    parser.add_argument('--output', type=str, default=None,
                      help='保存结果的JSON文件路径')
    
    args = parser.parse_args()
    
    # 读取annotation文件
    print(f"📖 读取annotation文件: {args.annotation_file}")
    
    if not os.path.exists(args.annotation_file):
        print(f"❌ Annotation文件不存在: {args.annotation_file}")
        sys.exit(1)
    
    with open(args.annotation_file, 'r') as f:
        data = json.load(f)
    
    # 支持两种格式：直接列表或包含'list'键的字典
    if isinstance(data, dict) and 'list' in data:
        annotations = data['list']
    elif isinstance(data, list):
        annotations = data
    else:
        print(f"❌ 无法识别的annotation格式")
        sys.exit(1)
    
    print(f"✅ 找到 {len(annotations)} 个视频")
    
    # 选择要检查的样本
    num_samples = min(args.num_samples, len(annotations))
    import random
    random.seed(42)
    sample_indices = random.sample(range(len(annotations)), num_samples)
    
    print(f"\n🔍 检查 {num_samples} 个随机样本...")
    print(f"使用方法: {args.method}")
    
    # 检查每个样本
    results = []
    normal_count = 0
    abnormal_count = 0
    
    for idx in sample_indices:
        ann = annotations[idx]
        video_path = os.path.join(args.data_root, ann['path'])
        
        result = check_video_file(video_path, method=args.method)
        results.append(result)
        
        print_result(result, detailed=True)
        
        # 统计
        if result['readable'] and result['pixel_range']:
            max_val = result['pixel_range'][1]
            if max_val > 200:  # 正常范围
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
    
    if abnormal_count > num_samples * 0.5:
        print(f"\n🔴 警告: 超过50%的视频异常！")
        print(f"   可能的原因:")
        print(f"   1. 视频文件损坏")
        print(f"   2. 视频解码器问题")
        print(f"   3. 数据集准备错误")
        print(f"\n   建议: 检查数据集准备过程或尝试其他解码器")
    elif abnormal_count > 0:
        print(f"\n⚠️  发现 {abnormal_count} 个异常视频")
        print(f"   建议: 检查这些视频文件")
    else:
        print(f"\n✅ 所有视频正常！")
        print(f"   像素范围符合预期 [0, 255]")
        print(f"   数据范围异常问题可能在Dataset或训练脚本中")
    
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
                    'method': args.method
                },
                'results': results
            }, f, indent=2)
        
        print(f"\n💾 结果已保存到: {output_path}")
    
    print(f"\n{'='*80}")
    print(f"下一步:")
    print(f"  1. 如果视频正常 → 运行 scripts/check_dataset_output.py")
    print(f"  2. 如果视频异常 → 检查数据集准备过程")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

