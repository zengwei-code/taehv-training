#!/usr/bin/env python3
"""
MiniDataset数据加载器
专门用于处理驾驶场景视频数据
"""

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm


class MiniDataset(Dataset):
    """MiniDataset数据加载器"""
    
    def __init__(self, 
                 annotation_file: str,
                 data_dir: str,
                 patch_hw: int,
                 n_frames: int,
                 min_frame_delta: int = 1,
                 max_frame_delta: int = 3,
                 augmentation: bool = True,
                 cache_videos: bool = False):
        """
        Args:
            annotation_file: JSON注解文件路径
            data_dir: 视频文件目录
            patch_hw: patch大小
            n_frames: 加载帧数
            min_frame_delta: 最小帧间隔
            max_frame_delta: 最大帧间隔
            augmentation: 是否启用数据增强
            cache_videos: 是否缓存视频到内存（小数据集推荐）
        """
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)['list']
            
        self.data_dir = Path(data_dir)
        self.patch_hw = patch_hw
        self.n_frames = n_frames
        self.min_frame_delta = min_frame_delta
        self.max_frame_delta = max_frame_delta
        self.augmentation = augmentation
        self.cache_videos = cache_videos
        
        # 过滤存在的文件
        valid_annotations = []
        for ann in self.annotations:
            video_path = self.data_dir / ann['path']
            if video_path.exists():
                valid_annotations.append({
                    **ann,
                    'full_path': str(video_path)
                })
        
        self.annotations = valid_annotations
        
        # 缓存视频（如果启用）
        self.video_cache = {}
        if self.cache_videos and len(self.annotations) <= 100:  # 只为小数据集启用缓存
            print("Caching videos to memory...")
            for ann in tqdm(self.annotations[:50]):  # 限制缓存数量
                try:
                    frames = self._load_video_frames(ann['full_path'], cache_mode=True)
                    self.video_cache[ann['full_path']] = frames
                except Exception as e:
                    print(f"Failed to cache {ann['path']}: {e}")
        
        print(f"Loaded {len(self.annotations)} valid videos from {annotation_file}")
        if self.cache_videos:
            print(f"Cached {len(self.video_cache)} videos in memory")
    
    def __len__(self):
        return len(self.annotations)
    
    def _load_video_frames(self, video_path: str, cache_mode: bool = False) -> List[np.ndarray]:
        """从视频加载所有帧（用于缓存）或指定数量的帧"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if not cache_mode and len(frames) >= self.n_frames * 10:  # 限制内存使用
                break
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        
        return frames
    
    def _sample_frames(self, frames: List[np.ndarray], n_frames: int) -> List[Image.Image]:
        """从帧列表中采样指定数量的帧"""
        if len(frames) < n_frames:
            # 如果帧数不够，重复最后一帧
            frames = frames + [frames[-1]] * (n_frames - len(frames))
        
        # 随机选择起始点和间隔
        frame_delta = random.randint(self.min_frame_delta, self.max_frame_delta)
        max_start = max(0, len(frames) - n_frames * frame_delta)
        start_idx = random.randint(0, max_start) if max_start > 0 else 0
        
        selected_frames = []
        for i in range(n_frames):
            idx = min(start_idx + i * frame_delta, len(frames) - 1)
            frame = frames[idx]
            selected_frames.append(Image.fromarray(frame))
        
        return selected_frames
    
    def _load_frames_from_video(self, video_path: str, n_frames: int) -> List[Image.Image]:
        """从视频文件直接加载帧"""
        frame_delta = random.randint(self.min_frame_delta, self.max_frame_delta)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 随机选择起始帧
        if total_frames > n_frames * frame_delta:
            start_frame = random.randint(0, total_frames - n_frames * frame_delta)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for i in range(n_frames):
            # 跳过frame_delta帧
            for _ in range(frame_delta if i > 0 else 1):
                ret, frame = cap.read()
                if not ret:
                    # 如果到达视频末尾，重新开始
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
            
            if frame is not None:
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        
        cap.release()
        return frames
    
    def _apply_augmentation(self, images: List[Image.Image]) -> List[Image.Image]:
        """应用数据增强"""
        if not self.augmentation:
            return images
        
        # 随机水平翻转
        if random.random() < 0.5:
            images = [TF.hflip(img) for img in images]
        
        # 颜色抖动
        if random.random() < 0.7:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            saturation = random.uniform(0.9, 1.1)
            hue = random.uniform(-0.05, 0.05)
            
            images = [TF.adjust_brightness(img, brightness) for img in images]
            images = [TF.adjust_contrast(img, contrast) for img in images]
            images = [TF.adjust_saturation(img, saturation) for img in images]
            images = [TF.adjust_hue(img, hue) for img in images]
        
        return images
    
    def _random_resize_crop(self, images: List[Image.Image], target_size: int) -> List[Image.Image]:
        """随机裁剪和缩放"""
        # 填充小图像
        min_size = min(images[0].size)
        if min_size < target_size:
            pad_x = max(0, target_size - images[0].width)
            pad_y = max(0, target_size - images[0].height)
            images = [TF.pad(img, (pad_x//2, pad_y//2, pad_x-pad_x//2, pad_y-pad_y//2), 
                           padding_mode="reflect") for img in images]
        
        # 随机裁剪尺寸
        crop_scale = random.uniform(0.8, 1.0) if self.augmentation else 1.0
        crop_size = max(target_size, int(min(images[0].size) * crop_scale))
        
        i = random.randint(0, max(0, images[0].height - crop_size))
        j = random.randint(0, max(0, images[0].width - crop_size))
        
        cropped = [TF.crop(img, i, j, crop_size, crop_size) for img in images]
        resized = [TF.resize(img, (target_size, target_size), antialias=True) for img in cropped]
        
        return resized
    
    def __getitem__(self, idx: int) -> th.Tensor:
        ann = self.annotations[idx]
        video_path = ann['full_path']
        
        try:
            # 加载视频帧
            if video_path in self.video_cache:
                # 从缓存加载
                cached_frames = self.video_cache[video_path]
                frames = self._sample_frames(cached_frames, self.n_frames)
            else:
                # 直接从文件加载
                frames = self._load_frames_from_video(video_path, self.n_frames)
            
            # 应用增强
            frames = self._apply_augmentation(frames)
            
            # 随机裁剪
            frames = self._random_resize_crop(frames, self.patch_hw)
            
            # 转换为tensor
            frames = [TF.to_tensor(frame) for frame in frames]
            frames_tensor = th.stack(frames, 0)  # T, C, H, W
            
            return frames_tensor
            
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            # 返回随机索引的数据作为fallback
            return self.__getitem__((idx + 1) % len(self.annotations))
    
    def get_annotation(self, idx: int) -> dict:
        """获取指定索引的注解信息"""
        return self.annotations[idx]
    
    def create_dataloader(self, 
                         batch_size: int,
                         num_workers: int = 4,
                         shuffle: bool = True,
                         pin_memory: bool = True,
                         drop_last: bool = True) -> DataLoader:
        """创建DataLoader"""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last
        )


def create_train_val_datasets(annotation_file: str,
                             data_dir: str,
                             patch_hw: int,
                             n_frames: int,
                             train_ratio: float = 0.9,
                             **kwargs) -> Tuple[MiniDataset, MiniDataset]:
    """创建训练集和验证集"""
    
    # 读取所有注解
    with open(annotation_file, 'r', encoding='utf-8') as f:
        all_annotations = json.load(f)['list']
    
    # 随机划分
    random.shuffle(all_annotations)
    split_idx = int(len(all_annotations) * train_ratio)
    
    train_annotations = all_annotations[:split_idx]
    val_annotations = all_annotations[split_idx:]
    
    # 创建临时文件
    train_file = annotation_file.replace('.json', '_train_temp.json')
    val_file = annotation_file.replace('.json', '_val_temp.json')
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump({'total': len(train_annotations), 'list': train_annotations}, f, ensure_ascii=False, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump({'total': len(val_annotations), 'list': val_annotations}, f, ensure_ascii=False, indent=2)
    
    # 创建数据集
    train_dataset = MiniDataset(train_file, data_dir, patch_hw, n_frames, augmentation=True, **kwargs)
    val_dataset = MiniDataset(val_file, data_dir, patch_hw, n_frames, augmentation=False, **kwargs)
    
    # 清理临时文件
    os.remove(train_file)
    os.remove(val_file)
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # 测试数据集
    dataset = MiniDataset(
        "/data/matrix-project/MiniDataset/stage1_annotations_500.json",
        "/data/matrix-project/MiniDataset/data",
        64, 8
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试加载
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    
    # 测试DataLoader
    dataloader = dataset.create_dataloader(batch_size=2, num_workers=0)
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        break
