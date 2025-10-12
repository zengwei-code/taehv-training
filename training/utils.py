#!/usr/bin/env python3
"""
训练工具函数
"""

import torch
import torch.nn as nn
from typing import Optional


class RefVAE(nn.Module):
    """参考VAE模型封装"""
    
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16, cogvideox_model_path: Optional[str] = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.latent_channels = 16
        self.time_downscale = 4
        self.space_downscale = 8
        
        try:
            from diffusers import AutoencoderKLCogVideoX
            # 使用配置中指定的CogVideoX-2b模型路径
            model_path = cogvideox_model_path if cogvideox_model_path else "THUDM/CogVideoX-2b"
            self.vae = AutoencoderKLCogVideoX.from_pretrained(
                model_path, 
                subfolder="vae", 
                torch_dtype=dtype
            )
            self.vae_type = "cogvideox"
            print(f"✓ Loaded CogVideoX VAE from {'local path' if cogvideox_model_path else 'HuggingFace Hub'}: {model_path}")
        except Exception as e:
            print(f"Failed to load CogVideoX VAE: {e}")
            try:
                from diffusers import AutoencoderKLWan
                self.vae = AutoencoderKLWan.from_pretrained(
                    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", 
                    subfolder="vae", 
                    torch_dtype=dtype
                )
                self.vae_type = "wan"
                print("✓ Loaded Wan VAE")
            except Exception as e2:
                print(f"Failed to load Wan VAE: {e2}")
                print("Warning: No reference VAE available, will use dummy VAE")
                self.vae = None
                self.vae_type = "dummy"
    
    @torch.no_grad()
    def encode_video(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码视频到latent空间
        Args:
            x: NTCHW RGB视频张量，范围[0,1]
        Returns:
            NTCHW latent张量
        """
        if self.vae is None:
            # Dummy encoding for testing
            N, T, C, H, W = x.shape
            return torch.randn(N, T, self.latent_channels, H//8, W//8, 
                              device=x.device, dtype=x.dtype)
        
        assert x.ndim == 5, f"Expected NTCHW input, got {x.shape}"
        
        # 转换为 NCTHW 格式
        x = x.transpose(1, 2)  # N,T,C,H,W -> N,C,T,H,W
        
        # 转换到VAE的数据范围 [-1, 1]
        y = x.to(self.dtype).mul(2).sub_(1)
        
        try:
            # 编码
            latent_dist = self.vae.encode(y).latent_dist
            latent = latent_dist.sample()
            
            # 转回 NTCHW 格式
            latent = latent.transpose(1, 2)  # N,C,T,H,W -> N,T,C,H,W
            
            return latent.to(x.dtype)
            
        except Exception as e:
            print(f"VAE encoding failed: {e}")
            # 返回dummy结果
            N, C, T, H, W = y.shape
            return torch.randn(N, T, self.latent_channels, H//8, W//8,
                              device=x.device, dtype=x.dtype)
    
    @torch.no_grad()
    def decode_video(self, x: torch.Tensor) -> torch.Tensor:
        """
        从latent空间解码视频
        Args:
            x: NTCHW latent张量
        Returns:
            NTCHW RGB视频张量，范围[0,1]
        """
        if self.vae is None:
            # Dummy decoding for testing
            N, T, C, H, W = x.shape
            return torch.rand(N, T, 3, H*8, W*8, device=x.device, dtype=x.dtype)
        
        assert x.ndim == 5, f"Expected NTCHW input, got {x.shape}"
        
        # 转换为 NCTHW 格式
        x = x.transpose(1, 2)  # N,T,C,H,W -> N,C,T,H,W
        
        try:
            # 解码
            y = x.to(self.dtype)
            decoded = self.vae.decode(y).sample
            
            # 转换数据范围 [-1,1] -> [0,1]
            decoded = decoded.mul_(0.5).add_(0.5).clamp_(0, 1)
            
            # 转回 NTCHW 格式
            decoded = decoded.transpose(1, 2)  # N,C,T,H,W -> N,T,C,H,W
            
            return decoded.to(x.dtype)
            
        except Exception as e:
            print(f"VAE decoding failed: {e}")
            # 返回dummy结果
            N, C, T, H, W = x.shape
            return torch.rand(N, T, 3, H*8, W*8, device=x.device, dtype=x.dtype)


def get_ref_vae(device: str = "cuda", dtype: torch.dtype = torch.bfloat16, cogvideox_model_path: Optional[str] = None) -> RefVAE:
    """获取参考VAE模型"""
    return RefVAE(device=device, dtype=dtype, cogvideox_model_path=cogvideox_model_path)


def calculate_model_size(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters()) / 1e6


def log_model_info(model, logger):
    """记录模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {total_params / 1e6:.2f}M parameters")


def create_model_card(output_dir: str, config):
    """创建模型卡片"""
    model_card_content = f"""
# TAEHV Model

This model is a fine-tuned version of TAEHV trained on driving scene videos.

## Model Details
- Base model: TAEHV (Tiny AutoEncoder for Hunyuan Video)
- Training data: MiniDataset driving scenes
- Training steps: {config.max_train_steps}
- Batch size: {config.train_batch_size}
- Learning rate: {config.learning_rate}

## Usage

```python
from models.taehv import TAEHV
import torch

# Load model
model = TAEHV(None)
model.load_state_dict(torch.load('final_model.pth'))

# Encode video
encoded = model.encode_video(video_tensor)

# Decode video
decoded = model.decode_video(encoded)
```

## Training Configuration
- Optimizer: {config.optimizer}
- Learning rate scheduler: {config.lr_scheduler}
- Mixed precision: {config.mixed_precision}
- Gradient accumulation steps: {config.gradient_accumulation_steps}
"""
    
    with open(f"{output_dir}/README.md", "w") as f:
        f.write(model_card_content)
