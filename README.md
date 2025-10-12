# 🎬 TAEHV Training Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive training framework for **TAEHV (Tiny AutoEncoder for Video)** with advanced evaluation tools and H100 optimization.

## 🌟 Features

- 🚀 **Complete Training Pipeline** - From data loading to model deployment
- 📊 **Advanced Evaluation Suite** - PSNR, SSIM, LPIPS metrics with visualization
- ⚡ **H100 Optimized** - DeepSpeed integration and mixed precision training  
- 🔧 **Easy Configuration** - Centralized config management
- 📖 **Comprehensive Documentation** - Detailed guides and troubleshooting
- 🎯 **Seraena Integration** - Optional adversarial training for enhanced quality

## 📁 Project Structure

```
taehv-training/
├── 🏋️ training/           # Training scripts and configs
│   ├── taehv_train.py     # Main training script
│   ├── dataset.py         # MiniDataset data loader
│   ├── utils.py           # Training utilities
│   └── configs/           # Configuration files
│       └── taehv_config_h100.py
├── 📊 evaluation/         # Evaluation tools and analysis
│   ├── evaluate_vae.py    # Model evaluation script
│   ├── quick_evaluate.py  # One-click evaluation
│   ├── README.md          # Evaluation guide
│   └── TROUBLESHOOTING.md # Issue resolution
├── 🧠 models/             # Model architectures
│   ├── taehv.py          # TAEHV model implementation
│   └── seraena.py        # Adversarial training (optional)
├── ⚙️ accelerate_configs/ # Distributed training configs
│   └── deepspeed_8gpu.yaml
├── 📚 docs/              # Additional documentation
├── 🔧 checkpoints/       # Pretrained model weights
└── 🎯 inference.py       # Inference and testing script
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/taehv-training.git
cd taehv-training

# Create conda environment
conda env create -f environment.yml
conda activate tiny-vae

# Install additional dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your video data as follows:
```
/path/to/your/data/
├── video1.mp4
├── video2.mp4
├── video3.mp4
└── annotations.json
```

The `annotations.json` should contain:
```json
{
  "total": 500,
  "list": [
    {
      "full_path": "/path/to/video1.mp4",
      "duration": 10.5,
      "frames": 315
    }
  ]
}
```

### 3. Configuration

Edit the training configuration in `training/configs/taehv_config_h100.py`:

```python
# Dataset paths
args.data_root = "/path/to/your/data"
args.annotation_file = "/path/to/annotations.json"

# Training parameters
args.train_batch_size = 2        # Adjust based on GPU memory
args.learning_rate = 5e-5
args.max_train_steps = 100000

# Model parameters
args.patch_size = 1
args.latent_channels = 16

# Loss weights (IMPORTANT for reconstruction quality)
args.reconstruction_loss_weight = 10.0  # Increase for better reconstruction
args.perceptual_loss_weight = 1.0
```

### 4. Training

```bash
# Start training with H100 optimization
./train_taehv_h100.sh

# Or manual launch
accelerate launch \
    --config_file accelerate_configs/deepspeed_8gpu.yaml \
    training/taehv_train.py \
    --config training/configs/taehv_config_h100.py
```

### 5. Evaluation

```bash
cd evaluation

# Quick evaluation (recommended)
python quick_evaluate.py \
    --data_root /path/to/your/data \
    --annotation_file /path/to/annotations.json

# Detailed evaluation
python evaluate_vae.py \
    --model_path ../output/2025-XX-XX_XX-XX-XX/final_model.pth \
    --config ../training/configs/taehv_config_h100.py \
    --num_samples 100 \
    --data_root /path/to/your/data \
    --annotation_file /path/to/annotations.json
```

## 📊 Evaluation Metrics

The evaluation suite provides comprehensive quality assessment:

| Metric | Description | Excellent | Good | Fair | Poor |
|--------|-------------|-----------|------|------|------|
| **PSNR** | Peak Signal-to-Noise Ratio | >30 dB | 25-30 dB | 20-25 dB | <20 dB |
| **SSIM** | Structural Similarity | >0.90 | 0.80-0.90 | 0.70-0.80 | <0.70 |
| **LPIPS** | Perceptual Similarity | <0.10 | 0.10-0.20 | 0.20-0.30 | >0.30 |

### Sample Results

After proper training, you should expect:
- **PSNR**: 28-35 dB
- **SSIM**: 0.85-0.95
- **LPIPS**: 0.05-0.15
- **Overall Score**: 70-90/100

## 🛠️ Configuration Guide

### Key Training Parameters

```python
# For high-quality reconstruction (recommended)
args.reconstruction_loss_weight = 10.0  # Primary reconstruction objective
args.perceptual_loss_weight = 1.0       # Perceptual quality
args.encoder_loss_weight = 0.1          # Distribution matching
args.seraena_loss_weight = 0.1          # Adversarial training

# For fast convergence
args.learning_rate = 1e-4
args.lr_scheduler = "constant_with_warmup"
args.lr_warmup_steps = 1000

# For memory optimization
args.train_batch_size = 1               # Reduce if OOM
args.gradient_accumulation_steps = 8    # Maintain effective batch size
args.gradient_checkpointing = True      # Save memory
```

### Hardware-Specific Settings

**Single GPU (< 16GB VRAM):**
```python
args.train_batch_size = 1
args.gradient_accumulation_steps = 8
args.height = 320
args.width = 480
```

**8x H100 (Recommended):**
```python
args.train_batch_size = 2
args.gradient_accumulation_steps = 8
args.height = 480
args.width = 720
args.use_deepspeed = True
```

## 📈 Monitoring Training

### TensorBoard

```bash
# View training progress
tensorboard --logdir logs/taehv_h100_production --port 6006
```

### Key Metrics to Watch

- `train/reconstruction_loss`: Should decrease steadily (target: <0.01)
- `train/encoder_loss`: Distribution matching quality
- `train/seraena_loss`: Adversarial training progress
- GPU memory usage and training speed

## 🔧 Troubleshooting

### Common Issues

#### 1. Poor Reconstruction Quality (PSNR < 10)

**Problem**: Model not learning to reconstruct properly
**Solution**: 
```python
# Increase reconstruction loss weight
args.reconstruction_loss_weight = 10.0  # or higher
args.perceptual_loss_weight = 1.0
```

#### 2. Out of Memory Errors

**Problem**: CUDA OOM during training
**Solution**:
```python
# Reduce memory usage
args.train_batch_size = 1
args.gradient_checkpointing = True
args.height = 320  # Reduce resolution
```

#### 3. Training Stalls/No Learning

**Problem**: Loss stops decreasing after initial epochs
**Solution**:
```python
# Adjust learning rate schedule
args.learning_rate = 1e-4
args.lr_scheduler = "constant_with_warmup"
```

#### 4. Environment Issues

**Problem**: Package conflicts or missing dependencies
**Solution**:
```bash
# Recreate environment
conda env remove -n tiny-vae
conda env create -f environment.yml
conda activate tiny-vae
pip install -r requirements.txt
```

For detailed troubleshooting, see: [evaluation/TROUBLESHOOTING.md](evaluation/TROUBLESHOOTING.md)

## 📚 Documentation

- 📖 [**Evaluation Guide**](evaluation/README.md) - Complete evaluation documentation
- 🔧 [**Troubleshooting**](evaluation/TROUBLESHOOTING.md) - Common issues and solutions
- 🎯 [**Model Diagnosis**](evaluation/DIAGNOSIS_REPORT.md) - Performance analysis guide
- 📊 [**Training Tips**](docs/) - Best practices and optimization

## 🎯 Advanced Features

### Seraena Adversarial Training

Enable enhanced reconstruction quality with adversarial training:

```python
args.use_seraena = True
args.seraena_loss_weight = 0.3
args.n_seraena_frames = 1  # Recommended for stability
```

### Reference VAE Integration

Use CogVideoX as reference for improved training:

```python
args.use_ref_vae = True
args.ref_vae_dtype = "bfloat16"
args.cogvideox_model_path = "CogVideoX-2b"
```

### Model Inference

```bash
# Basic inference test
python inference.py \
    --model_path output/final_model.pth \
    --num_samples 10 \
    --output_format simple

# Enhanced inference with full analysis
python inference.py \
    --model_path output/final_model.pth \
    --num_samples 50 \
    --use_ref_vae \
    --output_format complete
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black . && isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [TAEHV Original Implementation](https://github.com/wilson1yan/teco) - Base model architecture
- [CogVideoX](https://github.com/THUDM/CogVideo) - Reference VAE implementation
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Distributed training optimization
- [Hugging Face Accelerate](https://github.com/huggingface/accelerate) - Training utilities

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@misc{taehv-training-2025,
  title={TAEHV Training Framework: A Comprehensive Video Autoencoder Training Suite},
  author={Your Name},
  year={2025},
  url={https://github.com/YOUR_USERNAME/taehv-training}
}
```

---

**⭐ If this project helps you, please consider giving it a star!**

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/taehv-training.svg?style=social&label=Star)](https://github.com/YOUR_USERNAME/taehv-training)
[![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/taehv-training.svg?style=social&label=Fork)](https://github.com/YOUR_USERNAME/taehv-training/fork)
