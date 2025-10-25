# Inference.py 工作原理与评估指南

## 📖 目录
1. [Inference.py Pipeline 详解](#inference-pipeline)
2. [如何验证修复效果](#验证修复效果)
3. [使用 Evaluation 工具](#使用evaluation工具)
4. [快速开始指南](#快速开始)

---

## 🔍 Inference Pipeline 详解

### 核心架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference.py Pipeline                     │
└─────────────────────────────────────────────────────────────┘

输入 → 数据加载 → 模型推理 → 指标计算 → 可视化输出
 │        │           │           │           │
 │        │           │           │           └─→ 图片/视频
 │        │           │           └─→ PSNR/SSIM/LPIPS
 │        │           └─→ 编码/解码
 │        └─→ MiniDataset
 └─→ 原始视频
```

### 详细数据流

```python
# ============================================
# 完整数据流追踪
# ============================================

Step 1: 输入
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- annotation_file: JSON文件路径
- data_root: 视频文件目录
- model_path: 训练好的模型checkpoint

Step 2: 数据加载 (Line 444)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dataset = MiniDataset(
    annotation_file=annotation_file,
    data_dir=dataset_path,
    patch_hw=128,      # patch大小
    n_frames=12        # 帧数
)
↓
输出: Dataset对象 (500个样本)

Step 3: 单样本处理 (Line 455-457)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sample = dataset[i]                        # [T, C, H, W]
frames = sample.unsqueeze(0).float()       # [1, T, C, H, W]
                                           # ✅ 已经是 [0,1] (修复后)
frames = frames.to(device)                 # 移到GPU
↓
数据范围: [0, 1] float32
形状: [1, 12, 3, 128, 128]

Step 4: 模型编码 (Line 465)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
encoded = model.encode_video(frames, parallel=True)
↓
潜在表示 (latent): [1, latent_dim, ...]
压缩比: ~8-16x

Step 5: 模型解码 (Line 471)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
decoded = model.decode_video(encoded, parallel=True)
↓
重建视频: [1, 12, 3, 128, 128], [0, 1]

Step 6: 指标计算 (Line 477)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
metrics = calculate_comprehensive_metrics(frames, decoded)
↓
- MSE: 均方误差
- PSNR: 峰值信噪比 (越高越好, >30 excellent)
- SSIM: 结构相似度 (越接近1越好)
- LPIPS: 感知损失 (越低越好)

Step 7: 输出生成 (Line 497-503)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 原始视频
- 重建视频
- 对比视频 (side-by-side)
- 潜在表示可视化
- 指标报告 (JSON)
- HTML报告
```

### 关键组件说明

#### 1. TAEHVInference 类

```python
class TAEHVInference:
    """完整的TAEHV推理器"""
    
    def __init__(self, model_path, device="cuda"):
        # 组件1: 加载模型
        self.model = self.load_model(model_path)
        
        # 组件2: 初始化指标计算器
        self.ssim = StructuralSimilarityIndexMeasure()
        self.lpips = LearnedPerceptualImagePatchSimilarity()
```

**作用**: 封装了模型加载、推理、指标计算的完整流程

#### 2. 核心方法

| 方法 | 作用 | 输入 | 输出 |
|------|------|------|------|
| `load_model()` | 加载训练好的模型 | model_path | TAEHV模型 |
| `calculate_comprehensive_metrics()` | 计算质量指标 | original, reconstructed | metrics dict |
| `run_inference()` | 执行完整推理流程 | dataset配置 | 结果文件 |
| `create_structured_output()` | 生成可视化输出 | frames, metrics | 图片/视频 |

#### 3. 输入参数

```python
# inference.py 接受的参数
--model_path        # 模型checkpoint路径 (必需)
--data_root         # 视频数据目录
--annotation_file   # JSON注解文件
--num_samples       # 测试样本数 (默认10)
--output_dir        # 输出目录
--device            # 设备 (cuda/cpu)
--output_format     # 输出格式 (simple/enhanced/complete)
--use_ref_vae       # 是否使用参考VAE对比
```

#### 4. 输出结构

```
output_dir/
├── sample_000_original.mp4         # 原始视频
├── sample_000_reconstructed.mp4    # 重建视频
├── sample_000_comparison.mp4       # 对比视频
├── sample_000_latents.png          # 潜在表示可视化
├── sample_000_metrics.json         # 详细指标
├── inference_results.json          # 总体结果
└── inference_report.html           # HTML报告
```

---

## ✅ 验证修复效果

### 方法 1: 直接使用 inference.py（需要已训练模型）

**前提**: 你需要有一个训练好的模型checkpoint

#### Step 1: 检查是否有可用模型

```bash
# 查找所有模型文件
find outputs -name "*.safetensors" -o -name "*.pth" 2>/dev/null

# 查看最新的模型
ls -lht outputs/*/checkpoints/*.pth | head -5
```

#### Step 2: 运行推理测试

```bash
# 假设找到了模型: outputs/taehv_run1/checkpoints/checkpoint-1000.pth

python inference.py \
    --model_path outputs/taehv_run1/checkpoints/checkpoint-1000.pth \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --data_root /data/matrix-project/MiniDataset/data \
    --num_samples 10 \
    --output_dir evaluation/evaluation_results/after_fix \
    --output_format enhanced
```

#### Step 3: 查看生成的图片

```bash
# 查看输出目录
ls -lh evaluation/evaluation_results/after_fix/

# 应该看到:
# - sample_000_original.mp4
# - sample_000_reconstructed.mp4
# - sample_000_comparison.mp4
# - ...
```

#### 期望结果（修复后）

✅ **成功标志**:
- 重建视频**不是全黑**
- PSNR > 15 (理想情况 > 25)
- SSIM > 0.5 (理想情况 > 0.8)
- 视觉上可以看到内容

❌ **失败标志**（修复前的症状）:
- 重建视频全黑或接近全黑
- PSNR < 10
- SSIM < 0.3
- 完全看不到内容

---

## 🎯 使用 Evaluation 工具

### Option 1: evaluate_vae.py（最全面）

**功能**: 完整的模型评估，包括定量指标和可视化

```bash
# 基础用法
python evaluation/evaluate_vae.py \
    --model_path <模型路径> \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --data_root /data/matrix-project/MiniDataset/data \
    --num_samples 50 \
    --output_dir evaluation/evaluation_results/full_eval
```

**输出**:
- 定量指标报告
- 重建质量可视化
- 潜在空间分析
- 对比图表

### Option 2: quick_evaluate.py（一键评估）

**功能**: 自动化评估流程

```bash
# 进入evaluation目录
cd evaluation

# 运行快速评估
python quick_evaluate.py \
    --model_path ../outputs/taehv_run1/checkpoints/checkpoint-1000.pth \
    --config ../training/configs/taehv_config_a800.py \
    --num_samples 50 \
    --output_dir evaluation_results/quick_eval
```

**包含**:
1. 训练日志分析
2. 模型定量评估
3. 生成评估报告

### Option 3: quick_evaluate.sh（Shell脚本）

**功能**: 预配置的评估脚本

```bash
# 查看脚本内容
cat evaluation/quick_evaluate.sh

# 修改脚本中的路径
# MODEL_PATH="..."
# OUTPUT_DIR="..."

# 运行
bash evaluation/quick_evaluate.sh
```

---

## 🚀 快速开始：验证修复效果

### 场景 1: 没有训练好的模型

**无法使用 inference.py，但可以验证数据**

```bash
# 已经完成！你之前运行的这个就是验证
python scripts/check_dataset_output.py \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --data_root /data/matrix-project/MiniDataset/data \
    --num_samples 5

# 结果: ✅ 数据范围 [0, 1]，修复成功
```

**下一步**: 训练一个模型

```bash
# 快速训练测试（100步）
accelerate launch --config_file configs/accelerate_config.yaml \
    training/taehv_train.py \
    --config training/configs/taehv_config_a800.py
```

### 场景 2: 有训练好的模型

**Step 1**: 找到模型路径

```bash
MODEL_PATH=$(find outputs -name "*.pth" -type f | head -1)
echo "找到模型: $MODEL_PATH"
```

**Step 2**: 运行推理生成重建图片

```bash
# 使用 inference.py
python inference.py \
    --model_path "$MODEL_PATH" \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --data_root /data/matrix-project/MiniDataset/data \
    --num_samples 10 \
    --output_dir evaluation/verification_after_fix \
    --output_format enhanced

echo ""
echo "✅ 重建完成！查看结果:"
echo "  ls -lh evaluation/verification_after_fix/"
```

**Step 3**: 查看生成的图片

```bash
# 查看输出文件
ls -lh evaluation/verification_after_fix/

# 用图片查看器打开
# 或者复制到本地查看
```

**Step 4**: 分析结果

```bash
# 查看指标摘要
cat evaluation/verification_after_fix/inference_results.json | grep -A 5 "average"

# 期望看到:
# "psnr": > 20  ✅
# "ssim": > 0.6 ✅
# "mse": < 0.01 ✅
```

---

## 📊 修复前后对比实验（可选）

如果你想直观对比修复前后的效果：

### Step 1: 保存当前（修复后）的结果

```bash
# 运行推理
python inference.py \
    --model_path <模型路径> \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --data_root /data/matrix-project/MiniDataset/data \
    --num_samples 5 \
    --output_dir evaluation/after_fix

# 保存指标
cp evaluation/after_fix/inference_results.json evaluation/metrics_after_fix.json
```

### Step 2: 临时恢复旧代码测试（仅用于对比）

**⚠️ 警告**: 仅用于对比实验，测试完立即改回！

```bash
# 备份修复后的代码
cp inference.py inference.py.fixed

# 临时恢复旧代码（手动编辑）
# 在 inference.py 456行改为:
# frames = sample.unsqueeze(0).float() / 255.0  # 恢复旧bug
```

### Step 3: 运行对比测试

```bash
# 用旧代码运行（应该生成全黑图像）
python inference.py \
    --model_path <模型路径> \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --data_root /data/matrix-project/MiniDataset/data \
    --num_samples 5 \
    --output_dir evaluation/before_fix

# 保存指标
cp evaluation/before_fix/inference_results.json evaluation/metrics_before_fix.json

# ⚠️ 立即恢复修复后的代码！
cp inference.py.fixed inference.py
```

### Step 4: 对比结果

```bash
# 对比指标
echo "修复前:"
cat evaluation/metrics_before_fix.json | grep -A 3 "average"

echo ""
echo "修复后:"
cat evaluation/metrics_after_fix.json | grep -A 3 "average"

# 对比图片
ls -lh evaluation/before_fix/sample_000_reconstructed.mp4
ls -lh evaluation/after_fix/sample_000_reconstructed.mp4

# before_fix 应该是全黑
# after_fix 应该有正常内容
```

---

## 💡 实用命令速查

### 快速验证数据范围

```bash
# 验证 MiniDataset 输出（2分钟）
python scripts/check_dataset_output.py \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --data_root /data/matrix-project/MiniDataset/data \
    --num_samples 3
```

### 查找可用模型

```bash
# 查找所有模型
find outputs -name "*.pth" -o -name "*.safetensors"

# 查找最新模型
ls -lt outputs/*/checkpoints/*.pth | head -1
```

### 运行推理生成图片

```bash
# 替换 <MODEL_PATH> 为实际路径
python inference.py \
    --model_path <MODEL_PATH> \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --data_root /data/matrix-project/MiniDataset/data \
    --num_samples 10 \
    --output_dir evaluation/test_results
```

### 查看结果

```bash
# 列出生成的文件
ls -lh evaluation/test_results/

# 查看指标
cat evaluation/test_results/inference_results.json

# 查看HTML报告（如果生成了）
# 在浏览器中打开 evaluation/test_results/inference_report.html
```

---

## 🎯 关键要点

### Inference.py 的核心

1. **输入**: 视频数据 + 训练好的模型
2. **处理**: 编码 → 潜在表示 → 解码 → 重建
3. **输出**: 重建视频 + 质量指标 + 可视化

### 修复验证的关键

1. **数据层面**: MiniDataset 输出 [0, 1] ✅
2. **模型层面**: 推理时不再重复归一化 ✅
3. **效果层面**: 重建图片不是全黑 ✅

### 评估工具选择

| 工具 | 用途 | 时间 |
|------|------|------|
| `check_dataset_output.py` | 验证数据范围 | 2分钟 |
| `inference.py` | 生成重建图片 | 5-10分钟 |
| `evaluate_vae.py` | 完整模型评估 | 10-30分钟 |
| `quick_evaluate.py` | 一键全面评估 | 20-40分钟 |

---

## 📚 相关文档

- `CHANGES.md` - 变更日志与修复合并记录
- `TROUBLESHOOTING.md` - 故障排除（环境/数据/模型）
- `数据范围检查使用指南.md` - 数据与PSNR诊断

---

**现在你可以开始验证了！** 🚀

如果没有训练好的模型，先训练一个；如果有模型，直接运行 inference.py 生成重建图片验证修复效果。

