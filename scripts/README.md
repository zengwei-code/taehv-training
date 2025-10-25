# Scripts 目录说明

本目录包含用于诊断和检查 TAEHV 训练过程中数据范围问题的诊断脚本。

---

## 📋 目录结构

```
scripts/
├── README.md                      # 本文档
├── diagnose_data_range.sh         # 🔧 一键诊断脚本（推荐使用）
├── check_raw_videos.py            # 检查原始视频文件
├── check_dataset_output.py        # 检查 MiniDataset 输出
├── check_model_data_range.py      # 检查模型数据范围
└── notebooks/
    └── TAEHV_Training_Example.ipynb  # 训练示例 Notebook
```

---

## 🎯 脚本用途

### 1. diagnose_data_range.sh（推荐）

**用途**: 一键诊断数据范围异常问题的完整工具链

**适用场景**:
- PSNR 异常低（< 15 dB）
- 训练 loss 不下降
- 重建图像质量极差
- 怀疑数据范围有问题

**使用方法**:

```bash
# 基本用法（使用默认配置）
bash scripts/diagnose_data_range.sh

# 查看输出
cat diagnosis_results/diagnosis_report.txt
```

**默认配置**:
- Annotation 文件: `/data/matrix-project/MiniDataset/stage1_annotations_500.json`
- 数据目录: `/data/matrix-project/MiniDataset/data`
- 检查样本数: 5
- 输出目录: `diagnosis_results/`

**自定义配置**: 编辑脚本中的配置变量

```bash
# 修改这些变量
ANNOTATION_FILE="/path/to/annotations.json"
DATA_ROOT="/path/to/data"
NUM_SAMPLES=10
OUTPUT_DIR="my_diagnosis"
```

**输出文件**:
- `diagnosis_results/raw_videos_check.json` - 原始视频检查结果
- `diagnosis_results/dataset_output_check.json` - Dataset 输出检查结果
- `diagnosis_results/diagnosis_report.txt` - 综合诊断报告

**诊断流程**:
1. ✅ 检查原始视频文件（是否存在、可读、像素范围）
2. ✅ 检查 MiniDataset 输出（数据类型、范围、统计信息）
3. ✅ 生成诊断报告（问题定位和修复建议）

---

### 2. check_raw_videos.py

**用途**: 检查原始视频文件的像素范围和基本信息

**适用场景**:
- 验证视频文件是否损坏
- 检查视频读取是否正常
- 确认原始数据范围

**使用方法**:

```bash
python scripts/check_raw_videos.py \
    --annotation_file /path/to/annotations.json \
    --data_root /path/to/videos \
    --num_samples 5 \
    --method decord \
    --output results/raw_videos_check.json
```

**参数说明**:
- `--annotation_file`: Annotation JSON 文件路径（必需）
- `--data_root`: 视频文件根目录（必需）
- `--num_samples`: 检查的样本数量（默认: 5）
- `--method`: 视频读取方法，可选 `decord` 或 `opencv`（默认: decord）
- `--output`: 保存结果的 JSON 文件路径（可选）

**检查项目**:
- ✅ 视频文件是否存在
- ✅ 视频是否可读
- ✅ 分辨率、帧数、帧率
- ✅ 像素范围 [0, 255]
- ✅ 均值、标准差
- ✅ 异常检测（黑屏、过暗、过亮）

**输出示例**:

```
================================================================================
📹 视频: sample_video.mp4
================================================================================
✅ 文件存在且可读

📊 基本信息:
  • 帧数: 120
  • 分辨率: 1920x1080
  • 帧率: 30.00 fps

🎨 像素值统计:
  • 数据类型: uint8
  • 范围: [0.0000, 255.0000]
  • 均值: 127.4521
  • 标准差: 65.2134

✓ 诊断:
  ✅ 像素范围正常 [0, 255]
  ✅ 所有检查通过
```

---

### 3. check_dataset_output.py

**用途**: 检查 MiniDataset 的输出数据范围和格式

**适用场景**:
- 原始视频正常，但怀疑 Dataset 处理有问题
- 检查是否重复归一化
- 验证数据增强是否正确

**使用方法**:

```bash
python scripts/check_dataset_output.py \
    --annotation_file /path/to/annotations.json \
    --data_root /path/to/videos \
    --num_samples 5 \
    --height 480 \
    --width 720 \
    --n_frames 12 \
    --output results/dataset_output_check.json
```

**参数说明**:
- `--annotation_file`: Annotation JSON 文件路径（必需）
- `--data_root`: 视频文件根目录（必需）
- `--num_samples`: 检查的样本数量（默认: 5）
- `--height`: 目标高度（默认: 480）
- `--width`: 目标宽度（默认: 720）
- `--n_frames`: 帧数（默认: 12）
- `--output`: 保存结果的 JSON 文件路径（可选）

**检查项目**:
- ✅ Tensor shape、dtype、device
- ✅ 数据范围（最小值、最大值）
- ✅ 统计信息（均值、标准差）
- ✅ 归一化状态检测
- ✅ 异常诊断

**输出示例**:

```
================================================================================
📦 样本 #0
   文件: videos/sample.mp4
================================================================================
✅ 读取成功

📊 Tensor信息:
  • Shape: [12, 3, 480, 480]
  • Dtype: torch.uint8
  • Device: cpu

🎨 数据统计:
  • 范围: [0.000000, 255.000000]
  • 均值: 127.452100
  • 标准差: 65.213400

✓ 诊断:
  ✅ Dtype正确: uint8 (原始像素值)
  ✅ 范围正确: [0, 255] (uint8)
  ✅ 均值正常: 127.45 (uint8)
  ✅ 所有检查通过

================================================================================
📊 检查总结
================================================================================
总样本数: 5
正常: 5 (100.0%)
异常: 0 (0.0%)

📝 Dataset输出特征:
  • Dtype: torch.uint8
  • 典型范围: [0, 255.0000]

✅ Dataset输出未归一化 [0, 255]
   → 训练脚本需要归一化: batch.float() / 255.0
```

**诊断逻辑**:

| 情况 | 诊断结果 | 建议 |
|------|---------|------|
| uint8 & 范围 [0, 255] | ✅ 正常，未归一化 | 训练脚本需要 `/ 255.0` |
| float & 范围 [0, 1] | ✅ 正常，已归一化 | 训练脚本**不要**再 `/ 255.0` |
| 最大值 < 0.01 | 🔴 严重问题 | 检查视频读取逻辑 |
| 其他异常范围 | ⚠️ 需要检查 | 查看 MiniDataset 实现 |

---

### 4. check_model_data_range.py

**用途**: 对已训练的模型进行事后数据范围检查

**适用场景**:
- 训练完成后分析 checkpoint
- PSNR 异常低，需要诊断原因
- 验证模型输入输出范围

**⚠️ 注意**: 
- 此脚本已重写为**独立版本**，不依赖外部工具模块
- 所有检查功能都在脚本内部实现
- 详见 [修复说明](./FIX_check_model_data_range.md)

**使用方法**:

```bash
python scripts/check_model_data_range.py \
    --model_path output/checkpoint-1000/model.pth \
    --config training/configs/taehv_config_a800.py \
    --data_root /path/to/data \
    --annotation_file /path/to/annotations.json \
    --output_dir ./check_results \
    --num_samples 5 \
    --batch_size 2 \
    --device cuda:0
```

**参数说明**:
- `--model_path`: 训练好的模型 checkpoint 文件路径（必需）
- `--config`: 配置文件路径（必需）
- `--pretrained_path`: 预训练基础模型路径（默认: `checkpoints/taecvx.pth`）
- `--data_root`: 数据目录（可选，覆盖配置文件）
- `--annotation_file`: Annotation 文件（可选，覆盖配置文件）
- `--output_dir`: 结果输出目录（默认: `./check_results`）
- `--num_samples`: 检查样本数（默认: 5）
- `--batch_size`: 批次大小（默认: 2）
- `--device`: 运行设备（默认: `cuda:0`）

**检查项目**:
1. ✅ 模型配置检查
2. ✅ 输入数据范围
3. ✅ 重建输出范围
4. ✅ 重建误差（MSE、MAE）
5. ✅ 估算 PSNR
6. ✅ 异常警告

**输出文件**:
- `check_results/batch_0_result.json` - 各批次检查结果
- `check_results/summary.json` - 汇总统计
- `check_results/full_check_history.json` - 完整历史记录

**输出示例**:

```
===============================================================================
🔍 Model Data Range and Configuration Checker
===============================================================================

📂 Loading configuration from training/configs/taehv_config_a800.py
⚙️  Checking model configuration...
🏗️  Loading model from output/checkpoint-1000/model.pth
   ✅ Model loaded successfully

📊 Loading dataset...
   ✅ Loaded 5 samples

🔍 Performing data range checks on 5 samples...

Processing batch 1/3...

===============================================================================
🔍 Data Range Check (Step 0, check)
===============================================================================

📥 Input Videos:
  Range: [0.0123, 0.9877]  (Expected: [0, 1])
  Mean: 0.4521 ± 0.2134

📤 Reconstructions:
  Range: [0.0089, 0.9932]
  Mean: 0.4498 ± 0.2156

📊 Reconstruction Errors:
  MSE:  0.023456
  MAE:  0.124532
  
✅ No warnings

===============================================================================
📊 Summary of All Checks
===============================================================================

Total batches checked: 3
Total warnings: 0
High severity warnings: 0

Average input range: 0.9754
Average reconstruction range: 0.9843
Average MSE: 0.021234
Average MAE: 0.118765
Estimated PSNR: 16.73 dB

✅ Check completed!
📁 Results saved to: ./check_results
```

---

### 5. notebooks/TAEHV_Training_Example.ipynb

**用途**: Jupyter Notebook 训练示例（参考用）

**内容**:
- 使用 Seraena discriminator 的训练示例
- Dataset 加载和可视化
- VAE 编解码演示
- 训练循环示例代码

**注意**:
- 这是一个**独立的训练示例**，使用不同的训练方法
- 实际项目使用 `training/taehv_train.py` 进行训练
- 可以作为理解 TAEHV 模型的参考

**使用方法**:

```bash
cd scripts/notebooks
jupyter notebook TAEHV_Training_Example.ipynb
```

---

## 🚀 快速诊断流程

### 场景 1: PSNR 异常低

```bash
# 1. 运行一键诊断
bash scripts/diagnose_data_range.sh

# 2. 查看诊断报告
cat diagnosis_results/diagnosis_report.txt

# 3. 根据报告建议修复问题
```

### 场景 2: 检查训练好的模型

```bash
# 检查 checkpoint
python scripts/check_model_data_range.py \
    --model_path output/xxx/checkpoint-1000/model.pth \
    --config training/configs/taehv_config_a800.py \
    --output_dir model_check_results
```

### 场景 3: 只检查数据集

```bash
# 单独检查原始视频
python scripts/check_raw_videos.py \
    --annotation_file /path/to/annotations.json \
    --data_root /path/to/data

# 单独检查 Dataset 输出
python scripts/check_dataset_output.py \
    --annotation_file /path/to/annotations.json \
    --data_root /path/to/data
```

---

## 📊 常见问题诊断

### 问题 1: Dataset 输出几乎全是 0

**症状**:
```
⚠️ 最大值 < 0.01 - 数据几乎全是0！
```

**可能原因**:
- 视频读取失败
- 视频文件损坏
- 路径配置错误

**解决方法**:
```bash
# 检查原始视频
python scripts/check_raw_videos.py ...
```

### 问题 2: 重复归一化

**症状**:
```
Dataset输出: float [0, 1] （已归一化）
训练代码: batch.float() / 255.0  ← 重复归一化！
结果: 数据范围 [0, 0.004]
```

**解决方法**:
- 检查 `training/taehv_train.py`
- 如果 Dataset 已归一化，改为 `batch.float()`（删除 `/ 255.0`）

### 问题 3: uint8 但范围异常

**症状**:
```
Dtype: uint8
范围: [0, 10]  ← 异常低
```

**可能原因**:
- 暗场景视频
- 视频解码问题

**解决方法**:
- 检查视频内容
- 尝试不同的解码器（decord vs opencv）

---

## 🔧 维护说明

### 修改诊断配置

编辑 `diagnose_data_range.sh`:

```bash
# 修改默认路径
ANNOTATION_FILE="/your/path/annotations.json"
DATA_ROOT="/your/path/data"
NUM_SAMPLES=10  # 增加检查样本数
```

### 添加新的检查项

在 `training/utils/data_range_checker.py` 中:

```python
# 添加新的检查规则
if new_condition:
    warnings.append({
        "type": "new_check",
        "severity": "high",
        "message": "Your warning message"
    })
```

---

## 🔧 已知问题和修复

### check_model_data_range.py 修复

#### 修复 1: 模块导入错误（2025-10-14）

**问题**: ModuleNotFoundError: No module named 'training.utils'

**原因**: 脚本依赖的 `training.utils.data_range_checker` 模块不存在

**修复**: 
- ✅ 重写为独立脚本，不依赖外部工具模块
- ✅ 所有功能在脚本内部实现
- ✅ 保持原有功能完整

**详细说明**: 查看 [FIX_check_model_data_range.md](./FIX_check_model_data_range.md)

#### 修复 2: 预训练模型路径错误（2025-10-14）

**问题**: FileNotFoundError: [Errno 2] No such file or directory: 'taehv.pth'

**原因**: 
- TAEHV 类默认尝试加载 `taehv.pth`
- 实际预训练模型在 `checkpoints/taecvx.pth`

**修复**:
- ✅ 添加 `--pretrained_path` 参数（默认: `checkpoints/taecvx.pth`）
- ✅ 正确处理预训练模型和训练 checkpoint 的加载
- ✅ 改进错误处理和日志输出

**使用方法**:
```bash
python scripts/check_model_data_range.py \
    --model_path output/best_model/model.pth \
    --config training/configs/taehv_config_a800.py \
    --pretrained_path checkpoints/taecvx.pth  # 可选，默认就是这个路径
```

**详细说明**: 查看 [FIX_pretrained_model_path.md](./FIX_pretrained_model_path.md)

---

## 📚 相关文档

### 项目文档
- [数据范围异常诊断指南](../docs/数据范围异常诊断指南.md)
- [数据范围检查使用指南](../docs/数据范围检查使用指南.md)
- [数据范围检查功能设计](../docs/数据范围检查功能设计.md)
- [QUICK_START](../docs/QUICK_START.md)

### Scripts 修复文档
- [模块导入错误修复](./FIX_check_model_data_range.md) - 第一次修复
- [预训练模型路径修复](./FIX_pretrained_model_path.md) - 第二次修复
- [修复总结](./修复总结.md) - 第一次修复总结
- [工作完成报告](./工作完成报告.md) - 完整工作报告

---

## 📞 支持

如果遇到问题：

1. 查看 `docs/` 目录下的相关文档
2. 运行 `bash scripts/diagnose_data_range.sh` 获取诊断报告
3. 检查输出的 JSON 文件了解详细信息
4. 查看修复说明文档了解已知问题的解决方案

