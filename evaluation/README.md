# 🎯 VAE 模型评估工具

完整的 TAEHV 模型评估工具集，支持定量评估、训练日志分析和可视化。

---

## 🚀 快速开始

### 一键评估（推荐）

```bash
cd /data/matrix-project/seraena/my_taehv_training/evaluation

# 使用自定义数据集
python quick_evaluate.py \
    --data_root /data/matrix-project/MiniDataset/data \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --num_samples 100
```

### 分步评估

**步骤 1: 评估模型**
```bash
python evaluate_vae.py \
    --model_path ../output/2025-10-01_19-59-50/final_model.pth \
    --config ../training/configs/taehv_config_h100.py \
    --num_samples 100 \
    --batch_size 4 \
    --data_root /data/matrix-project/MiniDataset/data \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json
```

**步骤 2: 分析训练日志（可选）**
```bash
export LD_LIBRARY_PATH=/data1/anaconda3/envs/tiny-vae/lib:$LD_LIBRARY_PATH

python analyze_training_logs.py \
    --log_dir ../logs/taehv_h100_production \
    --output_dir ./evaluation_results
```

---

## 📊 评估指标说明

| 指标 | 说明 | 优秀标准 | 评估维度 |
|------|------|----------|----------|
| **PSNR** | 峰值信噪比 | >30 dB | 像素级重建质量 |
| **SSIM** | 结构相似性 | >0.90 | 结构信息保留 |
| **LPIPS** | 感知相似度 | <0.10 | 人眼感知质量 |
| **MSE** | 均方误差 | 越低越好 | 像素级误差 |
| **MAE** | 平均绝对误差 | 越低越好 | 像素级误差 |

### 综合评分标准
- **85-100分**: 🌟 Excellent（优秀）- 可以部署
- **70-84分**: ✅ Good（良好）- 质量良好
- **55-69分**: ⚠️ Fair（一般）- 需要改进
- **<55分**: ❌ Poor（较差）- 需要重新训练

---

## 📁 输出结果

所有结果保存在 `evaluation/evaluation_results/` 目录：

```
evaluation_results/
├── evaluation_results.json       # 详细评估数据
├── metrics_distribution.png      # 指标分布图
├── sample_1.png                  # 重建样本对比图
├── sample_2.png                  # 更多样本...
├── training_losses.png           # 训练损失曲线（可选）
├── training_metrics.png          # 训练指标曲线（可选）
└── training_analysis.json        # 训练分析报告（可选）
```

---

## 🛠️ 工具说明

### 1. `evaluate_vae.py` - 模型定量评估

**功能**:
- 加载训练好的 TAEHV 模型
- 在验证集上计算 PSNR、SSIM、LPIPS 等指标
- 生成重建样本可视化
- 输出详细的评估报告

**常用参数**:
```bash
--model_path PATH       # 模型权重文件路径
--config PATH           # 训练配置文件
--num_samples N         # 评估样本数（建议 50-200）
--batch_size N          # 批次大小（根据显存调整，建议 1-4）
--data_root PATH        # 数据集根目录
--annotation_file PATH  # 标注文件路径
```

**示例**:
```bash
# 快速测试（5个样本）
python evaluate_vae.py \
    --model_path ../output/2025-10-01_19-59-50/final_model.pth \
    --config ../training/configs/taehv_config_h100.py \
    --num_samples 5 \
    --batch_size 1 \
    --data_root /data/matrix-project/MiniDataset/data \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json

# 完整评估（100个样本）
python evaluate_vae.py \
    --model_path ../output/2025-10-01_19-59-50/final_model.pth \
    --config ../training/configs/taehv_config_h100.py \
    --num_samples 100 \
    --batch_size 4 \
    --data_root /data/matrix-project/MiniDataset/data \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json
```

---

### 2. `analyze_training_logs.py` - 训练日志分析

**功能**:
- 读取 TensorBoard 事件文件
- 绘制训练损失和指标曲线
- 分析收敛性和趋势
- 识别最佳 checkpoint
- 提供训练优化建议

**参数**:
```bash
--log_dir PATH         # TensorBoard 日志目录
--output_dir PATH      # 输出目录
```

**示例**:
```bash
# 需要先设置环境变量（解决 GLIBCXX 问题）
export LD_LIBRARY_PATH=/data1/anaconda3/envs/tiny-vae/lib:$LD_LIBRARY_PATH

python analyze_training_logs.py \
    --log_dir ../logs/taehv_h100_production \
    --output_dir ./evaluation_results
```

---

### 3. `quick_evaluate.py` - 一键完整评估

**功能**:
- 自动运行训练日志分析
- 自动运行模型评估
- 生成综合评估报告

**参数**:
```bash
--model_path PATH           # 模型权重文件
--log_dir PATH              # 训练日志目录
--config PATH               # 配置文件
--num_samples N             # 评估样本数
--output_dir PATH           # 输出目录
--data_root PATH            # 数据集根目录（覆盖配置）
--annotation_file PATH      # 标注文件（覆盖配置）
--skip_logs                 # 跳过训练日志分析
```

**示例**:
```bash
# 完整评估
python quick_evaluate.py \
    --data_root /data/matrix-project/MiniDataset/data \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \
    --num_samples 100

# 跳过日志分析
python quick_evaluate.py \
    --skip_logs \
    --data_root /data/matrix-project/MiniDataset/data \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json
```

---

## 📖 使用场景

### 场景 1: 训练完成后的标准评估

```bash
# 1. 完整评估（推荐）
python quick_evaluate.py \
    --data_root /your/data/path \
    --annotation_file /your/annotation.json \
    --num_samples 100

# 2. 查看结果
ls -lh evaluation_results/
cat evaluation_results/evaluation_results.json | python -m json.tool
```

### 场景 2: 快速验证模型是否正常

```bash
# 使用少量样本快速测试
python evaluate_vae.py \
    --model_path ../output/xxx/final_model.pth \
    --config ../training/configs/taehv_config_h100.py \
    --num_samples 5 \
    --batch_size 1 \
    --data_root /your/data/path \
    --annotation_file /your/annotation.json
```

### 场景 3: 只分析训练过程

```bash
export LD_LIBRARY_PATH=/data1/anaconda3/envs/tiny-vae/lib:$LD_LIBRARY_PATH

python analyze_training_logs.py \
    --log_dir ../logs/taehv_h100_production \
    --output_dir ./evaluation_results
```

### 场景 4: 比较不同配置的数据集

```bash
# 评估数据集 A
python evaluate_vae.py \
    --model_path ../output/xxx/final_model.pth \
    --config ../training/configs/taehv_config_h100.py \
    --data_root /path/to/dataset_A \
    --annotation_file /path/to/dataset_A/annotations.json \
    --num_samples 100

# 评估数据集 B
python evaluate_vae.py \
    --model_path ../output/xxx/final_model.pth \
    --config ../training/configs/taehv_config_h100.py \
    --data_root /path/to/dataset_B \
    --annotation_file /path/to/dataset_B/annotations.json \
    --num_samples 100
```

---

## 🔧 技术细节

### TAEHV 模型特性

**帧数裁剪**:
- TAEHV 的 `decode_video()` 会自动裁剪输出帧数
- 默认配置: 输入 12 帧 → 输出 9 帧（裁剪前 3 帧）
- 评估工具会自动对齐帧数，无需手动处理

**数据范围**:
- TAEHV 要求输入/输出范围: **[0, 1]**
- 评估工具会自动处理数据范围转换
- 指标计算统一使用 [0, 1] 范围

**模型接口**:
- 编码: `model.encode_video(videos, parallel=True)`
- 解码: `model.decode_video(latents, parallel=True)`

---

## ⚙️ 环境要求

### 必需依赖
```bash
pip install torch torchvision
pip install lpips scikit-image
pip install opencv-python
```

### 可选依赖（用于训练日志分析）
```bash
pip install matplotlib seaborn
pip install pandas scipy
pip install tensorboard tensorflow
```

### 环境变量（解决 GLIBCXX 问题）
```bash
# 如果遇到 matplotlib 报错，运行：
export LD_LIBRARY_PATH=/data1/anaconda3/envs/tiny-vae/lib:$LD_LIBRARY_PATH

# 永久设置（添加到 ~/.bashrc）
echo 'export LD_LIBRARY_PATH=/data1/anaconda3/envs/tiny-vae/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## 📊 查看结果

### 查看评估指标
```bash
cd evaluation_results

# 使用 Python 格式化输出
python -c "
import json
with open('evaluation_results.json', 'r') as f:
    data = json.load(f)
    print(f'PSNR:  {data[\"psnr\"][\"mean\"]:.2f} ± {data[\"psnr\"][\"std\"]:.2f}')
    print(f'SSIM:  {data[\"ssim\"][\"mean\"]:.4f} ± {data[\"ssim\"][\"std\"]:.4f}')
    print(f'LPIPS: {data[\"lpips\"][\"mean\"]:.4f} ± {data[\"lpips\"][\"std\"]:.4f}')
"
```

### 查看可视化结果
```bash
# 列出所有生成的图像
ls -lh evaluation_results/*.png

# 在本地浏览器查看（使用 SCP 或文件管理器）
# - metrics_distribution.png: 指标分布图
# - sample_*.png: 重建样本对比
# - training_losses.png: 训练损失曲线
```

### 启动 TensorBoard
```bash
cd /data/matrix-project/seraena/my_taehv_training

tensorboard --logdir logs/taehv_h100_production --port 6006 --bind_all
```

---

## ❓ 常见问题

### Q1: 数据集路径不存在
```
FileNotFoundError: [Errno 2] No such file or directory
```
**解决**: 使用 `--data_root` 和 `--annotation_file` 参数指定正确路径。

### Q2: 显存不足 (OOM)
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**解决**: 降低 `--batch_size` 到 1 或 2。

### Q3: GLIBCXX 版本问题
```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```
**解决**: 
```bash
export LD_LIBRARY_PATH=/data1/anaconda3/envs/tiny-vae/lib:$LD_LIBRARY_PATH
```

### Q4: 模型加载失败
```
RuntimeError: Error(s) in loading state_dict
```
**解决**: 检查模型路径和配置文件中的 `patch_size`、`latent_channels` 是否正确。

### Q5: 评估很慢
**优化建议**:
- 减少评估样本数: `--num_samples 50`
- 增加批次大小: `--batch_size 8` (根据显存)
- 使用更少的 DataLoader workers

---

## 📞 获取帮助

```bash
# 查看命令行帮助
python quick_evaluate.py --help
python evaluate_vae.py --help
python analyze_training_logs.py --help

# 查看故障排除指南
cat TROUBLESHOOTING.md
```

---

## 📝 更新日志

### 最新版本
- ✅ 修复 TAEHV 接口调用（encode_video/decode_video）
- ✅ 修复帧数对齐问题
- ✅ 修复数据范围转换
- ✅ 修复 JSON 序列化错误
- ✅ 支持命令行参数覆盖数据集路径
- ✅ 改进错误提示和用户体验

---

**祝评估顺利！** 🚀

如有问题，请查看 `TROUBLESHOOTING.md` 或联系开发者。
