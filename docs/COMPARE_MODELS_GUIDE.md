# 🔬 模型对比评估指南

## 📝 功能特性

`compare_models.py` 提供全面的模型对比功能：

✅ **并排可视化对比** - 直观展示重建质量差异
✅ **详细指标对比** - PSNR、SSIM、MSE、MAE等
✅ **改进百分比分析** - 量化训练效果
✅ **统计信息** - 均值、标准差、置信区间
✅ **综合对比大图** - 一张图展示所有关键信息
✅ **Markdown报告** - 详细的对比分析报告

## 🚀 快速开始

### 基本用法

```bash
cd /data/matrix-project/seraena/my_taehv_training

python compare_models.py \
    --model1 checkpoints/taecvx.pth \
    --model1_name "Untrained" \
    --model2 output/checkpoint-19000-merged/merged_model_final.pth \
    --model2_name "Trained (19k steps)" \
    --data_root /data/matrix-project/MiniDataset/5M_validation_set/data \
    --annotation_file /data/matrix-project/MiniDataset/5M_validation_set/stage1_annotations_500_validation_5.json \
    --num_samples 5
```

### 完整参数说明

```bash
python compare_models.py \
    # 必需参数
    --model1 <path>              # 第一个模型路径（例如：未训练模型）
    --model2 <path>              # 第二个模型路径（例如：训练后模型）
    --data_root <path>           # 验证集数据目录
    --annotation_file <path>     # 验证集标注文件
    
    # 可选参数
    --model1_name "Name1"        # 第一个模型的显示名称（默认：Model 1）
    --model2_name "Name2"        # 第二个模型的显示名称（默认：Model 2）
    --model3 <path>              # 第三个模型路径（可选）
    --model3_name "Name3"        # 第三个模型的显示名称
    
    --config <path>              # 配置文件路径（默认：taehv_config_16gpu_h100.py）
    --num_samples 10             # 评估样本数量（默认：5）
    --batch_size 2               # 批次大小（默认：1）
    --output_dir <path>          # 输出目录（默认：evaluation_comparison）
    --device cuda                # 设备（默认：cuda）
```

## 📊 输出内容

运行完成后，会在输出目录生成以下文件：

```
evaluation_comparison/
├── side_by_side_comparison.png      # 并排重建对比
├── metrics_comparison.png           # 指标对比柱状图
├── improvement_analysis.png         # 改进百分比分析
├── comprehensive_comparison.png     # 综合对比大图 ⭐
└── COMPARISON_REPORT.md             # 详细对比报告
```

### 📸 可视化说明

#### 1. `side_by_side_comparison.png`
- 第一行：原图 + 各模型重建结果
- 第二行：各模型的误差热图

#### 2. `metrics_comparison.png`
- PSNR、SSIM、MSE、MAE 的柱状图对比
- 包含误差棒（标准差）

#### 3. `improvement_analysis.png`
- 相对于基线模型的改进百分比
- 正值表示改进，负值表示退化

#### 4. `comprehensive_comparison.png` ⭐ **推荐查看**
综合大图包含：
- 重建结果对比
- 多个指标的柱状图
- 改进百分比表格

#### 5. `COMPARISON_REPORT.md`
详细的文字报告，包含：
- 模型列表
- 指标对比表
- 改进分析
- 总结建议

## 💡 使用场景

### 场景1：训练前后对比

比较未训练模型和训练后模型的性能：

```bash
python compare_models.py \
    --model1 checkpoints/taecvx.pth \
    --model1_name "Pretrained (Baseline)" \
    --model2 output/checkpoint-19000-merged/merged_model_final.pth \
    --model2_name "Fine-tuned (19k)" \
    --data_root /data/matrix-project/MiniDataset/5M_validation_set/data \
    --annotation_file /data/matrix-project/MiniDataset/5M_validation_set/stage1_annotations_500_validation_5.json \
    --num_samples 10
```

### 场景2：不同checkpoint对比

比较不同训练步数的模型：

```bash
# 首先合并其他checkpoint
cd /data/matrix-project/seraena/my_taehv_training/output
python merge_distributed_checkpoint.py --checkpoint checkpoint-10000
python merge_distributed_checkpoint.py --checkpoint checkpoint-15000

# 然后对比
cd ..
python compare_models.py \
    --model1 output/checkpoint-10000-merged/merged_model_final.pth \
    --model1_name "Checkpoint 10k" \
    --model2 output/checkpoint-15000-merged/merged_model_final.pth \
    --model2_name "Checkpoint 15k" \
    --model3 output/checkpoint-19000-merged/merged_model_final.pth \
    --model3_name "Checkpoint 19k" \
    --data_root /data/matrix-project/MiniDataset/5M_validation_set/data \
    --annotation_file /data/matrix-project/MiniDataset/5M_validation_set/stage1_annotations_500_validation_5.json
```

### 场景3：不同数据集验证

在多个验证集上对比：

```bash
# 验证集1
python compare_models.py \
    --model1 checkpoints/taecvx.pth --model1_name "Untrained" \
    --model2 output/checkpoint-19000-merged/merged_model_final.pth --model2_name "Trained" \
    --data_root /path/to/val_set_1/data \
    --annotation_file /path/to/val_set_1/annotations.json \
    --output_dir comparison_val1

# 验证集2
python compare_models.py \
    --model1 checkpoints/taecvx.pth --model1_name "Untrained" \
    --model2 output/checkpoint-19000-merged/merged_model_final.pth --model2_name "Trained" \
    --data_root /path/to/val_set_2/data \
    --annotation_file /path/to/val_set_2/annotations.json \
    --output_dir comparison_val2
```

## 📈 解读结果

### PSNR (Peak Signal-to-Noise Ratio)
- **范围**: 通常 20-50 dB
- **越高越好**
- **改进**: +1 dB 就是显著提升

### SSIM (Structural Similarity Index)
- **范围**: 0-1
- **越高越好**（1.0 表示完全相同）
- **改进**: +0.01 就是可见的提升

### MSE (Mean Squared Error)
- **越低越好**
- 与 PSNR 呈反比关系

### MAE (Mean Absolute Error)
- **越低越好**
- 对异常值不如 MSE 敏感

### 改进百分比解读

```
PSNR: +12.8%  ✅ 显著改进
SSIM: +8.2%   ✅ 明显提升
MSE:  -46.7%  ✅ 大幅降低（好事！）
```

## 🎯 你的具体任务

根据你的需求，运行：

```bash
cd /data/matrix-project/seraena/my_taehv_training

python compare_models.py \
    --model1 checkpoints/taecvx.pth \
    --model1_name "Untrained" \
    --model2 output/checkpoint-19000-merged/merged_model_final.pth \
    --model2_name "Trained (19k steps)" \
    --data_root /data/matrix-project/MiniDataset/5M_validation_set/data \
    --annotation_file /data/matrix-project/MiniDataset/5M_validation_set/stage1_annotations_500_validation_5.json \
    --num_samples 5 \
    --output_dir evaluate/comparison
```

### 预期输出

```
🔬 Model Comparison Framework
================================================================================

📦 Loading model: Untrained
   Path: checkpoints/taecvx.pth
   ✅ Loaded (11.32M parameters)

📦 Loading model: Trained (19k steps)
   Path: output/checkpoint-19000-merged/merged_model_final.pth
   ✅ Loaded (11.32M parameters)

📂 Loading dataset...
✅ Loaded 5 samples

================================================================================
🔍 Evaluating 2 models on 5 samples...
================================================================================

Processing batches: 100%|████████████████████| 5/5 [00:XX<00:00]

✅ Evaluation complete!

================================================================================
🎨 Generating comparison visualizations...
================================================================================
   📊 Creating side-by-side comparison...
   📊 Creating metrics comparison chart...
   📊 Creating improvement analysis...
   📊 Creating comprehensive comparison...
✅ Visualizations saved to evaluate/comparison

   📄 Generating comparison report...
✅ Report saved to: evaluate/comparison/COMPARISON_REPORT.md

✅ All results saved to: evaluate/comparison

================================================================================
🎉 Comparison completed successfully!
================================================================================
```

## 🔍 查看结果

```bash
# 查看生成的图片
ls -lh evaluate/comparison/*.png

# 查看报告
cat evaluate/comparison/COMPARISON_REPORT.md

# 或在IDE中打开图片查看
```

## ⚙️ 高级选项

### 调整评估样本数

```bash
# 快速测试（1-2个样本）
--num_samples 2

# 标准评估（5-10个样本）
--num_samples 10

# 完整评估（所有样本）
--num_samples 100
```

### 批次大小

```bash
# 内存有限
--batch_size 1

# 加速评估（如果内存足够）
--batch_size 4
```

## 🆘 故障排查

### 问题1：CUDA out of memory

**解决**：
```bash
--batch_size 1
--num_samples 5
```

### 问题2：模型加载失败

**检查**：
```bash
python -c "
import torch
m = torch.load('path/to/model.pth', map_location='cpu')
print('Keys:', list(m.keys())[:5] if isinstance(m, dict) else 'Direct state_dict')
print('Params:', len(m) if isinstance(m, dict) else 'N/A')
"
```

### 问题3：数据集路径错误

**验证**：
```bash
ls /data/matrix-project/MiniDataset/5M_validation_set/data
cat /data/matrix-project/MiniDataset/5M_validation_set/stage1_annotations_500_validation_5.json | head
```

## 💻 与 evaluate.py 的区别

| 特性 | evaluate.py | compare_models.py |
|------|-------------|-------------------|
| 目的 | 单模型深度评估 | 多模型对比 |
| 指标 | 非常详细（7类指标） | 核心指标（4个） |
| 可视化 | 单模型样本 | 并排对比 |
| 报告 | 深度分析报告 | 对比分析报告 |
| 速度 | 较慢（详细计算） | 较快（核心指标） |
| 用途 | 模型调试、论文 | 快速对比、选择模型 |

## 🎓 建议工作流

1. **快速对比** - 使用 `compare_models.py` 快速查看训练效果
2. **深度评估** - 对表现好的模型使用 `evaluate.py` 进行详细分析
3. **论文图表** - 使用两者的输出制作论文图表

---

**准备好了吗？** 现在就运行对比评估，看看你的模型训练效果如何！🚀

