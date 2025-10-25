# Notebooks 目录说明

## ⚠️ 注意事项

本目录下的 Notebook 是**参考示例**，**不是**项目的主要训练代码。

---

## 📓 TAEHV_Training_Example.ipynb

### 状态：参考用（Reference Only）

这是一个 TAEHV 模型的**独立训练示例**，展示了如何从零开始训练一个视频编解码器。

### 特点

- 📚 教学性质：展示了完整的训练流程
- 🔧 自包含：包含数据加载、模型定义、训练循环
- 🎨 可视化：包含图像和视频的可视化代码

### 与项目主代码的区别

| 特性 | Notebook 示例 | 项目主代码 (`training/taehv_train.py`) |
|------|---------------|----------------------------------------|
| 训练框架 | 原生 PyTorch | Hugging Face Accelerate |
| 数据加载 | 自定义 Dataset | MiniDataset |
| 分布式训练 | 不支持 | 支持多 GPU/多节点 |
| Seraena 集成 | 手动实现 | 通过配置启用 (`use_seraena=True`) |
| 配置管理 | 硬编码 | 配置文件 (`configs/*.py`) |
| 检查点保存 | 手动 | 自动（with resume） |
| 日志系统 | 简单 print | TensorBoard + 文件日志 |

### 使用场景

✅ **适合用于**:
- 理解 TAEHV 模型的基本原理
- 学习视频编解码器的训练流程
- 快速原型验证
- Jupyter 环境下的实验

❌ **不适合用于**:
- 生产环境训练
- 大规模数据集训练
- 多 GPU 分布式训练
- 需要完整训练管道的场景

### 如何使用项目主训练代码

如果你想进行实际的训练，请使用：

```bash
# 单 GPU 训练
bash train_taehv_a800.sh 1 custom training/configs/taehv_config_a800.py

# 多 GPU 训练  
bash train_taehv_h100.sh 8 custom training/configs/taehv_config_h100.py
```

详见：[训练指南](../../docs/训练指南.md)

### 如何运行此 Notebook

如果你想查看此 Notebook 作为参考：

```bash
cd /data/matrix-project/seraena/my_taehv_training/scripts/notebooks
jupyter notebook TAEHV_Training_Example.ipynb
```

**注意**：
- 需要修改数据路径配置
- 需要单独下载 Wan VAE 模型
- 训练效果可能与主项目不同

---

## 📚 相关文档

- [主训练脚本](../../training/taehv_train.py)
- [配置文件说明](../../docs/配置修改说明.md)
- [训练指南](../../docs/训练指南.md)
- [Scripts 目录 README](../README.md)

