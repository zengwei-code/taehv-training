# 🔧 故障排除指南

本文档记录了评估工具的所有已知问题和解决方案。

---

## 📋 目录

1. [环境依赖问题](#环境依赖问题)
2. [模型接口问题](#模型接口问题)
3. [数据集问题](#数据集问题)
4. [帧数对齐问题](#帧数对齐问题)
5. [性能优化](#性能优化)

---

## 环境依赖问题

### 问题 1: GLIBCXX 版本冲突

**错误信息**:
```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```

**原因**:
系统的 `libstdc++.so.6` 库版本太旧，不支持 matplotlib 需要的 `GLIBCXX_3.4.29`。

**解决方案**:

**方法 1: 使用 conda 环境的库（推荐）**
```bash
export LD_LIBRARY_PATH=/data1/anaconda3/envs/tiny-vae/lib:$LD_LIBRARY_PATH
```

**方法 2: 永久设置**
```bash
echo 'export LD_LIBRARY_PATH=/data1/anaconda3/envs/tiny-vae/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**方法 3: 在脚本中设置**
在 `quick_evaluate.py` 或 `analyze_training_logs.py` 开头添加：
```python
import os
os.environ['LD_LIBRARY_PATH'] = '/data1/anaconda3/envs/tiny-vae/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
```

---

### 问题 2: 缺少 opencv-python

**错误信息**:
```
ModuleNotFoundError: No module named 'cv2'
```

**解决方案**:
```bash
conda activate tiny-vae
pip install opencv-python
```

---

### 问题 3: 缺少其他依赖

**解决方案**:
```bash
conda activate tiny-vae

# 核心依赖
pip install torch torchvision
pip install lpips scikit-image
pip install opencv-python

# 可选依赖（用于日志分析）
pip install matplotlib seaborn
pip install pandas scipy
pip install tensorboard tensorflow
```

---

## 模型接口问题

### 问题 4: 'TAEHV' object has no attribute 'encode'

**错误信息**:
```python
AttributeError: 'TAEHV' object has no attribute 'encode'. Did you mean: 'encoder'?
```

**原因**:
TAEHV 模型使用 `encode_video()` 和 `decode_video()` 方法，而不是 `encode()` 和 `decode()`。

**解决方案**:
已在代码中修复。如果你手动调用模型，请使用：
```python
# ❌ 错误
latents = model.encode(videos)
reconstructions = model.decode(latents)

# ✅ 正确
latents = model.encode_video(videos, parallel=True, show_progress_bar=False)
reconstructions = model.decode_video(latents, parallel=True, show_progress_bar=False)
```

---

### 问题 5: 数据范围不匹配

**原因**:
TAEHV 模型要求输入/输出数据范围为 **[0, 1]**，而不是 [-1, 1]。

**解决方案**:
已在代码中自动处理。如果手动使用模型：
```python
# 确保数据在 [0, 1] 范围
if videos.min() < 0:
    videos = (videos + 1) / 2  # [-1, 1] -> [0, 1]

# 使用模型
latents = model.encode_video(videos, parallel=True, show_progress_bar=False)
reconstructions = model.decode_video(latents, parallel=True, show_progress_bar=False)

# 输出已经在 [0, 1] 范围，可以直接使用
```

---

## 数据集问题

### 问题 6: 数据集路径不存在

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/mnt/project_modelware/...'
```

**原因**:
配置文件中的数据集路径是训练环境的路径，在评估环境中不存在。

**解决方案**:
使用命令行参数覆盖配置文件中的路径：
```bash
python evaluate_vae.py \
    --model_path ../output/xxx/final_model.pth \
    --config ../training/configs/taehv_config_h100.py \
    --data_root /data/matrix-project/MiniDataset/data \
    --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json
```

---

### 问题 7: 数据集类名不匹配

**错误信息**:
```
ImportError: cannot import name 'VideoDataset' from 'dataset'
```

**原因**:
代码中导入了不存在的 `VideoDataset` 类，实际应该是 `MiniDataset`。

**解决方案**:
已在代码中修复：
```python
# ❌ 错误
from dataset import VideoDataset

# ✅ 正确
from dataset import MiniDataset
```

---

### 问题 8: 数据加载格式错误

**错误信息**:
```python
TypeError: new(): invalid data type 'str'
```

**原因**:
`MiniDataset` 返回的是 tensor，而不是字典。

**解决方案**:
已在代码中修复，自动检测数据格式：
```python
# 兼容两种格式
if isinstance(batch, dict):
    videos = batch['video'].to(self.device)
else:
    videos = batch.to(self.device)
```

---

## 帧数对齐问题

### 问题 9: 帧数越界错误

**错误信息**:
```python
IndexError: index 9 is out of bounds for axis 1 with size 9
```

**原因**:
TAEHV 的 `decode_video()` 会自动裁剪输出帧数：
- 输入: 12 帧
- `frames_to_trim`: 3
- 输出: 9 帧（裁剪掉前 3 帧）

**解决方案**:
已在代码中修复，自动对齐帧数：
```python
# 对齐帧数
frames_to_trim = self.model.frames_to_trim
if reconstructions.shape[1] < videos.shape[1]:
    videos_trimmed = videos[:, frames_to_trim:frames_to_trim + reconstructions.shape[1]]
else:
    videos_trimmed = videos

# 使用对齐后的视频计算指标
batch_metrics = self._compute_metrics(videos_trimmed, reconstructions)
```

**帧数对应关系**:

| decoder_time_upscale | frames_to_trim | 输入 → 输出 |
|---------------------|----------------|-------------|
| (False, False) | 0 | 12 → 12 |
| (True, False) | 1 | 12 → 11 |
| (True, True) | 3 | 12 → 9 |
| (True, True, True) | 7 | 16 → 9 |

---

## 性能优化

### 问题 10: 评估速度慢

**优化建议**:

**1. 减少评估样本数**
```bash
python evaluate_vae.py --num_samples 50  # 默认 100
```

**2. 增加批次大小（如果显存充足）**
```bash
python evaluate_vae.py --batch_size 8  # 默认 4
```

**3. 减少 DataLoader workers**
修改代码中的 `num_workers=2`（默认 4）

**4. 使用更少的指标**
如果只需要快速验证，可以注释掉 LPIPS 计算（最耗时）。

---

### 问题 11: 显存不足 (OOM)

**错误信息**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**:

**1. 降低批次大小**
```bash
python evaluate_vae.py --batch_size 1
```

**2. 减少评估样本数**
```bash
python evaluate_vae.py --num_samples 50
```

**3. 清理 GPU 缓存**
```python
import torch
torch.cuda.empty_cache()
```

**4. 限制 CUDA 内存分配**
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## JSON 序列化问题

### 问题 12: numpy 类型无法序列化

**错误信息**:
```python
TypeError: Object of type ndarray is not JSON serializable
TypeError: Object of type bool_ is not JSON serializable
```

**原因**:
numpy 的 `ndarray` 和 `bool_` 类型无法直接被 JSON 序列化。

**解决方案**:
已在代码中修复，强制转换为 Python 原生类型：
```python
return {
    'moving_average': moving_avg.values.tolist(),  # ndarray → list
    'is_converged': bool(is_converged),  # numpy.bool_ → bool
    'slope': float(slope),  # numpy.float64 → float
    # ...
}
```

---

## 模型加载问题

### 问题 13: 权重加载失败

**错误信息**:
```python
RuntimeError: Error(s) in loading state_dict
```

**可能原因**:
1. 模型配置不匹配（`patch_size`、`latent_channels` 等）
2. checkpoint 格式问题
3. 权重文件损坏

**解决方案**:

**1. 检查配置文件**
确保配置文件中的参数与训练时一致：
```python
args.patch_size = 1  # 必须与训练时一致
args.latent_channels = 16  # 必须与训练时一致
```

**2. 使用 strict=False**
已在代码中使用 `strict=False` 允许部分加载：
```python
model.load_state_dict(state_dict, strict=False)
```

**3. 检查 checkpoint 格式**
代码已自动处理多种格式：
```python
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
```

---

## 其他问题

### 问题 14: TensorBoard 日志文件不存在

**错误信息**:
```
❌ No event files found in ../logs/taehv_h100_production
```

**解决方案**:
- 跳过日志分析：`python quick_evaluate.py --skip_logs`
- 或指定正确的日志目录：`--log_dir /correct/path/to/logs`

---

### 问题 15: matplotlib 后端问题

**错误信息**:
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**解决方案**:
这不是错误，只是警告。如果需要交互式绘图：
```python
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'
```

---

## 完整修复清单

所有已修复的问题：

- [x] 模块导入错误 (TAECVX → TAEHV)
- [x] 数据集类名错误 (VideoDataset → MiniDataset)
- [x] 模型接口错误 (encode → encode_video)
- [x] 数据范围不匹配 ([-1,1] vs [0,1])
- [x] JSON 序列化错误 (numpy types)
- [x] 数据加载格式 (tensor vs dict)
- [x] 帧数对齐问题
- [x] GLIBCXX 版本冲突
- [x] 输出路径配置

---

## 调试技巧

### 1. 使用详细输出
```bash
python -u evaluate_vae.py [参数] 2>&1 | tee evaluation_log.txt
```

### 2. 查看系统资源
```bash
# GPU 使用情况
nvidia-smi

# CPU/内存使用情况
htop
```

### 3. Python 交互式调试
```python
# 在代码中添加断点
import pdb; pdb.set_trace()

# 或使用 IPython
from IPython import embed; embed()
```

### 4. 检查数据形状
```python
print(f"Videos shape: {videos.shape}")
print(f"Reconstructions shape: {reconstructions.shape}")
print(f"frames_to_trim: {self.model.frames_to_trim}")
```

---

## 获取帮助

如果问题仍未解决：

1. **查看完整错误堆栈**
   ```bash
   python -u evaluate_vae.py [参数] 2>&1 | tee error.log
   ```

2. **检查依赖版本**
   ```bash
   pip list | grep -E "torch|lpips|opencv"
   ```

3. **验证数据集**
   ```python
   import json
   with open('annotation.json', 'r') as f:
       data = json.load(f)
       print(f"Total videos: {len(data['list'])}")
       print(f"First video: {data['list'][0]}")
   ```

4. **联系开发者**
   提供以下信息：
   - 完整错误信息
   - 运行的命令
   - Python 和依赖库版本
   - GPU 型号和驱动版本

---

**持续更新中...** 

遇到新问题？欢迎提交 Issue 或 Pull Request！

