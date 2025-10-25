# 多机16卡训练配置指南

## 🎯 问题诊断

你的环境是：
- **2台机器** (`worker-0` 和 `worker-1`)
- **每台机器 8个 H100 GPU**
- **总共 16个 GPU**

当前配置错误地将其视为**单机16卡**，导致每台机器都尝试访问 GPU 0-15，但实际每台机器只有 GPU 0-7！

---

## ❌ 错误表现

```
RuntimeError: CUDA error: invalid device ordinal
```

**原因**：
- 启动脚本设置：`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`
- DeepSpeed 配置：`num_machines: 1`, `num_processes: 16`
- 结果：每台机器都试图启动16个进程，访问 GPU 0-15
- 实际：每台机器只有 GPU 0-7 ❌

---

## ✅ 解决方案

### 方案1: 使用单机8卡配置（最简单，推荐）

在多机环境下，**每台机器单独启动8卡训练**：

```bash
# 在 worker-0 上执行
export MASTER_ADDR=jo-dbxfe2k2l635222i-worker-0
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=2
./train_taehv_h100.sh 8 h100

# 在 worker-1 上执行
export MASTER_ADDR=jo-dbxfe2k2l635222i-worker-0
export MASTER_PORT=29500
export RANK=1
export WORLD_SIZE=2
./train_taehv_h100.sh 8 h100
```

**优点**：
- 简单，不需要修改配置
- 使用成熟的8卡配置
- 两台机器协同工作，总共16卡

**缺点**：
- 需要在每台机器上分别启动
- 需要手动设置环境变量

---

### 方案2: 使用多机启动脚本（推荐生产环境）

我已经修改了 `accelerate_configs/deepspeed_16gpu.yaml`，将 `num_machines` 改为 2。

现在需要修改启动方式：

#### 2.1 修改启动脚本中的GPU分配

编辑 `train_taehv_h100.sh`，找到第193-202行：

```bash
# CUDA优化设置 - H100/A800专用
GPU_IDS=""
for ((i=0; i<GPU_COUNT; i++)); do
    if [ $i -eq 0 ]; then
        GPU_IDS="$i"
    else
        GPU_IDS="$GPU_IDS,$i"
    fi
done
```

改为：

```bash
# CUDA优化设置 - H100/A800专用
# 多机环境下，每台机器使用8个GPU
if [ "$GPU_COUNT" -eq 16 ] && [ ! -z "$WORLD_SIZE" ] && [ "$WORLD_SIZE" -gt 1 ]; then
    # 多机16卡：每台机器8个GPU
    GPU_IDS="0,1,2,3,4,5,6,7"
    GPU_COUNT_PER_NODE=8
else
    # 单机：使用所有GPU
    GPU_IDS=""
    for ((i=0; i<GPU_COUNT; i++)); do
        if [ $i -eq 0 ]; then
            GPU_IDS="$i"
        else
            GPU_IDS="$GPU_IDS,$i"
        fi
    done
    GPU_COUNT_PER_NODE=$GPU_COUNT
fi
```

#### 2.2 修改 num_processes 计算

找到第321行：

```bash
--num_processes $((WORLD_SIZE * GPU_COUNT)) \
```

改为：

```bash
--num_processes $((WORLD_SIZE * GPU_COUNT_PER_NODE)) \
```

#### 2.3 启动训练

```bash
# 在 worker-0 上
export MASTER_ADDR=jo-dbxfe2k2l635222i-worker-0
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=2
./train_taehv_h100.sh 16 h100

# 在 worker-1 上
export MASTER_ADDR=jo-dbxfe2k2l635222i-worker-0
export MASTER_PORT=29500
export RANK=1
export WORLD_SIZE=2
./train_taehv_h100.sh 16 h100
```

---

### 方案3: 使用集群调度系统（如果有SLURM）

如果你的集群使用 SLURM 或其他调度系统，可以创建一个作业脚本：

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

srun ./train_taehv_h100.sh 8 h100
```

---

## 🔍 验证配置

### 检查每台机器的GPU数量

```bash
# 在每台机器上执行
nvidia-smi --list-gpus | wc -l
# 应该输出: 8
```

### 检查网络连通性

```bash
# 在 worker-1 上测试到 worker-0 的连接
ping jo-dbxfe2k2l635222i-worker-0

# 测试端口
nc -zv jo-dbxfe2k2l635222i-worker-0 29500
```

---

## 📊 多机 vs 单机对比

| 配置 | 机器数 | 每台GPU | 总GPU | 配置复杂度 | 通信开销 |
|------|--------|---------|-------|------------|----------|
| 单机8卡 | 1 | 8 | 8 | 低 | 低 |
| 单机16卡 | 1 | 16 | 16 | 中 | 低 |
| 多机16卡 (2×8) | 2 | 8 | 16 | 高 | 中 |

---

## ⚠️ 多机训练注意事项

### 1. 网络延迟
- 机器间通信延迟高于单机
- 需要高速网络（InfiniBand 推荐）
- 以太网可用但性能受限

### 2. 时钟同步
```bash
# 检查时钟同步
date
# 确保所有机器时间一致
```

### 3. 共享存储
- 数据集路径在所有机器上必须一致
- 或使用共享文件系统（NFS、Lustre等）

### 4. 环境一致性
- 所有机器的Python环境必须一致
- 所有机器的代码版本必须一致

---

## 🎯 推荐配置（最简单）

**使用方案1**：每台机器单独启动8卡训练

### Worker-0 启动脚本 (`start_worker0.sh`)

```bash
#!/bin/bash
export MASTER_ADDR=jo-dbxfe2k2l635222i-worker-0
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

./train_taehv_h100.sh 8 h100
```

### Worker-1 启动脚本 (`start_worker1.sh`)

```bash
#!/bin/bash
export MASTER_ADDR=jo-dbxfe2k2l635222i-worker-0
export MASTER_PORT=29500
export RANK=1
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

./train_taehv_h100.sh 8 h100
```

### 启动顺序

```bash
# 1. 在 worker-0 上先启动
bash start_worker0.sh

# 2. 在 worker-1 上启动（30秒内）
bash start_worker1.sh
```

---

## 📝 DeepSpeed 配置已修改

`accelerate_configs/deepspeed_16gpu.yaml` 已更新：

```yaml
num_machines: 2              # ✅ 2台机器
num_processes: 16            # ✅ 总共16个进程 (2×8)
```

---

## 🔧 调试命令

### 查看当前进程分布

```bash
# 在 worker-0 上
ps aux | grep taehv_train | wc -l
# 应该看到 8 个进程

# 在 worker-1 上
ps aux | grep taehv_train | wc -l
# 应该看到 8 个进程
```

### 查看 GPU 使用

```bash
# 在每台机器上
watch -n 1 nvidia-smi
# 应该看到 GPU 0-7 都在使用
```

### 查看 NCCL 通信

```bash
# 启用 NCCL 调试
export NCCL_DEBUG=INFO
```

---

## ✅ 成功标志

训练成功启动后，你应该看到：

```
[INFO] Initialized process group with 16 processes
[INFO] Rank 0-7 on worker-0
[INFO] Rank 8-15 on worker-1
```

---

## 💡 总结

**最简单的解决方案**：

1. ✅ 已修改 `accelerate_configs/deepspeed_16gpu.yaml`
2. 在 worker-0 上：
   ```bash
   export MASTER_ADDR=jo-dbxfe2k2l635222i-worker-0
   export MASTER_PORT=29500
   export RANK=0
   export WORLD_SIZE=2
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ./train_taehv_h100.sh 8 h100
   ```
3. 在 worker-1 上：
   ```bash
   export MASTER_ADDR=jo-dbxfe2k2l635222i-worker-0
   export MASTER_PORT=29500
   export RANK=1
   export WORLD_SIZE=2
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ./train_taehv_h100.sh 8 h100
   ```

就这么简单！🚀


