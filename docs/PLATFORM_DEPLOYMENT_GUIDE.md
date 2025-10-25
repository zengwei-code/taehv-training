# 商用平台多机训练部署指南

## 🎯 适用场景

本指南适用于使用**商用容器平台**（如Kubernetes、KubeBrain等）进行多机分布式训练的场景。

**平台特点**：
- ✅ 自动在每个节点运行同一个启动脚本
- ✅ 通过环境变量提供分布式配置
- ✅ 无需手动登录每个节点
- ✅ 统一的作业提交方式

---

## 📦 环境变量说明

商用平台会自动设置以下环境变量：

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `WORLD_SIZE` | 节点总数 | `2` |
| `RANK` | 当前节点编号（从0开始） | `0` 或 `1` |
| `MASTER_ADDR` | Master节点地址 | `jo-dbxfe2k2l635222i-worker-0` |
| `MASTER_PORT` | Master端口 | `29500` |

**脚本会自动读取这些变量，无需手动设置！**

---

## 🚀 快速开始

### 1. 准备启动脚本

使用 `start_multinode_platform.sh`：

```bash
# 脚本已创建在项目根目录
ls -lh start_multinode_platform.sh
```

### 2. 在平台提交作业

根据你的平台类型，提交方式略有不同：

#### Kubernetes (kubectl)
```yaml
# job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: taehv-training-16gpu
spec:
  parallelism: 2  # 2个节点
  template:
    spec:
      containers:
      - name: training
        image: your-training-image
        command: ["/bin/bash"]
        args: ["/workspace/my_taehv_training/start_multinode_platform.sh"]
        resources:
          limits:
            nvidia.com/gpu: 8  # 每节点8个GPU
```

#### KubeBrain / 自定义平台
```bash
# 直接指定启动脚本路径
STARTUP_SCRIPT=/mnt/project_modelware/zhaojian/matrix_rfl/my_taehv_training/start_multinode_platform.sh
```

---

## 📝 配置修改

### 1. 修改Conda环境名称

编辑 `start_multinode_platform.sh` 第97行：

```bash
# 改为你的实际环境名称
conda activate tiny-vae  # 改为你的环境名
```

### 2. 修改Conda路径（如果需要）

编辑第84行：

```bash
if [ -f "/mnt/project_modelware/zhaojian/miniconda3/etc/profile.d/conda.sh" ]; then
    # 改为你的实际路径
```

### 3. 添加自定义环境变量

在第80行后添加：

```bash
# 你的自定义配置
export YOUR_CUSTOM_VAR="value"
```

---

## 🔍 脚本工作流程

### 阶段1: 环境初始化
```
1. 读取平台环境变量 (WORLD_SIZE, RANK, etc.)
2. 检测每节点GPU数量
3. 计算总进程数
4. 配置CUDA_VISIBLE_DEVICES
```

### 阶段2: NCCL配置
```
1. 设置NCCL通信参数
2. 配置InfiniBand（如果有）
3. 设置超时和优化参数
```

### 阶段3: 环境验证
```
1. 激活Conda环境
2. 检查Python/PyTorch
3. 验证GPU可用性
```

### 阶段4: 选择配置
```
根据总GPU数量自动选择:
- 16卡 → taehv_config_16gpu_h100.py
- 8卡  → taehv_config_h100.py
```

### 阶段5: 启动训练
```
Master节点 (RANK=0):  启动训练，协调所有节点
Worker节点 (RANK>0):  启动进程，等待Master协调
```

---

## 📊 支持的配置

| GPU配置 | 节点数 | 每节点GPU | 训练配置 | DeepSpeed配置 |
|---------|--------|-----------|----------|--------------|
| 8卡 | 1 | 8 | `taehv_config_h100.py` | `deepspeed_8gpu.yaml` |
| 16卡 | 2 | 8 | `taehv_config_16gpu_h100.py` | `deepspeed_16gpu.yaml` |

---

## ✅ 验证部署

### 1. 检查脚本输出

成功启动后应该看到：

```
==========================================
TAEHV 多机训练环境初始化
==========================================
[INFO] 检测到每个节点有 8 个GPU
[INFO] 节点总数: 2
[INFO] 当前节点: 0 (或 1)
[INFO] Master地址: xxx.xxx.xxx.xxx:29500
[INFO] 每节点GPU: 8
[INFO] 总进程数: 16
...
[SUCCESS] 当前是Master节点 (Rank 0)，启动训练...
```

### 2. 检查GPU使用

在平台监控页面或通过日志查看：
- 每个节点的8个GPU都在工作
- 显存使用在45-60GB范围

### 3. 检查训练日志

```
[INFO] Initialized process group with 16 processes
[INFO] World size: 16, Rank: 0-15
Step 1/100000 | Loss: 0.15 | PSNR: 18.5 dB
```

---

## 🔧 故障排查

### 问题1: NCCL初始化失败

**症状**：
```
NCCL error: unhandled system error
```

**解决**：
1. 检查网络接口名称：
   ```bash
   ip addr  # 查看实际接口名
   # 修改脚本中的 NCCL_SOCKET_IFNAME
   ```

2. 检查防火墙：
   ```bash
   # 确保Master端口开放
   # 通常是29500
   ```

### 问题2: Worker节点无法连接Master

**症状**：
```
Connection refused to MASTER_ADDR:MASTER_PORT
```

**解决**：
1. 验证MASTER_ADDR正确：
   ```bash
   echo $MASTER_ADDR
   # 应该是worker-0的实际地址
   ```

2. 检查端口可用性：
   ```bash
   nc -zv $MASTER_ADDR $MASTER_PORT
   ```

### 问题3: GPU设备号错误

**症状**：
```
RuntimeError: CUDA error: invalid device ordinal
```

**解决**：
检查CUDA_VISIBLE_DEVICES设置：
```bash
# 在脚本中确认
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 问题4: Conda环境未激活

**症状**：
```
ModuleNotFoundError: No module named 'torch'
```

**解决**：
1. 检查Conda路径：
   ```bash
   # 修改脚本中的conda路径
   source /your/path/to/conda.sh
   ```

2. 检查环境名称：
   ```bash
   conda activate your-env-name
   ```

---

## 📈 性能优化建议

### 1. InfiniBand配置

如果平台支持InfiniBand，在脚本中设置：

```bash
# 查找IB设备
ibstat

# 设置环境变量
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
```

### 2. 调整NCCL算法

根据网络拓扑选择：

```bash
# Ring算法（默认）
export NCCL_ALGO=Ring

# Tree算法（某些情况更快）
export NCCL_ALGO=Tree
```

### 3. 增加超时时间

如果网络延迟较高：

```bash
export NCCL_TIMEOUT=10800  # 3小时
export TORCH_DISTRIBUTED_TIMEOUT=10800
```

---

## 🎓 与原启动脚本对比

| 对比项 | 原脚本 (`train_taehv_h100.sh`) | 平台脚本 (`start_multinode_platform.sh`) |
|--------|--------------------------------|------------------------------------------|
| **适用场景** | 手动SSH登录每台机器 | 商用平台统一启动 |
| **环境变量** | 手动设置 | 平台自动提供 |
| **启动方式** | 每台机器单独执行 | 所有节点执行同一脚本 |
| **Master识别** | 通过脚本参数 | 通过RANK=0判断 |
| **配置复杂度** | 中 | 低 |

---

## 📋 完整的作业提交示例

### 示例1: KubeBrain平台

```yaml
# kubebrain_job.yaml
apiVersion: kubebrain.io/v1
kind: TrainingJob
metadata:
  name: taehv-16gpu-training
spec:
  replicaSpecs:
    Worker:
      replicas: 2  # 2个节点
      template:
        spec:
          containers:
          - name: training
            image: your-registry/taehv-training:latest
            command:
            - /bin/bash
            - /mnt/project_modelware/zhaojian/matrix_rfl/my_taehv_training/start_multinode_platform.sh
            resources:
              limits:
                nvidia.com/gpu: 8
            volumeMounts:
            - name: data
              mountPath: /mnt/project_modelware
          volumes:
          - name: data
            hostPath:
              path: /mnt/project_modelware
```

提交作业：
```bash
kubectl apply -f kubebrain_job.yaml
```

### 示例2: 直接命令行提交

```bash
# 在平台上提交（具体命令根据平台而定）
submit_job \
  --name taehv-16gpu \
  --workers 2 \
  --gpus-per-worker 8 \
  --script /path/to/start_multinode_platform.sh
```

---

## 🔍 日志查看

### 查看Master节点日志
```bash
# 查看Rank 0的日志
kubectl logs taehv-training-16gpu-worker-0
```

### 查看Worker节点日志
```bash
# 查看Rank 1的日志
kubectl logs taehv-training-16gpu-worker-1
```

### 查看所有节点日志
```bash
# 聚合查看
kubectl logs -l job-name=taehv-training-16gpu
```

---

## ✨ 关键差异总结

### 原方式（手动启动）
```bash
# Worker-0
./start_worker0_16gpu.sh

# Worker-1
./start_worker1_16gpu.sh
```
❌ 需要登录每台机器  
❌ 手动设置环境变量  
❌ 启动时序要求严格

### 新方式（平台统一）
```bash
# 平台自动在所有节点执行
./start_multinode_platform.sh
```
✅ 一次提交，自动分发  
✅ 环境变量自动配置  
✅ 节点自动协调

---

## 📞 获取帮助

如遇问题：
1. 查看脚本输出的详细日志
2. 检查平台的作业状态页面
3. 查看训练日志: `logs/train_*.log`
4. 参考平台文档的分布式训练章节

---

## 🎉 总结

使用 `start_multinode_platform.sh`，你可以：
- ✅ 在商用平台上一键启动多机训练
- ✅ 自动适配平台提供的环境变量
- ✅ 无需手动配置每个节点
- ✅ 支持8卡和16卡配置

**现在就提交你的训练作业吧！** 🚀

---

**创建日期**: 2025-10-17  
**版本**: 1.0.0  
**维护**: TAEHV Team


