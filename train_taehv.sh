#!/bin/bash

# TAEHV训练启动脚本
# 使用accelerate + deepspeed进行分布式训练

set -e

# 环境配置
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export TOKENIZERS_PARALLELISM=true

# NCCL优化配置 - 针对通信超时问题的优化
export NCCL_P2P_DISABLE=1                    # 禁用P2P，使用更稳定的通信方式
export NCCL_TIMEOUT=7200                     # 增加超时时间到2小时
export NCCL_BLOCKING_WAIT=1                  # 启用阻塞等待
export NCCL_IB_DISABLE=1                     # 禁用InfiniBand（如果有网络问题）
export NCCL_SOCKET_NTHREADS=4               # 增加socket线程数
export NCCL_NSOCKS_PERTHREAD=8              # 增加每线程socket数
export NCCL_BUFFSIZE=8388608                # 增加缓冲区大小(8MB)
export NCCL_NET_GDR_LEVEL=5                 # 优化GPU Direct RDMA
export NCCL_DEBUG=INFO                       # 启用调试信息（可选，调试时开启）

# PyTorch分布式优化
export TORCH_NCCL_ENABLE_MONITORING=0       # 禁用NCCL监控减少开销 
export TORCH_NCCL_BLOCKING_WAIT=1           # PyTorch层面的阻塞等待
export TORCH_DISTRIBUTED_DEBUG=INFO         # 启用分布式调试信息

# CUDA优化设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 分布式训练设置
if [ -z "$MASTER_ADDR" ]; then
    export MASTER_ADDR=localhost
fi
if [ -z "$MASTER_PORT" ]; then
    export MASTER_PORT=8000
fi
if [ -z "$RANK" ]; then
    export RANK=0
fi
if [ -z "$WORLD_SIZE" ]; then
    export WORLD_SIZE=1
fi

# 配置文件
GPU_IDS="0,1,2,3,4,5,6,7"
ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed.yaml"
CONFIG_FILE="training/configs/taehv_config.py"

# 检查文件是否存在
echo "检查必要文件..."
if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
    echo "错误: 找不到accelerate配置文件 $ACCELERATE_CONFIG_FILE"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 找不到训练配置文件 $CONFIG_FILE"
    exit 1
fi

echo "✓ 配置文件检查完成"

# 检查conda环境
if conda list | grep -q "accelerate\|torch"; then
    echo "✓ 检测到accelerate和pytorch环境"
else
    echo "警告: 请确保已安装accelerate和pytorch"
fi

# 检查GPU
if nvidia-smi > /dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ 检测到 $GPU_COUNT 张GPU"
else
    echo "警告: 未检测到GPU"
fi

# 创建输出目录
mkdir -p logs

echo ""
echo "=========================================="
echo "TAEHV训练启动"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "DeepSpeed配置: $ACCELERATE_CONFIG_FILE"
echo "GPU数量: 8"
echo "Master地址: $MASTER_ADDR:$MASTER_PORT"
echo "=========================================="
echo ""

# 启动训练
accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --num_machines $WORLD_SIZE \
    --num_processes $((WORLD_SIZE * 8)) \
    --gpu_ids $GPU_IDS \
    training/taehv_train.py \
    --config $CONFIG_FILE \
    --mixed_precision bf16 \
    --report_to tensorboard

echo ""
echo "=========================================="
echo "训练完成"
echo "=========================================="

# 查找最新的输出目录
LATEST_OUTPUT=$(ls -td output/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_OUTPUT" ]; then
    echo "训练结果保存在: $LATEST_OUTPUT"
    echo ""
    echo "要测试推理效果，请运行:"
    echo "python inference.py --model_path ${LATEST_OUTPUT}final_model.pth"
    echo ""
    echo "要查看训练日志，请运行:"
    echo "tensorboard --logdir ${LATEST_OUTPUT}logs/"
else
    echo "未找到训练输出目录"
fi
