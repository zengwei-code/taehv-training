#!/bin/bash

# Worker-1 启动脚本 - 多机16卡训练 (2×8)
# 在 worker-1 机器上执行此脚本

set -e

echo "=========================================="
echo "启动 Worker-1 (Rank 1)"
echo "=========================================="

# 多机分布式配置
export MASTER_ADDR=jo-dbxfe2k2l635222i-worker-0
export MASTER_PORT=29500
export RANK=1
export WORLD_SIZE=2

# 每台机器使用8个GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Master地址: $MASTER_ADDR:$MASTER_PORT"
echo "当前机器Rank: $RANK"
echo "总机器数: $WORLD_SIZE"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""
echo "等待 Worker-0 启动..."
sleep 5
echo "开始启动 Worker-1..."
echo ""

# 启动8卡训练
./train_taehv_h100.sh 8 h100

