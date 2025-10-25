#!/bin/bash

################################################################################
# TAEHV 多机训练统一启动脚本 - 适配商用平台（Kubernetes等）
# 
# 使用场景：
#   - 商用容器平台自动在每个节点执行此脚本
#   - 平台通过环境变量提供分布式配置
#   - 不需要手动在每个节点单独启动
#
# 平台环境变量：
#   - WORLD_SIZE: 节点总数（如：2）
#   - RANK: 当前节点编号（0开始）
#   - MASTER_ADDR: Master节点地址
#   - MASTER_PORT: Master端口（默认29500）
################################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

################################################################################
# 1. 分布式环境配置（从平台环境变量读取）
################################################################################

print_header "TAEHV 多机训练环境初始化"

# 获取每个节点的 GPU 数量
GPUS_PER_NODE=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
print_info "检测到每个节点有 ${GPUS_PER_NODE} 个GPU"

# 从平台环境变量读取分布式配置
NNODES=${WORLD_SIZE:-1}           # 节点总数
NODE_RANK=${RANK:-0}              # 当前节点编号
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}

# 计算总进程数
TOTAL_PROCESSES=$((NNODES * GPUS_PER_NODE))

print_info "节点总数: $NNODES"
print_info "当前节点: $NODE_RANK"
print_info "Master地址: $MASTER_ADDR:$MASTER_PORT"
print_info "每节点GPU: $GPUS_PER_NODE"
print_info "总进程数: $TOTAL_PROCESSES"

# 设置CUDA可见设备（使用当前节点的所有GPU）
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

################################################################################
# 2. NCCL优化配置
################################################################################

print_header "配置NCCL通信"

# NCCL基础配置
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-"eth0"}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-"eth0"}
export NCCL_ALGO=${NCCL_ALGO:-"Ring"}

# 如果有InfiniBand，配置IB
if [ ! -z "$NCCL_IB_HCA" ]; then
    print_info "检测到InfiniBand配置: $NCCL_IB_HCA"
    export NCCL_IB_DISABLE=0
else
    export NCCL_IB_DISABLE=1
    print_warning "未配置InfiniBand，使用以太网"
fi

# H100优化的NCCL配置
export NCCL_P2P_DISABLE=0

# 根据节点数调整NCCL超时（节点越多，超时越长）
if [ "$NNODES" -ge 8 ]; then
    export NCCL_TIMEOUT=14400                 # 8+节点：4小时超时
    export NCCL_BUFFSIZE=33554432             # 32MB大缓冲区
    export NCCL_NSOCKS_PERTHREAD=8            # 更多socket
    print_info "使用大规模多机配置（8+节点）"
elif [ "$NNODES" -ge 6 ]; then
    export NCCL_TIMEOUT=10800                 # 6-7节点：3小时超时
    export NCCL_BUFFSIZE=25165824             # 24MB缓冲区
    export NCCL_NSOCKS_PERTHREAD=6            # 适中socket
    print_info "使用中大规模多机配置（6-7节点）"
elif [ "$NNODES" -ge 2 ]; then
    export NCCL_TIMEOUT=7200                  # 2-5节点：2小时超时
    export NCCL_BUFFSIZE=16777216             # 16MB缓冲区
    export NCCL_NSOCKS_PERTHREAD=4
    print_info "使用中等规模多机配置（2-5节点）"
else
    export NCCL_TIMEOUT=3600                  # 单节点：1小时超时
    export NCCL_BUFFSIZE=16777216             # 16MB缓冲区
    export NCCL_NSOCKS_PERTHREAD=4
fi

export NCCL_SOCKET_NTHREADS=8
export NCCL_NET_GDR_LEVEL=5

# PyTorch分布式配置
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-27200}
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误处理

print_success "NCCL配置完成"

################################################################################
# 3. CUDA和PyTorch优化
################################################################################

print_header "配置CUDA优化"

# H100专用配置
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_LAUNCH_BLOCKING=0

# H100性能优化
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# 过滤警告
export PYTHONWARNINGS="ignore::UserWarning"

# 设置OMP线程数
export OMP_NUM_THREADS=1

print_success "CUDA优化配置完成"

################################################################################
# 4. Conda环境激活
################################################################################

print_header "激活Conda环境"

# 初始化conda（如果需要）
if [ -f "/mnt/project_modelware/zhaojian/miniconda3/etc/profile.d/conda.sh" ]; then
    source /mnt/project_modelware/zhaojian/miniconda3/etc/profile.d/conda.sh
    print_success "Conda已初始化"
else
    print_warning "Conda初始化脚本未找到，跳过"
fi

# 激活环境（根据你的实际环境名称修改）
if command -v conda &> /dev/null; then
    conda activate tiny-vae
    print_success "已激活conda环境: tiny-vae"
fi

################################################################################
# 5. 检查训练环境
################################################################################

print_header "验证训练环境"

# 检查Python环境
if python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'支持bf16: {torch.cuda.is_bf16_supported()}')" 2>/dev/null; then
    print_success "Python环境检查通过"
else
    print_error "Python环境检查失败"
    exit 1
fi

# 检查GPU
if nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    print_success "GPU: $GPU_NAME (${GPU_MEMORY}MB)"
else
    print_error "未检测到GPU"
    exit 1
fi

################################################################################
# 6. 配置训练参数
################################################################################

print_header "配置训练参数"

# 根据总进程数选择配置文件
if [ "$TOTAL_PROCESSES" -eq 64 ]; then
    CONFIG_FILE="training/configs/taehv_config_64gpu_h100.py"
    ACCELERATE_CONFIG="accelerate_configs/deepspeed_64gpu.yaml"
    print_info "使用64卡配置（8机器×8GPU）"
elif [ "$TOTAL_PROCESSES" -eq 48 ]; then
    CONFIG_FILE="training/configs/taehv_config_48gpu_h100.py"
    ACCELERATE_CONFIG="accelerate_configs/deepspeed_48gpu.yaml"
    print_info "使用48卡配置（6机器×8GPU）"
elif [ "$TOTAL_PROCESSES" -eq 16 ]; then
    CONFIG_FILE="training/configs/taehv_config_16gpu_h100.py"
    ACCELERATE_CONFIG="accelerate_configs/deepspeed_16gpu.yaml"
    print_info "使用16卡配置（2机器×8GPU）"
elif [ "$TOTAL_PROCESSES" -eq 8 ]; then
    CONFIG_FILE="training/configs/taehv_config_h100.py"
    ACCELERATE_CONFIG="accelerate_configs/deepspeed_8gpu.yaml"
    print_info "使用8卡配置（1机器×8GPU）"
else
    print_error "不支持的GPU配置: ${TOTAL_PROCESSES}卡"
    print_error "支持的配置: 8卡(1×8), 16卡(2×8), 48卡(6×8), 64卡(8×8)"
    exit 1
fi

# 验证配置文件存在
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "配置文件不存在: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$ACCELERATE_CONFIG" ]; then
    print_error "Accelerate配置不存在: $ACCELERATE_CONFIG"
    exit 1
fi

print_success "配置文件: $CONFIG_FILE"
print_success "DeepSpeed配置: $ACCELERATE_CONFIG"

################################################################################
# 7. 创建必要目录
################################################################################

mkdir -p logs output

################################################################################
# 8. 显示启动信息
################################################################################

print_header "启动信息总览"

cat << EOF
节点信息:
  - 总节点数: $NNODES
  - 当前节点: $NODE_RANK
  - 每节点GPU: $GPUS_PER_NODE
  - 总GPU数: $TOTAL_PROCESSES

网络配置:
  - Master地址: $MASTER_ADDR
  - Master端口: $MASTER_PORT
  - 网络接口: $NCCL_SOCKET_IFNAME

训练配置:
  - 训练配置: $CONFIG_FILE
  - DeepSpeed配置: $ACCELERATE_CONFIG
  - GPU列表: $CUDA_VISIBLE_DEVICES

环境优化:
  - NCCL超时: ${NCCL_TIMEOUT}s
  - PyTorch超时: ${TORCH_DISTRIBUTED_TIMEOUT}s
  - InfiniBand: $([ "$NCCL_IB_DISABLE" -eq 0 ] && echo "启用" || echo "禁用")
EOF

################################################################################
# 9. 启动训练（只在Master节点执行实际训练，Worker节点等待）
################################################################################

print_header "启动训练进程"

if [ "$NODE_RANK" -eq 0 ]; then
    print_success "当前是Master节点 (Rank 0)，启动训练..."
    
    # Master节点启动训练
    accelerate launch \
        --config_file $ACCELERATE_CONFIG \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank $NODE_RANK \
        --num_machines $NNODES \
        --num_processes $TOTAL_PROCESSES \
        training/taehv_train.py \
        --config $CONFIG_FILE \
        --mixed_precision bf16 \
        --report_to tensorboard
    
    TRAIN_EXIT_CODE=$?
    
    print_header "训练完成"
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        print_success "训练成功完成"
    else
        print_error "训练异常退出 (退出码: $TRAIN_EXIT_CODE)"
    fi
    
    # 显示输出目录
    LATEST_OUTPUT=$(ls -td output/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST_OUTPUT" ]; then
        print_success "训练结果: $LATEST_OUTPUT"
    fi
    
    exit $TRAIN_EXIT_CODE
    
else
    print_info "当前是Worker节点 (Rank $NODE_RANK)，启动训练进程..."
    
    # Worker节点也需要启动accelerate，但由accelerate内部协调
    accelerate launch \
        --config_file $ACCELERATE_CONFIG \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank $NODE_RANK \
        --num_machines $NNODES \
        --num_processes $TOTAL_PROCESSES \
        training/taehv_train.py \
        --config $CONFIG_FILE \
        --mixed_precision bf16 \
        --report_to tensorboard
    
    TRAIN_EXIT_CODE=$?
    
    print_header "Worker节点完成"
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        print_success "Worker节点训练成功"
    else
        print_error "Worker节点异常退出 (退出码: $TRAIN_EXIT_CODE)"
    fi
    
    exit $TRAIN_EXIT_CODE
fi

