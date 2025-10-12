#!/bin/bash

# TAEHV生产环境训练启动脚本 - 支持多GPU配置选择
# 针对H100 80G环境优化

set -e

# 颜色输出函数
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# 显示使用说明
show_usage() {
    echo "用法: $0 [GPU数量] [配置类型] [其他选项]"
    echo ""
    echo "GPU数量选项:"
    echo "  1     - 单卡训练 (推荐用于快速测试)"
    echo "  8     - 8卡训练 (推荐用于H100生产)"
    echo "  16    - 16卡训练"
    echo ""
    echo "配置类型:"
    echo "  test     - 测试配置 (MiniDataset)"
    echo "  h100     - H100配置 (自动根据GPU数量选择单卡或多卡配置)"
    echo "  a800     - A800测试H100脚本配置 (仅支持8卡)"
    echo "  custom   - 自定义配置文件路径"
    echo ""
    echo "其他选项:"
    echo "  --resume PATH    - 从检查点恢复"
    echo "  --debug         - 启用调试模式"
    echo "  --profile       - 启用性能分析"
    echo ""
    echo "示例:"
    echo "  # 单卡H100训练 (自动使用单卡配置: 720×480×19帧)"
    echo "  $0 1 h100"
    echo ""
    echo "  # 8卡H100生产训练 (自动使用多卡配置)"
    echo "  $0 8 h100"
    echo ""
    echo "  # A800 8卡测试H100脚本"
    echo "  $0 8 a800"
    echo ""
    echo "  # 从检查点恢复"
    echo "  $0 8 h100 --resume output/checkpoint-5000"
}

# 默认参数
GPU_COUNT=${1:-8}
CONFIG_TYPE=${2:-"h100"}
RESUME_PATH=""
DEBUG_MODE=false
PROFILE_MODE=false

# 解析命令行参数
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --profile)
            PROFILE_MODE=true
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 验证GPU数量
if ! [[ "$GPU_COUNT" =~ ^(1|8|16)$ ]]; then
    print_error "不支持的GPU数量: $GPU_COUNT"
    print_info "支持的GPU数量: 1, 8, 16"
    exit 1
fi

# 设置配置文件
case $CONFIG_TYPE in
    "test")
        CONFIG_FILE="training/configs/taehv_config.py"
        print_info "使用测试配置 (MiniDataset)"
        ;;
    "h100")
        # 根据GPU数量自动选择H100配置
        if [ "$GPU_COUNT" -eq 1 ]; then
            CONFIG_FILE="training/configs/taehv_config_1gpu_h100.py"
            print_info "使用单卡H100配置 (720×480×19帧)"
        else
            CONFIG_FILE="training/configs/taehv_config_h100.py"
            print_info "使用${GPU_COUNT}卡H100生产配置"
        fi
        ;;
    "a800")
        CONFIG_FILE="training/configs/taehv_config_a800.py"
        if [ "$GPU_COUNT" -ne 8 ]; then
            print_warning "A800配置仅支持8卡"
            exit 1
        fi
        # A800和H100使用相同的DeepSpeed配置
        print_info "使用A800测试H100脚本配置（仅支持8卡，共享DeepSpeed配置）"
        ;;
    "custom")
        CONFIG_FILE="$3"
        if [ -z "$CONFIG_FILE" ]; then
            print_error "使用custom配置时必须提供配置文件路径"
            exit 1
        fi
        ;;
    *)
        print_error "不支持的配置类型: $CONFIG_TYPE"
        print_info "支持的配置类型: test, h100, 1gpu_h100, a800, custom"
        exit 1
        ;;
esac

# 设置accelerate配置文件
if [ "$GPU_COUNT" -eq 1 ]; then
    # 单卡训练可以选择使用DeepSpeed或不使用
    # 推荐使用deepspeed_1gpu.yaml以获得更好的优化效果
    ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_1gpu.yaml"
    print_info "单卡训练使用DeepSpeed优化配置"
else
    ACCELERATE_CONFIG_FILE="accelerate_configs/deepspeed_${GPU_COUNT}gpu.yaml"
fi

# H100优化环境配置
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
if [ "$DEBUG_MODE" = true ]; then
    export TORCHDYNAMO_VERBOSE=1
fi
export TOKENIZERS_PARALLELISM=false  # 设置为false减少警告

# H100 NCCL优化配置（仅多卡训练需要）
if [ "$GPU_COUNT" -gt 1 ]; then
    export NCCL_P2P_DISABLE=0                    # H100支持P2P，启用以获得更好性能
    export NCCL_TIMEOUT=7200                     # 保持较长超时
    # export NCCL_BLOCKING_WAIT=1                # 已弃用，使用TORCH_NCCL_BLOCKING_WAIT
    export NCCL_IB_DISABLE=0                     # 如果有InfiniBand则启用
    export NCCL_SOCKET_NTHREADS=8               # H100优化的线程数
    export NCCL_NSOCKS_PERTHREAD=4              # H100优化
    export NCCL_BUFFSIZE=16777216               # 16MB缓冲区适配H100
    export NCCL_NET_GDR_LEVEL=5                 # 启用GPU Direct RDMA
    if [ "$DEBUG_MODE" = true ]; then
        export NCCL_DEBUG=INFO
    fi
    
    # PyTorch分布式优化
    export TORCH_NCCL_ENABLE_MONITORING=0       # 禁用监控减少开销
    export TORCH_NCCL_BLOCKING_WAIT=1           # PyTorch层面的阻塞等待 (新API)
    export OMP_NUM_THREADS=1                   # 设置为1减少OMP_NUM_THREADS警告
    if [ "$DEBUG_MODE" = true ]; then
        export TORCH_DISTRIBUTED_DEBUG=INFO
    fi
    print_info "已配置多卡NCCL优化"
else
    print_info "单卡训练，跳过NCCL配置"
fi

# CUDA优化设置 - H100/A800专用
GPU_IDS=""
for ((i=0; i<GPU_COUNT; i++)); do
    if [ $i -eq 0 ]; then
        GPU_IDS="$i"
    else
        GPU_IDS="$GPU_IDS,$i"
    fi
done

export CUDA_VISIBLE_DEVICES=$GPU_IDS

# 根据配置类型设置不同的CUDA优化
if [ "$CONFIG_TYPE" = "a800" ]; then
    # A800显存优化配置
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,roundup_power2_divisions:16"
    export TORCH_CUDA_ARCH_LIST="8.0"  # A800架构
    print_info "使用A800显存优化配置"
else
    # H100标准配置
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
    export TORCH_CUDA_ARCH_LIST="9.0"  # H100架构
fi

# 过滤CUDA警告
export PYTHONWARNINGS="ignore::UserWarning"
export CUDA_LAUNCH_BLOCKING=0

# H100性能优化
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# 分布式训练设置
if [ -z "$MASTER_ADDR" ]; then
    export MASTER_ADDR=localhost
fi
if [ -z "$MASTER_PORT" ]; then
    export MASTER_PORT=29500  # 避免与其他服务冲突
fi
if [ -z "$RANK" ]; then
    export RANK=0
fi
if [ -z "$WORLD_SIZE" ]; then
    export WORLD_SIZE=1
fi

# 检查文件是否存在
print_info "检查必要文件..."
if [ ! -f "$ACCELERATE_CONFIG_FILE" ]; then
    print_error "找不到accelerate配置文件 $ACCELERATE_CONFIG_FILE"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    print_error "找不到训练配置文件 $CONFIG_FILE"
    exit 1
fi

print_success "配置文件检查完成"

# 检查环境
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_success "检测到GPU: $GPU_NAME (${GPU_MEMORY}MB显存)"
    
    # 验证是否为H100
    if [[ $GPU_NAME == *"H100"* ]]; then
        print_success "确认H100环境，启用所有优化"
    else
        print_warning "非H100环境，某些优化可能不适用"
    fi
else
    print_warning "未检测到GPU"
fi

# 检查Python环境
if python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'支持bf16: {torch.cuda.is_bf16_supported()}')" 2>/dev/null; then
    print_success "Python环境检查通过"
else
    print_error "Python环境检查失败"
    exit 1
fi

# 创建输出目录
mkdir -p logs output

# 如果指定了恢复路径，验证其存在
if [ -n "$RESUME_PATH" ]; then
    if [ ! -d "$RESUME_PATH" ]; then
        print_error "恢复路径不存在: $RESUME_PATH"
        exit 1
    fi
    print_info "将从检查点恢复: $RESUME_PATH"
    RESUME_FLAG="--resume_from_checkpoint $RESUME_PATH"
else
    RESUME_FLAG=""
fi

echo ""
echo "=========================================="
echo "TAEHV H100生产环境训练启动"
echo "=========================================="
echo "GPU配置: ${GPU_COUNT}x H100 80G"
echo "GPU IDs: $GPU_IDS"
echo "配置类型: $CONFIG_TYPE"
echo "训练配置: $CONFIG_FILE"
echo "DeepSpeed配置: $ACCELERATE_CONFIG_FILE"
echo "Master地址: $MASTER_ADDR:$MASTER_PORT"
if [ -n "$RESUME_PATH" ]; then
    echo "恢复训练: $RESUME_PATH"
fi
echo "调试模式: $DEBUG_MODE"
echo "性能分析: $PROFILE_MODE"
echo "=========================================="
echo ""

# 启动训练
LAUNCH_ARGS=""
if [ "$PROFILE_MODE" = true ]; then
    LAUNCH_ARGS="$LAUNCH_ARGS --use_deepspeed_engine"
fi

accelerate launch \
    --config_file $ACCELERATE_CONFIG_FILE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $RANK \
    --num_machines $WORLD_SIZE \
    --num_processes $((WORLD_SIZE * GPU_COUNT)) \
    --gpu_ids $GPU_IDS \
    $LAUNCH_ARGS \
    training/taehv_train.py \
    --config $CONFIG_FILE \
    --mixed_precision bf16 \
    --report_to tensorboard \
    $RESUME_FLAG

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    print_success "训练完成"
else
    print_error "训练异常退出 (退出码: $TRAIN_EXIT_CODE)"
fi
echo "=========================================="

# 查找最新的输出目录
LATEST_OUTPUT=$(ls -td output/*/ 2>/dev/null | head -1)
if [ -n "$LATEST_OUTPUT" ]; then
    print_success "训练结果保存在: $LATEST_OUTPUT"
    echo ""
    print_info "要测试推理效果，请运行:"
    echo "python inference.py --model_path ${LATEST_OUTPUT}final_model.pth"
    echo ""
    print_info "要查看训练日志，请运行:"
    echo "tensorboard --logdir ${LATEST_OUTPUT}logs/"
else
    print_warning "未找到训练输出目录"
fi

exit $TRAIN_EXIT_CODE
