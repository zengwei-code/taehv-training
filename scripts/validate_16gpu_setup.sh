#!/bin/bash

# 16卡H100训练环境验证脚本
# 在启动训练前运行此脚本以验证环境配置

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

ERRORS=0
WARNINGS=0

print_header "16卡H100训练环境验证"

# 1. 检查GPU数量和类型
print_header "1. GPU检查"
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_COUNT" -eq 16 ]; then
    print_success "检测到 16 块 GPU"
else
    print_error "GPU数量不正确: $GPU_COUNT (期望: 16)"
    ((ERRORS++))
fi

# 检查GPU类型
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ $GPU_NAME == *"H100"* ]]; then
    print_success "GPU类型: $GPU_NAME"
else
    print_warning "非H100 GPU: $GPU_NAME (某些优化可能不适用)"
    ((WARNINGS++))
fi

# 检查GPU显存
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEMORY" -ge 80000 ]; then
    print_success "GPU显存: ${GPU_MEMORY}MB"
else
    print_warning "GPU显存不足 80GB: ${GPU_MEMORY}MB"
    ((WARNINGS++))
fi

# 2. 检查配置文件
print_header "2. 配置文件检查"

if [ -f "training/configs/taehv_config_16gpu_h100.py" ]; then
    print_success "16卡训练配置文件存在"
    
    # 验证关键参数
    WORLD_SIZE=$(grep "world_size = " training/configs/taehv_config_16gpu_h100.py | grep -oP '\d+')
    if [ "$WORLD_SIZE" -eq 16 ]; then
        print_success "world_size = 16"
    else
        print_error "world_size != 16 (当前: $WORLD_SIZE)"
        ((ERRORS++))
    fi
    
    # 检查DeepSpeed配置文件路径
    if grep -q "deepspeed_16gpu.yaml" training/configs/taehv_config_16gpu_h100.py; then
        print_success "DeepSpeed配置指向 deepspeed_16gpu.yaml"
    else
        print_error "DeepSpeed配置文件路径不正确"
        ((ERRORS++))
    fi
else
    print_error "找不到 training/configs/taehv_config_16gpu_h100.py"
    ((ERRORS++))
fi

if [ -f "accelerate_configs/deepspeed_16gpu.yaml" ]; then
    print_success "DeepSpeed 16卡配置文件存在"
    
    # 验证num_processes
    NUM_PROCESSES=$(grep "num_processes:" accelerate_configs/deepspeed_16gpu.yaml | grep -oP '\d+')
    if [ "$NUM_PROCESSES" -eq 16 ]; then
        print_success "num_processes = 16"
    else
        print_error "num_processes != 16 (当前: $NUM_PROCESSES)"
        ((ERRORS++))
    fi
    
    # 检查ZeRO stage
    ZERO_STAGE=$(grep "zero_stage:" accelerate_configs/deepspeed_16gpu.yaml | grep -oP '\d+')
    if [ "$ZERO_STAGE" -eq 3 ]; then
        print_success "ZeRO Stage = 3"
    else
        print_warning "ZeRO Stage != 3 (当前: $ZERO_STAGE)"
        ((WARNINGS++))
    fi
else
    print_error "找不到 accelerate_configs/deepspeed_16gpu.yaml"
    ((ERRORS++))
fi

# 3. 检查Python环境
print_header "3. Python环境检查"

if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    print_success "PyTorch: $TORCH_VERSION"
    
    # 检查CUDA
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    print_success "CUDA: $CUDA_VERSION"
    
    # 检查bf16支持
    BF16_SUPPORT=$(python -c "import torch; print(torch.cuda.is_bf16_supported())")
    if [ "$BF16_SUPPORT" = "True" ]; then
        print_success "支持 BF16"
    else
        print_error "不支持 BF16"
        ((ERRORS++))
    fi
    
    # 检查分布式
    NCCL_AVAILABLE=$(python -c "import torch.distributed as dist; print(dist.is_nccl_available())")
    if [ "$NCCL_AVAILABLE" = "True" ]; then
        print_success "NCCL 可用"
    else
        print_error "NCCL 不可用"
        ((ERRORS++))
    fi
else
    print_error "无法导入PyTorch"
    ((ERRORS++))
fi

# 检查关键依赖
if python -c "import accelerate" 2>/dev/null; then
    ACCELERATE_VERSION=$(python -c "import accelerate; print(accelerate.__version__)")
    print_success "Accelerate: $ACCELERATE_VERSION"
else
    print_error "未安装 Accelerate"
    ((ERRORS++))
fi

if python -c "import deepspeed" 2>/dev/null; then
    DEEPSPEED_VERSION=$(python -c "import deepspeed; print(deepspeed.__version__)")
    print_success "DeepSpeed: $DEEPSPEED_VERSION"
else
    print_error "未安装 DeepSpeed"
    ((ERRORS++))
fi

# 检查Flash Attention（可选）
if python -c "import flash_attn" 2>/dev/null; then
    FLASH_ATTN_VERSION=$(python -c "import flash_attn; print(flash_attn.__version__)")
    print_success "Flash Attention: $FLASH_ATTN_VERSION"
else
    print_warning "未安装 Flash Attention (性能优化选项)"
    ((WARNINGS++))
fi

# 4. 检查数据集
print_header "4. 数据集检查"

DATA_ROOT=$(grep 'data_root = ' training/configs/taehv_config_16gpu_h100.py | grep -oP '"/[^"]*"' | tr -d '"')
if [ -d "$DATA_ROOT" ]; then
    VIDEO_COUNT=$(find "$DATA_ROOT" -name "*.mp4" 2>/dev/null | wc -l)
    print_success "数据目录存在: $DATA_ROOT"
    print_success "检测到 $VIDEO_COUNT 个视频文件"
else
    print_warning "数据目录不存在: $DATA_ROOT (如果使用MiniDataset请忽略)"
    ((WARNINGS++))
fi

ANNOTATION_FILE=$(grep 'annotation_file = ' training/configs/taehv_config_16gpu_h100.py | grep -oP '"/[^"]*"' | tr -d '"')
if [ -f "$ANNOTATION_FILE" ]; then
    ANNOTATION_SIZE=$(wc -l < "$ANNOTATION_FILE")
    print_success "标注文件存在: $ANNOTATION_FILE (${ANNOTATION_SIZE} 行)"
else
    print_warning "标注文件不存在: $ANNOTATION_FILE (如果使用MiniDataset请忽略)"
    ((WARNINGS++))
fi

# 5. 检查模型文件
print_header "5. 预训练模型检查"

PRETRAINED_MODEL=$(grep 'pretrained_model_path = ' training/configs/taehv_config_16gpu_h100.py | grep -oP '"[^"]*"' | tr -d '"')
if [ -f "$PRETRAINED_MODEL" ]; then
    MODEL_SIZE=$(du -h "$PRETRAINED_MODEL" | cut -f1)
    print_success "预训练模型存在: $PRETRAINED_MODEL (${MODEL_SIZE})"
else
    print_warning "预训练模型不存在: $PRETRAINED_MODEL"
    ((WARNINGS++))
fi

# 6. 检查磁盘空间
print_header "6. 磁盘空间检查"

OUTPUT_DIR="output"
if [ -d "$OUTPUT_DIR" ]; then
    AVAILABLE_SPACE=$(df -h "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
    print_success "输出目录可用空间: $AVAILABLE_SPACE"
else
    mkdir -p "$OUTPUT_DIR"
    print_success "创建输出目录: $OUTPUT_DIR"
fi

# 检查是否有足够空间（至少500GB）
AVAILABLE_GB=$(df "$OUTPUT_DIR" | awk 'NR==2 {print $4}')
if [ "$AVAILABLE_GB" -lt 524288000 ]; then  # 500GB in KB
    print_warning "可用空间可能不足（建议至少500GB用于检查点）"
    ((WARNINGS++))
fi

# 7. 检查网络配置（多卡训练关键）
print_header "7. 网络配置检查"

# 检查端口是否被占用
MASTER_PORT=29500
if netstat -tuln 2>/dev/null | grep -q ":$MASTER_PORT "; then
    print_warning "端口 $MASTER_PORT 已被占用"
    ((WARNINGS++))
else
    print_success "端口 $MASTER_PORT 可用"
fi

# 检查InfiniBand（如果有）
if command -v ibstat &> /dev/null; then
    IB_STATUS=$(ibstat | grep "State: Active" | wc -l)
    if [ "$IB_STATUS" -gt 0 ]; then
        print_success "InfiniBand 已激活 ($IB_STATUS 个端口)"
    else
        print_warning "InfiniBand 未激活（使用以太网）"
        ((WARNINGS++))
    fi
else
    print_warning "未检测到 InfiniBand（使用以太网）"
    ((WARNINGS++))
fi

# 8. 检查启动脚本
print_header "8. 启动脚本检查"

if [ -f "train_taehv_h100.sh" ]; then
    print_success "启动脚本存在"
    
    # 检查是否支持16卡配置
    if grep -q "taehv_config_16gpu_h100.py" train_taehv_h100.sh; then
        print_success "启动脚本支持16卡配置"
    else
        print_error "启动脚本未配置16卡支持"
        ((ERRORS++))
    fi
    
    # 检查是否可执行
    if [ -x "train_taehv_h100.sh" ]; then
        print_success "启动脚本可执行"
    else
        print_warning "启动脚本不可执行，尝试添加执行权限..."
        chmod +x train_taehv_h100.sh
        print_success "已添加执行权限"
    fi
else
    print_error "找不到 train_taehv_h100.sh"
    ((ERRORS++))
fi

# 9. 环境变量建议
print_header "9. 环境变量建议"

echo -e "${BLUE}建议的环境变量配置（启动脚本会自动设置）：${NC}"
cat << EOF

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=7200
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_BUFFSIZE=16777216
export NCCL_NET_GDR_LEVEL=5
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

EOF

# 10. 总结
print_header "验证总结"

if [ $ERRORS -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}✓ 所有检查通过！可以启动16卡训练。${NC}"
        echo -e "\n${BLUE}启动命令：${NC}"
        echo -e "  ./train_taehv_h100.sh 16 h100"
    else
        echo -e "${YELLOW}⚠ 检查完成，有 $WARNINGS 个警告（可以继续）${NC}"
        echo -e "\n${BLUE}启动命令：${NC}"
        echo -e "  ./train_taehv_h100.sh 16 h100"
    fi
else
    echo -e "${RED}✗ 发现 $ERRORS 个错误，$WARNINGS 个警告${NC}"
    echo -e "${RED}请修复错误后再启动训练${NC}"
    exit 1
fi

# 显示关键配置
print_header "关键配置参数"
cat << EOF

配置文件: training/configs/taehv_config_16gpu_h100.py
DeepSpeed配置: accelerate_configs/deepspeed_16gpu.yaml

训练参数:
- GPU数量: 16
- Per-GPU Batch Size: 8
- 梯度累积: 1
- 有效Batch Size: 128 (8×16×1)
- 学习率: 1e-4
- Warmup Steps: 1500
- 分辨率: 720×480×12帧
- ZeRO Stage: 3

预期性能:
- 训练速度: ~0.8-1.0 steps/sec
- 吞吐量: ~100-128 samples/sec
- Per-GPU显存: ~60-70GB

EOF

exit 0

