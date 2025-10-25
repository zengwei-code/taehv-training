#!/bin/bash

# 16卡H100训练支持 - 文件完整性验证脚本
# 验证所有新增和修改的文件是否正确部署

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
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

ERRORS=0

print_header "16卡H100训练支持 - 文件完整性验证"

# 检查新增文件
print_header "1. 检查新增配置文件"

if [ -f "training/configs/taehv_config_16gpu_h100.py" ]; then
    print_success "training/configs/taehv_config_16gpu_h100.py"
    
    # 验证关键参数
    if grep -q "world_size = 16" training/configs/taehv_config_16gpu_h100.py; then
        print_success "  world_size = 16"
    else
        print_error "  world_size 配置错误"
        ((ERRORS++))
    fi
    
    if grep -q "deepspeed_16gpu.yaml" training/configs/taehv_config_16gpu_h100.py; then
        print_success "  deepspeed_config_file = deepspeed_16gpu.yaml"
    else
        print_error "  DeepSpeed配置文件路径错误"
        ((ERRORS++))
    fi
    
    if grep -q "gradient_accumulation_steps = 1" training/configs/taehv_config_16gpu_h100.py; then
        print_success "  gradient_accumulation_steps = 1"
    else
        print_error "  梯度累积步数配置错误"
        ((ERRORS++))
    fi
else
    print_error "training/configs/taehv_config_16gpu_h100.py 不存在"
    ((ERRORS++))
fi

# 检查脚本文件
print_header "2. 检查验证脚本"

if [ -f "scripts/validate_16gpu_setup.sh" ]; then
    print_success "scripts/validate_16gpu_setup.sh"
    
    if [ -x "scripts/validate_16gpu_setup.sh" ]; then
        print_success "  可执行权限正常"
    else
        print_error "  缺少执行权限"
        chmod +x scripts/validate_16gpu_setup.sh
        print_success "  已添加执行权限"
    fi
else
    print_error "scripts/validate_16gpu_setup.sh 不存在"
    ((ERRORS++))
fi

# 检查文档文件
print_header "3. 检查文档文件"

# 检查完整指南文档
if [ -f "docs/16GPU_COMPLETE_GUIDE.md" ]; then
    SIZE=$(wc -l < "docs/16GPU_COMPLETE_GUIDE.md")
    print_success "docs/16GPU_COMPLETE_GUIDE.md (${SIZE} 行)"
else
    print_error "docs/16GPU_COMPLETE_GUIDE.md 不存在"
    ((ERRORS++))
fi

# 检查启动脚本修改
print_header "4. 检查启动脚本修改"

if [ -f "train_taehv_h100.sh" ]; then
    print_success "train_taehv_h100.sh 存在"
    
    # 检查16卡支持
    if grep -q "taehv_config_16gpu_h100.py" train_taehv_h100.sh; then
        print_success "  包含16卡配置支持"
    else
        print_error "  缺少16卡配置支持"
        ((ERRORS++))
    fi
    
    # 检查GPU数量16的判断逻辑
    if grep -q 'GPU_COUNT.*-eq 16' train_taehv_h100.sh; then
        print_success "  包含16卡判断逻辑"
    else
        print_error "  缺少16卡判断逻辑"
        ((ERRORS++))
    fi
    
    if [ -x "train_taehv_h100.sh" ]; then
        print_success "  可执行权限正常"
    else
        print_error "  缺少执行权限"
        ((ERRORS++))
    fi
else
    print_error "train_taehv_h100.sh 不存在"
    ((ERRORS++))
fi

# 检查DeepSpeed配置
print_header "5. 检查DeepSpeed配置"

if [ -f "accelerate_configs/deepspeed_16gpu.yaml" ]; then
    print_success "accelerate_configs/deepspeed_16gpu.yaml"
    
    # 验证关键参数
    if grep -q "num_processes: 16" accelerate_configs/deepspeed_16gpu.yaml; then
        print_success "  num_processes = 16"
    else
        print_error "  num_processes 配置错误"
        ((ERRORS++))
    fi
    
    if grep -q "zero_stage: 3" accelerate_configs/deepspeed_16gpu.yaml; then
        print_success "  zero_stage = 3"
    else
        print_error "  zero_stage 配置错误"
        ((ERRORS++))
    fi
else
    print_error "accelerate_configs/deepspeed_16gpu.yaml 不存在"
    ((ERRORS++))
fi

# 显示文件结构
print_header "6. 文件结构总览"

echo ""
echo "新增/修改的文件："
echo ""
echo "📁 training/configs/"
echo "  └── taehv_config_16gpu_h100.py    [新增] 16卡专用配置"
echo ""
echo "📁 accelerate_configs/"
echo "  └── deepspeed_16gpu.yaml          [已存在] ZeRO-3配置"
echo ""
echo "📁 scripts/"
echo "  ├── validate_16gpu_setup.sh       [新增] 环境验证"
echo "  └── verify_16gpu_changes.sh       [新增] 文件验证"
echo ""
echo "📁 docs/"
echo "  └── 16GPU_COMPLETE_GUIDE.md       [新增] 完整指南（26KB）"
echo ""
echo "📁 根目录"
echo "  └── train_taehv_h100.sh           [已修改] 添加16卡支持"
echo ""

# 总结
print_header "验证总结"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有文件验证通过！16卡H100训练支持已完整部署。${NC}"
    echo ""
    echo -e "${BLUE}下一步：${NC}"
    echo "1. 查看文档: cat docs/16GPU_COMPLETE_GUIDE.md"
    echo "2. 验证环境: ./scripts/validate_16gpu_setup.sh"
    echo "3. 启动训练: ./train_taehv_h100.sh 16 h100"
    echo ""
else
    echo -e "${RED}✗ 发现 $ERRORS 个错误，请检查上述问题${NC}"
    exit 1
fi

# 显示关键信息
print_header "关键配置信息"

cat << 'EOF'

配置对比 (8卡 → 16卡):
┌────────────────────────────┬──────────┬──────────┐
│ 参数                       │ 8卡      │ 16卡     │
├────────────────────────────┼──────────┼──────────┤
│ world_size                 │ 8        │ 16       │
│ gradient_accumulation      │ 2        │ 1        │
│ effective_batch_size       │ 128      │ 128      │
│ zero_stage                 │ 2        │ 3        │
│ dataloader_workers         │ 8        │ 12       │
│ checkpointing_steps        │ 2000     │ 1000     │
│ validation_steps           │ 1000     │ 500      │
│ nccl_timeout               │ 3600s    │ 7200s    │
└────────────────────────────┴──────────┴──────────┘

性能预期:
- 训练速度: ~0.8-1.0 steps/sec
- 吞吐量: ~100-128 samples/sec
- Per-GPU显存: ~45-60GB
- 100k步耗时: ~31小时

EOF

exit 0

