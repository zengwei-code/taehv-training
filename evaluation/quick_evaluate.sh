#!/bin/bash
# 快速评估脚本 - 一键运行完整评估流程

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
MODEL_PATH="../output/2025-10-01_19-59-50/final_model.pth"
LOG_DIR="../logs/taehv_h100_production"
CONFIG="../training/configs/taehv_config_h100.py"
NUM_SAMPLES=100
OUTPUT_DIR="../evaluation_results"

# 打印带颜色的消息
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: ./quick_evaluate.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --model_path PATH    模型路径 (默认: output/2025-10-01_19-59-50/final_model.pth)"
            echo "  --log_dir PATH       日志目录 (默认: logs/taehv_h100_production)"
            echo "  --num_samples N      评估样本数 (默认: 100)"
            echo "  --help               显示帮助信息"
            echo ""
            echo "示例:"
            echo "  ./quick_evaluate.sh"
            echo "  ./quick_evaluate.sh --model_path output/xxx/final_model.pth --num_samples 200"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 打印配置
print_header "🚀 快速评估 Tiny-VAE 模型"
echo "模型路径: $MODEL_PATH"
echo "日志目录: $LOG_DIR"
echo "评估样本数: $NUM_SAMPLES"
echo "结果目录: $OUTPUT_DIR"
echo ""

# 检查文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    print_error "模型文件不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$LOG_DIR" ]; then
    print_warning "日志目录不存在: $LOG_DIR，将跳过训练日志分析"
    SKIP_LOG_ANALYSIS=true
else
    SKIP_LOG_ANALYSIS=false
fi

if [ ! -f "$CONFIG" ]; then
    print_error "配置文件不存在: $CONFIG"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# ============================================
# 步骤1: 分析训练日志
# ============================================
if [ "$SKIP_LOG_ANALYSIS" = false ]; then
    print_header "📊 步骤1/3: 分析训练日志"
    
    python analyze_training_logs.py \
        --log_dir "$LOG_DIR" \
        --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        print_success "训练日志分析完成"
        echo "  - 损失曲线: $OUTPUT_DIR/training_losses.png"
        echo "  - 指标曲线: $OUTPUT_DIR/training_metrics.png"
        echo "  - 分析报告: $OUTPUT_DIR/training_analysis.json"
    else
        print_warning "训练日志分析失败，继续后续评估..."
    fi
else
    print_warning "跳过训练日志分析"
fi

echo ""

# ============================================
# 步骤2: 模型定量评估
# ============================================
print_header "🔍 步骤2/3: 模型定量评估"

python evaluate_vae.py \
    --model_path "$MODEL_PATH" \
    --config "$CONFIG" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size 4

if [ $? -eq 0 ]; then
    print_success "模型评估完成"
    echo "  - 评估结果: $OUTPUT_DIR/evaluation_results.json"
    echo "  - 指标分布: $OUTPUT_DIR/metrics_distribution.png"
    echo "  - 可视化样本: $OUTPUT_DIR/sample_*.png"
else
    print_error "模型评估失败"
    exit 1
fi

echo ""

# ============================================
# 步骤3: 生成评估报告摘要
# ============================================
print_header "📝 步骤3/3: 生成评估摘要"

# 使用Python提取关键信息
python - <<EOF
import json
import sys
from pathlib import Path

try:
    # 读取评估结果
    with open('$OUTPUT_DIR/evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    
    print("\n" + "="*60)
    print("🎯 评估摘要")
    print("="*60)
    
    # 提取关键指标
    psnr = eval_results['psnr']['mean']
    ssim = eval_results['ssim']['mean']
    lpips = eval_results['lpips']['mean']
    
    print(f"\n📊 关键指标:")
    print(f"  PSNR:  {psnr:.2f} dB")
    print(f"  SSIM:  {ssim:.4f}")
    print(f"  LPIPS: {lpips:.4f}")
    
    # 计算综合评分
    psnr_score = min(psnr / 35.0, 1.0) * 100
    ssim_score = ssim * 100
    lpips_score = max(1.0 - lpips * 10, 0.0) * 100
    overall_score = (psnr_score * 0.3 + ssim_score * 0.4 + lpips_score * 0.3)
    
    print(f"\n🏆 综合评分: {overall_score:.1f}/100")
    
    # 质量评级
    if overall_score >= 85:
        quality = "🌟 Excellent (优秀)"
        color = '\033[0;32m'  # Green
    elif overall_score >= 70:
        quality = "✅ Good (良好)"
        color = '\033[0;32m'  # Green
    elif overall_score >= 55:
        quality = "⚠️  Fair (一般)"
        color = '\033[1;33m'  # Yellow
    else:
        quality = "❌ Poor (较差)"
        color = '\033[0;31m'  # Red
    
    print(f"{color}  质量等级: {quality}\033[0m")
    
    # 读取训练分析（如果存在）
    training_analysis_path = Path('$OUTPUT_DIR/training_analysis.json')
    if training_analysis_path.exists():
        with open(training_analysis_path, 'r') as f:
            training_analysis = json.load(f)
        
        print("\n📈 训练状态:")
        
        # 查找主要loss指标
        loss_metrics = [k for k in training_analysis['metrics'].keys() if 'loss' in k.lower()]
        if loss_metrics:
            main_loss = loss_metrics[0]
            loss_info = training_analysis['metrics'][main_loss]
            
            print(f"  趋势: {loss_info['trend']}")
            print(f"  收敛: {'是' if loss_info['is_converged'] else '否'}")
            print(f"  最终值: {loss_info['final_value']:.6f}")
        
        # 显示建议
        if training_analysis['recommendations']:
            print("\n💡 建议:")
            for rec in training_analysis['recommendations'][:3]:  # 最多显示3条
                print(f"  • {rec}")
    
    print("\n" + "="*60)
    print("✅ 评估完成！")
    print("="*60)
    
except Exception as e:
    print(f"生成摘要时出错: {e}", file=sys.stderr)
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    print_warning "无法生成评估摘要，但评估结果已保存"
fi

echo ""

# ============================================
# 完成
# ============================================
print_success "所有评估步骤完成！"
echo ""
echo "📂 结果文件位置:"
echo "  - $OUTPUT_DIR/"
echo ""
echo "📖 查看详细指南:"
echo "  cat EVALUATION_GUIDE.md"
echo ""
echo "🎨 可视化结果:"
if command -v xdg-open &> /dev/null; then
    echo "  xdg-open $OUTPUT_DIR/sample_1.png"
elif command -v open &> /dev/null; then
    echo "  open $OUTPUT_DIR/sample_1.png"
else
    echo "  查看 $OUTPUT_DIR/sample_*.png"
fi
echo ""
echo "📊 TensorBoard (如需要):"
echo "  tensorboard --logdir $LOG_DIR --port 6006"
echo ""

