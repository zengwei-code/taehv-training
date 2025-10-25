#!/bin/bash
#
# 一键诊断数据范围异常问题
#
# Usage:
#   bash scripts/diagnose_data_range.sh
#

set -e

# 配置
ANNOTATION_FILE="/data/matrix-project/MiniDataset/stage1_annotations_500.json"
DATA_ROOT="/data/matrix-project/MiniDataset/data"
NUM_SAMPLES=5
OUTPUT_DIR="diagnosis_results"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   🔍 数据范围异常诊断工具                                    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 创建输出目录
mkdir -p ${OUTPUT_DIR}
echo -e "${GREEN}✅ 创建输出目录: ${OUTPUT_DIR}${NC}\n"

# Step 1: 检查原始视频
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 1: 检查原始视频文件${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

python scripts/check_raw_videos.py \
    --annotation_file ${ANNOTATION_FILE} \
    --data_root ${DATA_ROOT} \
    --num_samples ${NUM_SAMPLES} \
    --output ${OUTPUT_DIR}/raw_videos_check.json

STEP1_EXIT=$?

if [ $STEP1_EXIT -eq 0 ]; then
    echo -e "\n${GREEN}✅ Step 1 完成${NC}"
else
    echo -e "\n${RED}❌ Step 1 失败，退出码: ${STEP1_EXIT}${NC}"
    exit $STEP1_EXIT
fi

# Step 2: 检查Dataset输出
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Step 2: 检查MiniDataset输出${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

python scripts/check_dataset_output.py \
    --annotation_file ${ANNOTATION_FILE} \
    --data_root ${DATA_ROOT} \
    --num_samples ${NUM_SAMPLES} \
    --height 480 \
    --width 720 \
    --n_frames 12 \
    --output ${OUTPUT_DIR}/dataset_output_check.json

STEP2_EXIT=$?

if [ $STEP2_EXIT -eq 0 ]; then
    echo -e "\n${GREEN}✅ Step 2 完成${NC}"
else
    echo -e "\n${RED}❌ Step 2 失败，退出码: ${STEP2_EXIT}${NC}"
    exit $STEP2_EXIT
fi

# 生成综合报告
echo -e "\n${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}📊 生成综合诊断报告${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}\n"

REPORT_FILE="${OUTPUT_DIR}/diagnosis_report.txt"

cat > ${REPORT_FILE} << EOF
数据范围异常诊断报告
==================
生成时间: $(date '+%Y-%m-%d %H:%M:%S')

配置信息
--------
Annotation文件: ${ANNOTATION_FILE}
数据目录: ${DATA_ROOT}
检查样本数: ${NUM_SAMPLES}

诊断步骤
--------
1. 原始视频检查: ✅ 完成
   结果文件: ${OUTPUT_DIR}/raw_videos_check.json

2. Dataset输出检查: ✅ 完成
   结果文件: ${OUTPUT_DIR}/dataset_output_check.json

详细结果
--------
请查看以下JSON文件了解详细结果:
- ${OUTPUT_DIR}/raw_videos_check.json
- ${OUTPUT_DIR}/dataset_output_check.json

下一步
------
根据诊断结果:

1. 如果原始视频正常，但Dataset输出异常:
   → 检查 training/data/video_dataset.py 的实现
   → 可能是MiniDataset的bug

2. 如果Dataset输出是 [0, 255] uint8:
   → 检查训练脚本是否正确归一化
   → 查看 training/taehv_train.py 中的 batch.float() / 255.0

3. 如果Dataset输出是 [0, 1] float:
   → 训练脚本可能重复归一化！
   → 修改为 batch.float() (删除 / 255.0)

4. 如果Dataset输出几乎全是0:
   → 视频读取失败或数据集有问题
   → 检查视频文件和MiniDataset实现

参考文档
--------
docs/数据范围异常诊断指南.md

EOF

cat ${REPORT_FILE}

echo -e "\n${GREEN}✅ 诊断报告已生成: ${REPORT_FILE}${NC}"

# 总结
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   ✅ 诊断完成！                                               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}\n"

echo -e "${GREEN}诊断结果已保存到: ${OUTPUT_DIR}/${NC}"
echo -e "${GREEN}请查看诊断报告: ${REPORT_FILE}${NC}\n"

echo -e "${YELLOW}下一步建议:${NC}"
echo -e "  1. 查看 ${OUTPUT_DIR}/raw_videos_check.json"
echo -e "  2. 查看 ${OUTPUT_DIR}/dataset_output_check.json"
echo -e "  3. 根据结果修复问题"
echo -e "  4. 参考文档: docs/数据范围异常诊断指南.md\n"


