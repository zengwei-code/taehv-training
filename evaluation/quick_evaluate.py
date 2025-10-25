"""
Quick Evaluation Script - Python Version
快速评估脚本 - Python版本

一键运行完整评估流程，包括：
1. 训练日志分析
2. 模型定量评估
3. 生成评估报告

使用方法:
    python quick_evaluate.py --model_path xxx --data_root xxx --annotation_file xxx
"""

import subprocess
import argparse
import json
import os
from pathlib import Path
import sys

# 修复GLIBCXX版本问题（matplotlib依赖）
os.environ['LD_LIBRARY_PATH'] = '/data1/anaconda3/envs/tiny-vae/lib:' + os.environ.get('LD_LIBRARY_PATH', '')


def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"🎯 {text}")
    print("=" * 60)


def print_success(text):
    """打印成功消息"""
    print(f"✅ {text}")


def print_warning(text):
    """打印警告消息"""
    print(f"⚠️  {text}")


def print_error(text):
    """打印错误消息"""
    print(f"❌ {text}")


def detect_log_dir(project_root):
    """自动检测日志目录"""
    logs_dir = project_root / 'logs'
    if not logs_dir.exists():
        return None
    
    # 查找包含最多事件文件的日志目录
    log_candidates = []
    for subdir in logs_dir.iterdir():
        if subdir.is_dir():
            event_files = list(subdir.glob('events.out.tfevents.*'))
            if event_files:
                log_candidates.append((subdir, len(event_files)))
    
    if log_candidates:
        # 返回包含最多事件文件的目录
        log_candidates.sort(key=lambda x: x[1], reverse=True)
        return str(log_candidates[0][0])
    
    return None


def validate_paths(args, project_root):
    """验证和修正路径参数"""
    issues = []
    suggestions = []
    
    # 检查模型路径
    if not Path(args.model_path).exists():
        issues.append(f"模型文件不存在: {args.model_path}")
        
        # 尝试在output目录中查找最新的模型
        output_dir = project_root / 'output'
        if output_dir.exists():
            model_candidates = []
            for subdir in sorted(output_dir.iterdir(), reverse=True):
                if subdir.is_dir():
                    best_model = subdir / 'best_model' / 'model.pth'
                    final_model = subdir / 'final_model.pth'
                    if best_model.exists():
                        model_candidates.append(str(best_model))
                        break
                    elif final_model.exists():
                        model_candidates.append(str(final_model))
                        break
            
            if model_candidates:
                suggestions.append(f"建议使用: --model_path {model_candidates[0]}")
    
    # 检查配置文件
    if not Path(args.config).exists():
        issues.append(f"配置文件不存在: {args.config}")
        
        # 尝试查找其他配置文件
        config_dir = project_root / 'training' / 'configs'
        if config_dir.exists():
            config_files = list(config_dir.glob('taehv_config_*.py'))
            if config_files:
                suggestions.append(f"建议使用: --config {config_files[0]}")
    
    # 检查数据集路径（最重要）
    if not args.data_root or not Path(args.data_root).exists():
        issues.append("数据集根目录未指定或不存在")
        suggestions.append("必须使用: --data_root /path/to/dataset")
        suggestions.append("示例: --data_root /data/matrix-project/MiniDataset/data")
    
    if not args.annotation_file or not Path(args.annotation_file).exists():
        issues.append("标注文件未指定或不存在")
        suggestions.append("必须使用: --annotation_file /path/to/annotations.json")
        suggestions.append("示例: --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json")
    
    # 检查日志目录（可选）
    if args.log_dir and not Path(args.log_dir).exists():
        detected_log = detect_log_dir(project_root)
        if detected_log:
            issues.append(f"指定的日志目录不存在: {args.log_dir}")
            suggestions.append(f"检测到日志目录: --log_dir {detected_log}")
            args.log_dir = detected_log  # 自动修正
        else:
            issues.append("未找到训练日志目录，将跳过日志分析")
    
    return issues, suggestions


def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n⏳ {description}...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description}失败")
        return False
    except FileNotFoundError:
        print_error(f"命令未找到: {cmd[0]}")
        return False


def analyze_training_logs(log_dir, output_dir, script_dir):
    """分析训练日志"""
    print_header("步骤1/3: 分析训练日志")
    
    if not Path(log_dir).exists():
        print_warning(f"日志目录不存在: {log_dir}，跳过")
        return False
    
    # 使用绝对路径调用脚本
    analyze_script = script_dir / "analyze_training_logs.py"
    
    cmd = [
        "python", str(analyze_script),
        "--log_dir", log_dir,
        "--output_dir", output_dir
    ]
    
    success = run_command(cmd, "训练日志分析")
    
    if success:
        print_success("训练日志分析完成")
        print(f"  - 损失曲线: {output_dir}/training_losses.png")
        print(f"  - 指标曲线: {output_dir}/training_metrics.png")
        print(f"  - 分析报告: {output_dir}/training_analysis.json")
    
    return success


def evaluate_model(model_path, config, num_samples, batch_size, output_dir, script_dir, data_root=None, annotation_file=None):
    """评估模型"""
    print_header("步骤2/3: 模型定量评估")
    
    if not Path(model_path).exists():
        print_error(f"模型文件不存在: {model_path}")
        return False
    
    if not Path(config).exists():
        print_error(f"配置文件不存在: {config}")
        return False
    
    # 使用绝对路径调用脚本
    evaluate_script = script_dir / "evaluate_vae.py"
    
    cmd = [
        "python", str(evaluate_script),
        "--model_path", model_path,
        "--config", config,
        "--num_samples", str(num_samples),
        "--batch_size", str(batch_size)
    ]
    
    # 添加可选的数据集参数
    if data_root:
        cmd.extend(["--data_root", data_root])
    if annotation_file:
        cmd.extend(["--annotation_file", annotation_file])
    
    success = run_command(cmd, "模型评估")
    
    if success:
        print_success("模型评估完成")
        print(f"  - 评估结果: {output_dir}/evaluation_results.json")
        print(f"  - 指标分布: {output_dir}/metrics_distribution.png")
        print(f"  - 可视化样本: {output_dir}/sample_*.png")
    
    return success


def generate_summary(output_dir):
    """生成评估摘要"""
    print_header("步骤3/3: 生成评估摘要")
    
    try:
        # 读取评估结果
        eval_results_path = Path(output_dir) / "evaluation_results.json"
        if not eval_results_path.exists():
            print_warning("评估结果文件不存在")
            return False
        
        with open(eval_results_path, 'r') as f:
            eval_results = json.load(f)
        
        print("\n" + "=" * 60)
        print("🎯 评估摘要")
        print("=" * 60)
        
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
        elif overall_score >= 70:
            quality = "✅ Good (良好)"
        elif overall_score >= 55:
            quality = "⚠️  Fair (一般)"
        else:
            quality = "❌ Poor (较差)"
        
        print(f"  质量等级: {quality}")
        
        # 读取训练分析（如果存在）
        training_analysis_path = Path(output_dir) / "training_analysis.json"
        if training_analysis_path.exists():
            with open(training_analysis_path, 'r') as f:
                training_analysis = json.load(f)
            
            print("\n📈 训练状态:")
            
            # 查找主要loss指标
            loss_metrics = [k for k in training_analysis['metrics'].keys() 
                          if 'loss' in k.lower()]
            if loss_metrics:
                main_loss = loss_metrics[0]
                loss_info = training_analysis['metrics'][main_loss]
                
                print(f"  趋势: {loss_info['trend']}")
                print(f"  收敛: {'是' if loss_info['is_converged'] else '否'}")
                print(f"  最终值: {loss_info['final_value']:.6f}")
            
            # 显示建议
            if training_analysis.get('recommendations'):
                print("\n💡 建议:")
                for rec in training_analysis['recommendations'][:3]:
                    print(f"  • {rec}")
        
        print("\n" + "=" * 60)
        
        # 生成Markdown报告
        report_path = Path(output_dir) / "EVALUATION_REPORT.md"
        generate_markdown_report(eval_results, training_analysis if training_analysis_path.exists() else None, 
                                report_path, overall_score, quality)
        print_success(f"详细报告已保存: {report_path}")
        
        return True
        
    except Exception as e:
        print_error(f"生成摘要时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_markdown_report(eval_results, training_analysis, save_path, overall_score, quality):
    """生成Markdown格式的评估报告"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# 📊 VAE模型评估报告\n\n")
        f.write(f"**生成时间**: {Path.cwd()}\n\n")
        
        f.write("## 🎯 综合评估\n\n")
        f.write(f"- **综合评分**: {overall_score:.1f}/100\n")
        f.write(f"- **质量等级**: {quality}\n\n")
        
        f.write("## 📊 定量指标\n\n")
        f.write("| 指标 | 均值 | 标准差 | 最小值 | 最大值 |\n")
        f.write("|------|------|--------|--------|--------|\n")
        
        for metric_name, stats in eval_results.items():
            f.write(f"| {metric_name.upper()} | {stats['mean']:.4f} | "
                   f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |\n")
        
        f.write("\n### 📈 指标解读\n\n")
        
        psnr = eval_results['psnr']['mean']
        ssim = eval_results['ssim']['mean']
        lpips = eval_results['lpips']['mean']
        
        f.write(f"- **PSNR = {psnr:.2f} dB**\n")
        if psnr > 35:
            f.write("  - ✅ 优秀 - 重建质量非常高\n")
        elif psnr > 30:
            f.write("  - ✅ 良好 - 重建质量可接受\n")
        elif psnr > 25:
            f.write("  - ⚠️  一般 - 有明显失真\n")
        else:
            f.write("  - ❌ 较差 - 失真严重\n")
        
        f.write(f"\n- **SSIM = {ssim:.4f}**\n")
        if ssim > 0.95:
            f.write("  - ✅ 优秀 - 结构保留完美\n")
        elif ssim > 0.90:
            f.write("  - ✅ 良好 - 结构保留较好\n")
        elif ssim > 0.85:
            f.write("  - ⚠️  一般 - 有结构损失\n")
        else:
            f.write("  - ❌ 较差 - 结构变化明显\n")
        
        f.write(f"\n- **LPIPS = {lpips:.4f}**\n")
        if lpips < 0.05:
            f.write("  - ✅ 优秀 - 感知质量极高\n")
        elif lpips < 0.10:
            f.write("  - ✅ 良好 - 感知质量较好\n")
        elif lpips < 0.20:
            f.write("  - ⚠️  一般 - 感知差异可察觉\n")
        else:
            f.write("  - ❌ 较差 - 感知质量不佳\n")
        
        # 训练分析
        if training_analysis:
            f.write("\n## 📈 训练过程分析\n\n")
            
            loss_metrics = [k for k in training_analysis['metrics'].keys() 
                          if 'loss' in k.lower()]
            
            if loss_metrics:
                f.write("### Loss指标\n\n")
                for loss_name in loss_metrics:
                    loss_info = training_analysis['metrics'][loss_name]
                    f.write(f"**{loss_name}**:\n")
                    f.write(f"- 趋势: {loss_info['trend']}\n")
                    f.write(f"- 是否收敛: {'是' if loss_info['is_converged'] else '否'}\n")
                    f.write(f"- 最终值: {loss_info['final_value']:.6f}\n")
                    f.write(f"- 最佳值: {loss_info['best_value']:.6f}\n\n")
            
            if training_analysis.get('recommendations'):
                f.write("### 💡 建议\n\n")
                for rec in training_analysis['recommendations']:
                    f.write(f"- {rec}\n")
        
        f.write("\n## 📁 文件清单\n\n")
        f.write("评估过程生成了以下文件：\n\n")
        f.write("- `evaluation_results.json` - 详细评估数据\n")
        f.write("- `metrics_distribution.png` - 指标分布图\n")
        f.write("- `sample_*.png` - 可视化重建样本\n")
        f.write("- `training_losses.png` - 训练损失曲线\n")
        f.write("- `training_metrics.png` - 训练指标曲线\n")
        f.write("- `training_analysis.json` - 训练过程分析\n")
        
        f.write("\n---\n\n")
        f.write("*报告由 `quick_evaluate.py` 自动生成*\n")


def main():
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = Path(__file__).parent  # evaluation/
    project_root = script_dir.parent    # my_taehv_training/
    
    # 智能检测默认路径
    detected_log_dir = detect_log_dir(project_root)
    default_log_dir = detected_log_dir or str(project_root / 'logs')
    
    # 查找最新的模型
    default_model = None
    output_dir = project_root / 'output'
    if output_dir.exists():
        for subdir in sorted(output_dir.iterdir(), reverse=True):
            if subdir.is_dir():
                best_model = subdir / 'best_model' / 'model.pth'
                if best_model.exists():
                    default_model = str(best_model)
                    break
    
    parser = argparse.ArgumentParser(
        description='快速评估VAE模型 - 一键运行完整评估流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
完整示例:
  python evaluation/quick_evaluate.py \\
      --model_path output/a800_2025-10-14_12-11-50/best_model/model.pth \\
      --data_root /data/matrix-project/MiniDataset/data \\
      --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \\
      --num_samples 100

快速示例（自动检测模型）:
  python evaluation/quick_evaluate.py \\
      --data_root /data/matrix-project/MiniDataset/data \\
      --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json
        """
    )
    
    # 必需参数组
    required_group = parser.add_argument_group('必需参数（推荐明确指定）')
    required_group.add_argument(
        '--data_root',
        type=str,
        required=False,
        help='⭐ 数据集根目录 [必需] - 示例: /data/matrix-project/MiniDataset/data'
    )
    
    required_group.add_argument(
        '--annotation_file',
        type=str,
        required=False,
        help='⭐ 标注文件路径 [必需] - 示例: /data/matrix-project/MiniDataset/stage1_annotations_500.json'
    )
    
    # 模型相关参数
    model_group = parser.add_argument_group('模型相关参数')
    model_group.add_argument(
        '--model_path',
        type=str,
        default=default_model,
        help=f'模型checkpoint路径 (默认: 自动检测最新模型)'
    )
    
    model_group.add_argument(
        '--config',
        type=str,
        default=str(project_root / 'training' / 'configs' / 'taehv_config_a800.py'),
        help='配置文件路径 (默认: training/configs/taehv_config_a800.py)'
    )
    
    # 评估参数
    eval_group = parser.add_argument_group('评估参数')
    eval_group.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='评估样本数量 (默认: 100)'
    )
    
    eval_group.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='批次大小 (默认: 4)'
    )
    
    # 日志和输出
    log_group = parser.add_argument_group('日志和输出')
    log_group.add_argument(
        '--log_dir',
        type=str,
        default=default_log_dir,
        help=f'训练日志目录 (默认: {default_log_dir})'
    )
    
    log_group.add_argument(
        '--output_dir',
        type=str,
        default=str(script_dir / 'evaluation_results'),
        help='结果输出目录 (默认: evaluation/evaluation_results)'
    )
    
    log_group.add_argument(
        '--skip_logs',
        action='store_true',
        help='跳过训练日志分析'
    )
    
    args = parser.parse_args()
    
    # 验证路径
    issues, suggestions = validate_paths(args, project_root)
    
    # 打印配置
    print_header("🚀 快速评估 Tiny-VAE 模型")
    print(f"模型路径: {args.model_path}")
    print(f"日志目录: {args.log_dir}")
    print(f"配置文件: {args.config}")
    print(f"评估样本数: {args.num_samples}")
    print(f"批次大小: {args.batch_size}")
    print(f"结果目录: {args.output_dir}")
    if args.data_root:
        print(f"数据集根目录: {args.data_root}")
    if args.annotation_file:
        print(f"Annotation文件: {args.annotation_file}")
    
    # 显示问题和建议
    if issues:
        print("\n" + "="*60)
        print("⚠️  发现以下问题:")
        print("="*60)
        for issue in issues:
            print(f"  • {issue}")
        
        if suggestions:
            print("\n💡 建议:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")
        
        # 检查是否是致命错误（数据集路径）
        if not args.data_root or not Path(args.data_root).exists() or \
           not args.annotation_file or not Path(args.annotation_file).exists():
            print("\n" + "="*60)
            print_error("数据集路径是必需的，无法继续评估")
            print("="*60)
            print("\n请使用以下命令运行:")
            print("  python evaluation/quick_evaluate.py \\")
            print("      --data_root /data/matrix-project/MiniDataset/data \\")
            print("      --annotation_file /data/matrix-project/MiniDataset/stage1_annotations_500.json \\")
            if default_model:
                print(f"      --model_path {default_model} \\")
            print("      --num_samples 100")
            print()
            sys.exit(1)
        
        print("\n将尝试继续执行...")
        print("="*60)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 步骤1: 分析训练日志
    if not args.skip_logs:
        analyze_training_logs(args.log_dir, args.output_dir, script_dir)
    else:
        print_warning("跳过训练日志分析（--skip_logs）")
    
    # 步骤2: 评估模型
    if not evaluate_model(args.model_path, args.config, args.num_samples, args.batch_size,
                         args.output_dir, script_dir,
                         data_root=args.data_root, annotation_file=args.annotation_file):
        print_error("模型评估失败，终止")
        sys.exit(1)
    
    # 步骤3: 生成摘要
    generate_summary(args.output_dir)
    
    # 完成
    print("\n" + "=" * 60)
    print_success("所有评估步骤完成！")
    print("=" * 60)
    print(f"\n📂 结果文件位置: {args.output_dir}/")
    print(f"\n📄 详细数据: {args.output_dir}/evaluation_results.json")
    print(f"\n📖 查看使用指南: cat README.md")
    print(f"📖 查看故障排除: cat TROUBLESHOOTING.md")
    print(f"\n🎨 查看可视化结果:")
    print(f"   - {args.output_dir}/sample_1.png")
    print(f"   - {args.output_dir}/metrics_distribution.png")
    print(f"   - {args.output_dir}/training_losses.png")
    
    print(f"\n📊 启动TensorBoard查看详细训练过程:")
    print(f"   tensorboard --logdir {args.log_dir} --port 6006")
    print()


if __name__ == '__main__':
    main()

