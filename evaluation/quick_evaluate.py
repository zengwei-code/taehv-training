"""
Quick Evaluation Script - Python Version
快速评估脚本 - Python版本

一键运行完整评估流程，包括：
1. 训练日志分析
2. 模型定量评估
3. 生成评估报告

使用方法:
    python quick_evaluate.py
    python quick_evaluate.py --model_path output/xxx/final_model.pth --num_samples 200
"""

import subprocess
import argparse
import json
from pathlib import Path
import sys


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


def analyze_training_logs(log_dir, output_dir):
    """分析训练日志"""
    print_header("步骤1/3: 分析训练日志")
    
    if not Path(log_dir).exists():
        print_warning(f"日志目录不存在: {log_dir}，跳过")
        return False
    
    cmd = [
        "python", "analyze_training_logs.py",
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


def evaluate_model(model_path, config, num_samples, output_dir, data_root=None, annotation_file=None):
    """评估模型"""
    print_header("步骤2/3: 模型定量评估")
    
    if not Path(model_path).exists():
        print_error(f"模型文件不存在: {model_path}")
        return False
    
    if not Path(config).exists():
        print_error(f"配置文件不存在: {config}")
        return False
    
    cmd = [
        "python", "evaluate_vae.py",
        "--model_path", model_path,
        "--config", config,
        "--num_samples", str(num_samples),
        "--batch_size", "4"
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
    parser = argparse.ArgumentParser(
        description='快速评估VAE模型 - 一键运行完整评估流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quick_evaluate.py
  python quick_evaluate.py --model_path output/xxx/final_model.pth
  python quick_evaluate.py --num_samples 200
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='../output/2025-10-01_19-59-50/final_model.pth',
        help='模型checkpoint路径 (默认: ../output/2025-10-01_19-59-50/final_model.pth)'
    )
    
    parser.add_argument(
        '--log_dir',
        type=str,
        default='../logs/taehv_h100_production',
        help='训练日志目录 (默认: ../logs/taehv_h100_production)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='../training/configs/taehv_config_h100.py',
        help='配置文件路径 (默认: ../training/configs/taehv_config_h100.py)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=100,
        help='评估样本数量 (默认: 100)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./evaluation_results',
        help='结果输出目录 (默认: ./evaluation_results)'
    )
    
    parser.add_argument(
        '--skip_logs',
        action='store_true',
        help='跳过训练日志分析'
    )
    
    parser.add_argument(
        '--data_root',
        type=str,
        help='覆盖配置文件中的数据集根目录'
    )
    
    parser.add_argument(
        '--annotation_file',
        type=str,
        help='覆盖配置文件中的annotation文件路径'
    )
    
    args = parser.parse_args()
    
    # 打印配置
    print_header("🚀 快速评估 Tiny-VAE 模型")
    print(f"模型路径: {args.model_path}")
    print(f"日志目录: {args.log_dir}")
    print(f"配置文件: {args.config}")
    print(f"评估样本数: {args.num_samples}")
    print(f"结果目录: {args.output_dir}")
    if args.data_root:
        print(f"数据集根目录: {args.data_root} (覆盖)")
    if args.annotation_file:
        print(f"Annotation文件: {args.annotation_file} (覆盖)")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 步骤1: 分析训练日志
    if not args.skip_logs:
        analyze_training_logs(args.log_dir, args.output_dir)
    else:
        print_warning("跳过训练日志分析（--skip_logs）")
    
    # 步骤2: 评估模型
    if not evaluate_model(args.model_path, args.config, args.num_samples, args.output_dir,
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

