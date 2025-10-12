"""
Training Logs Analysis Script
分析TensorBoard训练日志，评估训练过程质量

评估内容：
1. 损失曲线趋势
2. 收敛性分析
3. 过拟合检测
4. 最佳checkpoint识别
"""

import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats
import argparse
import json


class TrainingAnalyzer:
    """训练日志分析器"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.metrics_data = {}
        
    def load_tensorboard_logs(self):
        """加载TensorBoard日志"""
        print(f"📂 Loading logs from: {self.log_dir}")
        
        # 查找所有event文件
        event_files = list(self.log_dir.glob('events.out.tfevents.*'))
        
        if not event_files:
            print(f"❌ No event files found in {self.log_dir}")
            return False
        
        print(f"✅ Found {len(event_files)} event files")
        
        # 合并所有event文件的数据
        all_scalars = {}
        
        for event_file in event_files:
            try:
                ea = event_accumulator.EventAccumulator(str(event_file))
                ea.Reload()
                
                # 获取所有标量指标
                scalar_tags = ea.Tags()['scalars']
                
                for tag in scalar_tags:
                    events = ea.Scalars(tag)
                    
                    if tag not in all_scalars:
                        all_scalars[tag] = {'steps': [], 'values': [], 'wall_time': []}
                    
                    for event in events:
                        all_scalars[tag]['steps'].append(event.step)
                        all_scalars[tag]['values'].append(event.value)
                        all_scalars[tag]['wall_time'].append(event.wall_time)
                
                print(f"   Loaded: {event_file.name}")
                
            except Exception as e:
                print(f"⚠️  Error loading {event_file.name}: {e}")
                continue
        
        # 转换为numpy数组并排序
        for tag in all_scalars:
            # 按step排序
            sorted_indices = np.argsort(all_scalars[tag]['steps'])
            all_scalars[tag]['steps'] = np.array(all_scalars[tag]['steps'])[sorted_indices]
            all_scalars[tag]['values'] = np.array(all_scalars[tag]['values'])[sorted_indices]
            all_scalars[tag]['wall_time'] = np.array(all_scalars[tag]['wall_time'])[sorted_indices]
        
        self.metrics_data = all_scalars
        
        print(f"\n✅ Loaded metrics: {list(all_scalars.keys())}")
        
        return True
    
    def analyze_convergence(self, metric_name, window_size=100):
        """分析收敛性"""
        if metric_name not in self.metrics_data:
            return None
        
        values = self.metrics_data[metric_name]['values']
        
        # 计算移动平均
        moving_avg = pd.Series(values).rolling(window=window_size, min_periods=1).mean()
        
        # 计算趋势（线性回归斜率）
        if len(values) > window_size:
            recent_values = values[-window_size:]
            x = np.arange(len(recent_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
            
            is_converged = abs(slope) < 0.0001  # 斜率接近0
            trend = 'decreasing' if slope < 0 else 'increasing' if slope > 0 else 'stable'
        else:
            slope = 0
            is_converged = False
            trend = 'insufficient_data'
        
        # 计算变异系数 (标准差/均值)
        recent_std = np.std(values[-window_size:]) if len(values) > window_size else np.std(values)
        recent_mean = np.mean(values[-window_size:]) if len(values) > window_size else np.mean(values)
        cv = recent_std / (recent_mean + 1e-8)
        
        return {
            'moving_average': moving_avg.values.tolist(),  # 转换为列表以便JSON序列化
            'trend': trend,
            'slope': float(slope),
            'is_converged': bool(is_converged),  # 转换为Python bool
            'coefficient_of_variation': float(cv),
            'final_value': float(values[-1]),
            'best_value': float(np.min(values)) if 'loss' in metric_name.lower() else float(np.max(values)),
            'mean_value': float(np.mean(values)),
            'std_value': float(np.std(values)),
        }
    
    def detect_overfitting(self, train_metric, val_metric):
        """检测过拟合"""
        if train_metric not in self.metrics_data or val_metric not in self.metrics_data:
            print(f"⚠️  Cannot detect overfitting: missing {train_metric} or {val_metric}")
            return None
        
        train_values = self.metrics_data[train_metric]['values']
        val_values = self.metrics_data[val_metric]['values']
        
        # 确保长度一致（取较短的）
        min_len = min(len(train_values), len(val_values))
        train_values = train_values[:min_len]
        val_values = val_values[:min_len]
        
        # 计算训练集和验证集的差距
        gap = np.abs(train_values - val_values)
        
        # 检测gap是否在增大
        if len(gap) > 100:
            early_gap = np.mean(gap[:len(gap)//2])
            late_gap = np.mean(gap[len(gap)//2:])
            gap_increase = late_gap - early_gap
            
            is_overfitting = gap_increase > early_gap * 0.2  # gap增加超过20%
        else:
            is_overfitting = False
            gap_increase = 0
        
        return {
            'is_overfitting': is_overfitting,
            'gap_increase': float(gap_increase),
            'final_gap': float(gap[-1]),
            'mean_gap': float(np.mean(gap)),
        }
    
    def find_best_checkpoint(self, metric_name, mode='min'):
        """找到最佳checkpoint"""
        if metric_name not in self.metrics_data:
            return None
        
        values = self.metrics_data[metric_name]['values']
        steps = self.metrics_data[metric_name]['steps']
        
        if mode == 'min':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        return {
            'best_step': int(steps[best_idx]),
            'best_value': float(values[best_idx]),
            'metric': metric_name,
        }
    
    def plot_training_curves(self, save_dir='evaluation_results'):
        """绘制训练曲线"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 分组绘制不同类型的指标
        loss_metrics = [k for k in self.metrics_data.keys() if 'loss' in k.lower()]
        other_metrics = [k for k in self.metrics_data.keys() if 'loss' not in k.lower()]
        
        # 1. 绘制Loss曲线
        if loss_metrics:
            n_loss = len(loss_metrics)
            fig, axes = plt.subplots((n_loss + 1) // 2, 2, figsize=(15, 5 * ((n_loss + 1) // 2)))
            if n_loss == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for idx, metric in enumerate(loss_metrics):
                steps = self.metrics_data[metric]['steps']
                values = self.metrics_data[metric]['values']
                
                axes[idx].plot(steps, values, linewidth=1.5, alpha=0.8, label=metric)
                
                # 添加移动平均
                analysis = self.analyze_convergence(metric)
                if analysis:
                    axes[idx].plot(steps, analysis['moving_average'], 
                                 linewidth=2, alpha=0.6, linestyle='--', 
                                 label=f'Moving Avg (trend: {analysis["trend"]})')
                
                axes[idx].set_xlabel('Training Steps')
                axes[idx].set_ylabel('Loss')
                axes[idx].set_title(metric)
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for idx in range(len(loss_metrics), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'training_losses.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 Loss curves saved to: {save_dir / 'training_losses.png'}")
        
        # 2. 绘制其他指标
        if other_metrics:
            n_metrics = len(other_metrics)
            fig, axes = plt.subplots((n_metrics + 1) // 2, 2, figsize=(15, 5 * ((n_metrics + 1) // 2)))
            if n_metrics == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            for idx, metric in enumerate(other_metrics):
                steps = self.metrics_data[metric]['steps']
                values = self.metrics_data[metric]['values']
                
                axes[idx].plot(steps, values, linewidth=1.5, alpha=0.8)
                axes[idx].set_xlabel('Training Steps')
                axes[idx].set_ylabel('Value')
                axes[idx].set_title(metric)
                axes[idx].grid(True, alpha=0.3)
            
            # 隐藏多余的子图
            for idx in range(len(other_metrics), len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'training_metrics.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📊 Metric curves saved to: {save_dir / 'training_metrics.png'}")
    
    def generate_report(self, save_path='evaluation_results/training_analysis.json'):
        """生成分析报告"""
        print("\n" + "="*60)
        print("📊 Training Analysis Report")
        print("="*60)
        
        report = {
            'metrics': {},
            'convergence': {},
            'best_checkpoints': {},
            'recommendations': []
        }
        
        # 分析每个指标
        for metric_name in self.metrics_data.keys():
            analysis = self.analyze_convergence(metric_name)
            
            if analysis:
                report['metrics'][metric_name] = analysis
                
                print(f"\n📈 {metric_name}")
                print(f"   Trend: {analysis['trend']}")
                print(f"   Converged: {'Yes' if analysis['is_converged'] else 'No'}")
                print(f"   Final Value: {analysis['final_value']:.6f}")
                print(f"   Best Value: {analysis['best_value']:.6f}")
                print(f"   Mean ± Std: {analysis['mean_value']:.6f} ± {analysis['std_value']:.6f}")
                print(f"   Variation Coef: {analysis['coefficient_of_variation']:.4f}")
        
        # 找到最佳checkpoint
        loss_metrics = [k for k in self.metrics_data.keys() if 'loss' in k.lower() and 'train' in k.lower()]
        
        if loss_metrics:
            main_loss = loss_metrics[0]  # 使用第一个训练loss作为主要指标
            best_ckpt = self.find_best_checkpoint(main_loss, mode='min')
            
            if best_ckpt:
                report['best_checkpoints']['by_loss'] = best_ckpt
                print(f"\n🏆 Best Checkpoint (by {main_loss})")
                print(f"   Step: {best_ckpt['best_step']}")
                print(f"   Value: {best_ckpt['best_value']:.6f}")
        
        # 生成建议
        recommendations = self._generate_recommendations(report)
        report['recommendations'] = recommendations
        
        print("\n" + "="*60)
        print("💡 Recommendations")
        print("="*60)
        for rec in recommendations:
            print(f"   • {rec}")
        
        # 保存报告
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to: {save_path}")
        print("="*60)
        
        return report
    
    def _generate_recommendations(self, report):
        """生成训练建议"""
        recommendations = []
        
        # 检查是否收敛
        loss_metrics = [k for k in report['metrics'].keys() if 'loss' in k.lower()]
        
        if loss_metrics:
            main_loss = loss_metrics[0]
            analysis = report['metrics'][main_loss]
            
            if not analysis['is_converged']:
                if analysis['trend'] == 'decreasing':
                    recommendations.append(
                        "Loss is still decreasing. Consider training for more steps."
                    )
                elif analysis['trend'] == 'increasing':
                    recommendations.append(
                        "⚠️  Loss is increasing! Check for training instability or overfitting."
                    )
            else:
                recommendations.append(
                    "✅ Training has converged. Model is ready for evaluation."
                )
            
            # 检查变异系数
            if analysis['coefficient_of_variation'] > 0.1:
                recommendations.append(
                    "Training loss has high variation. Consider reducing learning rate or using gradient clipping."
                )
        
        # 检查是否有validation指标
        val_metrics = [k for k in report['metrics'].keys() if 'val' in k.lower() or 'validation' in k.lower()]
        
        if not val_metrics:
            recommendations.append(
                "⚠️  No validation metrics found. Add validation to monitor overfitting."
            )
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Analyze Training Logs')
    parser.add_argument('--log_dir', type=str, required=True, 
                       help='Path to TensorBoard log directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = TrainingAnalyzer(args.log_dir)
    
    # 加载日志
    if not analyzer.load_tensorboard_logs():
        print("❌ Failed to load logs")
        return
    
    # 绘制训练曲线
    analyzer.plot_training_curves(save_dir=args.output_dir)
    
    # 生成报告
    analyzer.generate_report(save_path=f'{args.output_dir}/training_analysis.json')


if __name__ == '__main__':
    main()

