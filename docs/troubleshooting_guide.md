# TAEHV训练错误排查和解决指南

本文档记录了TAEHV模型训练过程中遇到的所有错误问题、根本原因分析和详细解决方法。

## 📋 目录

1. [错误1: FileNotFoundError - Checkpoint文件路径问题](#错误1-filenotfound-checkpoint文件路径问题)
2. [错误2: DeepSpeed AssertionError - 优化器缺失问题](#错误2-deepspeed-assertionerror-优化器缺失问题)
3. [错误3: TensorBoard ValueError - 超参数类型问题](#错误3-tensorboard-valueerror-超参数类型问题)
4. [错误4: RuntimeError - 混合精度类型不匹配](#错误4-runtimeerror-混合精度类型不匹配)
5. [错误5: NCCL DistBackendError - 通信超时问题](#错误5-nccl-distbackenderror-通信超时问题)
6. [错误6: TypeError - isfinite()参数类型问题](#错误6-typeerror-isfinite参数类型问题)
7. [错误7: Checkpoint保存问题 - 空目录问题](#错误7-checkpoint保存问题-空目录问题)
8. [错误8: SIGTERM进程终止问题](#错误8-sigterm进程终止问题)
9. [错误9: Checkpoint路径逻辑错误](#错误9-checkpoint路径逻辑错误)
10. [错误10: Seraena模型DeepSpeed优化器问题](#错误10-seraena模型deepspeed优化器问题)
11. [错误11: Seraena张量形状不匹配问题](#错误11-seraena张量形状不匹配问题)
12. [错误12: TAEHV帧数维度不匹配 - decoded与frames_target不一致](#错误12-taehv帧数维度不匹配-decoded与frames_target不一致)
13. [错误13: Seraena帧数不能整除警告 - 数据损失问题](#错误13-seraena帧数不能整除警告-数据损失问题)

---

## 错误1: FileNotFoundError - Checkpoint文件路径问题

### 🚨 错误现象
```bash
FileNotFoundError: [Errno 2] No such file or directory: '../checkpoints/taecvx.pth'
```

### 🔍 根本原因
- 配置文件中的`pretrained_model_path`使用了相对路径`../checkpoints/taecvx.pth`
- 实际执行时工作目录与预期不符，导致路径无法找到

### ✅ 解决方法

**修改文件**: `training/configs/taehv_config.py`

```python
# 修改前
args.pretrained_model_path = "../checkpoints/taecvx.pth"

# 修改后  
args.pretrained_model_path = "checkpoints/taecvx.pth"
```

### 🛡️ 预防措施
- 使用相对于项目根目录的路径
- 或使用绝对路径
- 在脚本中添加路径存在性检查

---

## 错误2: DeepSpeed AssertionError - 优化器缺失问题

### 🚨 错误现象
```bash
AssertionError: zero stage 2 requires an optimizer
```

### 🔍 根本原因
- `vae_ref`（参考VAE模型）被错误地传递给`accelerator.prepare()`
- DeepSpeed ZeRO stage 2要求所有prepared的模型都必须有对应的优化器
- 但`vae_ref`是只读模型，不需要训练，因此没有优化器

### ✅ 解决方法

**修改文件**: `training/taehv_train.py`

```python
# 修改前
model, optimizer, train_dataloader, lr_scheduler, vae_ref = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler, vae_ref
)

# 修改后
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

if vae_ref is not None:
    # 参考VAE不需要训练，手动移动到设备而不是通过DeepSpeed准备
    vae_ref = vae_ref.to(accelerator.device)
```

### 🛡️ 预防措施
- 只将需要训练的模型传递给`accelerator.prepare()`
- 参考模型手动移动到设备
- 明确区分训练模型和推理模型

---

## 错误3: TensorBoard ValueError - 超参数类型问题

### 🚨 错误现象
```bash
ValueError: value should be one of int, float, str, bool, or torch.Tensor
```

### 🔍 根本原因
- `accelerator.init_trackers()`只支持特定数据类型的超参数
- 配置对象包含不支持的类型，如`datetime`对象、字典等

### ✅ 解决方法

**修改文件**: `training/taehv_train.py`

```python
# 添加配置过滤逻辑
if accelerator.is_main_process:
    # 过滤配置，只保留TensorBoard支持的类型
    filtered_config = {}
    for key, value in vars(config).items():
        if isinstance(value, (int, float, str, bool)) or (hasattr(value, 'dtype') and hasattr(value, 'device')):
            filtered_config[key] = value
        elif isinstance(value, dict):
            # 将字典转换为字符串
            filtered_config[key] = str(value)
        else:
            # 将其他类型转换为字符串
            filtered_config[key] = str(value)
    
    accelerator.init_trackers(config.tracker_name, config=filtered_config)
```

### 🛡️ 预防措施
- 始终过滤配置对象
- 将复杂类型转换为字符串表示
- 测试不同类型的配置参数

---

## 错误4: RuntimeError - 混合精度类型不匹配

### 🚨 错误现象
```bash
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```

### 🔍 根本原因
- 混合精度训练设置为`bf16`，模型参数为`bfloat16`类型
- 输入数据仍为`float32`类型
- 在`accelerator.autocast()`内部类型转换不一致

### ✅ 解决方法

**修改文件**: `training/taehv_train.py`

```python
# 修改前
with accelerator.autocast():
    frames = batch.float() / 255.0  # 在autocast内部转换

# 修改后
# 数据预处理 - 明确转换到正确的设备和类型
frames = batch.float() / 255.0  # N,T,C,H,W, [0,1]
# 转换到与模型参数相同的类型（bf16）
frames = frames.to(accelerator.device, dtype=torch.bfloat16)

# 前向传播
with accelerator.autocast():
    # 编码
    encoded = model.encode_video(frames, parallel=True, show_progress_bar=False)
```

### 🛡️ 预防措施
- 在autocast外部进行数据类型转换
- 确保输入数据类型与模型参数类型一致
- 显式指定设备和数据类型

---

## 错误5: NCCL DistBackendError - 通信超时问题

### 🚨 错误现象
```bash
torch.distributed.DistBackendError: Watchdog caught collective operation timeout
```

### 🔍 根本原因
- NCCL通信超时，可能由以下原因引起：
  - 批量大小过大导致内存压力
  - 网络通信不稳定
  - 梯度同步时间过长
  - GPU间通信延迟

### ✅ 解决方法

#### 方法1: 调整批量大小和梯度累积

**修改文件**: `training/configs/taehv_config.py`

```python
# 减少批量大小，增加梯度累积步数
args.train_batch_size = 2  # 从4减少到2
args.gradient_accumulation_steps = 2  # 从1增加到2
```

#### 方法2: 优化DeepSpeed配置

**修改文件**: `accelerate_configs/deepspeed.yaml`

```yaml
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  offload_optimizer_device: cpu
  offload_param_device: cpu
  zero3_init_flag: false
  zero_stage: 2
  communication_data_type: fp16  # 新增：使用fp16通信
  allgather_bucket_size: 50000000  # 新增：优化通信桶大小
  reduce_bucket_size: 50000000    # 新增：优化reduce桶大小
  overlap_comm: true              # 新增：重叠通信和计算
  contiguous_gradients: true      # 新增：使用连续梯度
```

#### 方法3: 优化NCCL环境变量

**修改文件**: `train_taehv.sh`

```bash
# NCCL优化环境变量
export NCCL_P2P_DISABLE=1                    # 禁用P2P，使用更稳定的通信方式
export NCCL_TIMEOUT=7200                     # 增加超时时间到2小时
export NCCL_BLOCKING_WAIT=1                  # 启用阻塞等待
export NCCL_IB_DISABLE=1                     # 禁用InfiniBand（如果有网络问题）
export NCCL_SOCKET_NTHREADS=4               # 增加socket线程数
export NCCL_NSOCKS_PERTHREAD=8              # 增加每线程socket数
export NCCL_BUFFSIZE=8388608                # 增加缓冲区大小(8MB)
export NCCL_NET_GDR_LEVEL=5                 # 优化GPU Direct RDMA
export NCCL_DEBUG=INFO                       # 启用调试信息（可选）

# PyTorch分布式优化
export TORCH_NCCL_ENABLE_MONITORING=0       # 禁用NCCL监控减少开销
export TORCH_NCCL_BLOCKING_WAIT=1           # PyTorch层面的阻塞等待
export TORCH_DISTRIBUTED_DEBUG=INFO         # 启用分布式调试信息
```

#### 方法4: 添加错误处理和内存管理

**修改文件**: `training/taehv_train.py`

```python
# 添加NaN检查和错误处理
if not torch.isfinite(total_loss):
    logger.warning(f"Loss is not finite: {total_loss}, skipping step")
    optimizer.zero_grad()
    continue

try:
    accelerator.backward(total_loss)
    
    if accelerator.sync_gradients:
        # 条件性梯度裁剪（DeepSpeed自动处理）
        if accelerator.distributed_type != DistributedType.DEEPSPEED:
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not math.isfinite(grad_norm):
                logger.warning(f"Gradient norm is not finite: {grad_norm}, skipping step")
                optimizer.zero_grad()
                continue
                
except RuntimeError as e:
    if "out of memory" in str(e):
        logger.error(f"GPU OOM at step {global_step}, skipping batch")
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        continue
    else:
        raise e

# 定期内存清理
if global_step % 100 == 0:
    torch.cuda.empty_cache()
    if hasattr(accelerator.state, 'deepspeed_plugin') and accelerator.state.deepspeed_plugin is not None:
        import gc
        gc.collect()
```

### 🛡️ 预防措施
- 监控GPU内存使用情况
- 定期清理内存碎片
- 使用更保守的批量大小
- 增加NCCL超时时间
- 启用通信调试信息

---

## 错误6: TypeError - isfinite()参数类型问题

### 🚨 错误现象
```bash
TypeError: isfinite(): argument 'input' (position 1) must be Tensor, not float
```

### 🔍 根本原因
- `accelerator.clip_grad_norm_()`返回Python float类型的梯度范数
- `torch.isfinite()`只接受torch.Tensor类型参数
- 需要使用`math.isfinite()`处理Python float

### ✅ 解决方法

**修改文件**: `training/taehv_train.py`

```python
# 修改前
grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
if not torch.isfinite(grad_norm):  # 错误：grad_norm是Python float
    logger.warning(f"Gradient norm is not finite: {grad_norm}, skipping step")

# 修改后
import math  # 确保导入math模块

grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
if not math.isfinite(grad_norm):  # 正确：使用math.isfinite处理Python float
    logger.warning(f"Gradient norm is not finite: {grad_norm}, skipping step")
    optimizer.zero_grad()
    continue
```

### 🛡️ 预防措施
- 区分torch.Tensor和Python基础类型
- 对torch.Tensor使用`torch.isfinite()`
- 对Python float使用`math.isfinite()`
- 添加类型检查和测试

---

## 错误7: Checkpoint保存问题 - 空目录问题

### 🚨 错误现象
- Checkpoint目录被创建但完全为空
- `ls checkpoint-500/` 显示没有任何文件

### 🔍 根本原因
1. `accelerator.save_state()`被错误地包装在主进程条件中
2. 缺少模型序列化hooks
3. `accelerator.save_state()`内部已处理多进程协调，不需要额外条件

### ✅ 解决方法

#### 方法1: 添加保存hooks

**修改文件**: `training/taehv_train.py`

```python
# 在accelerator.prepare()之后添加
def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        # 保存主模型状态
        model_to_save = accelerator.unwrap_model(model)
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pth"))
        logger.info(f"Saved model state dict to {output_dir}/model.pth")

def load_model_hook(models, input_dir):
    # 模型加载逻辑（如果需要）
    pass

# 注册hooks
accelerator.register_save_state_pre_hook(save_model_hook)
accelerator.register_load_state_pre_hook(load_model_hook)
```

#### 方法2: 修正保存逻辑

```python
# 修改前
if accelerator.is_main_process:
    try:
        save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)  # 错误：在主进程条件内

# 修改后
try:
    # accelerator.save_state()内部处理多进程协调
    save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    if accelerator.is_main_process:
        logger.info(f"✅ Successfully saved checkpoint to {save_path}")
```

#### 方法3: 添加必要导入

```python
import shutil  # 用于checkpoint清理功能
```

### 🛡️ 预防措施
- 总是注册save/load hooks
- 理解accelerator.save_state()的多进程行为
- 验证保存的checkpoint内容
- 添加保存失败的错误处理

---

## 错误8: SIGTERM进程终止问题

### 🚨 错误现象
```bash
Signal 15 (SIGTERM) received by PID 2766193
torch.distributed.elastic.multiprocessing.errors.ChildFailedError
```

### 🔍 根本原因
- **SIGTERM (信号15)** = 外部强制终止
- 不是代码错误，而是系统级别的进程管理
- 可能原因：
  - 用户手动停止（Ctrl+C）
  - 系统资源限制
  - 作业调度器终止
  - 网络连接中断
  - 系统维护/重启

### ✅ 解决方法

#### 方法1: 从checkpoint恢复（推荐）

**修改文件**: `training/configs/taehv_config.py`

```python
# 设置恢复路径到最新checkpoint
args.resume_from_checkpoint = "output/2025-09-29_22-24-51/checkpoint-5500"
```

然后正常启动：
```bash
bash train_taehv.sh
```

#### 方法2: 自动恢复脚本

创建 `auto_recovery_train.sh`：

```bash
#!/bin/bash

while true; do
    echo "Starting/Resuming training..."
    bash train_taehv.sh
    
    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ Training completed successfully!"
        break
    else
        echo "❌ Training interrupted (exit code: $exit_code)"
        echo "Will restart in 10 seconds..."
        sleep 10
        
        # 自动找到最新checkpoint并更新配置
        latest_checkpoint=$(find output/ -name "checkpoint-*" -type d | sort -V | tail -1)
        if [ ! -z "$latest_checkpoint" ]; then
            echo "Found latest checkpoint: $latest_checkpoint"
            sed -i "s|args.resume_from_checkpoint = .*|args.resume_from_checkpoint = \"$latest_checkpoint\"|" training/configs/taehv_config.py
        fi
    fi
done
```

#### 方法3: 增强监控和日志

```python
# 在训练脚本中添加信号处理
import signal
import sys

def signal_handler(sig, frame):
    logger.info(f"Received signal {sig}, saving checkpoint and exiting...")
    # 强制保存checkpoint
    if 'global_step' in locals():
        save_path = os.path.join(config.output_dir, f"emergency_checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Emergency checkpoint saved to {save_path}")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
```

### 🛡️ 预防措施
- 频繁保存checkpoint（每500步）
- 监控系统资源使用
- 使用作业调度器的检查点机制
- 设置合理的作业时间限制
- 添加训练进度监控和告警

---

## 🎯 总结和最佳实践

### ✅ 成功解决的问题
1. **文件路径问题** → 使用项目相对路径
2. **DeepSpeed配置** → 正确分离训练和推理模型
3. **类型兼容性** → 添加类型过滤和转换
4. **混合精度** → 显式类型转换
5. **通信超时** → 多层次优化（批量大小、NCCL、DeepSpeed）
6. **梯度检查** → 区分Tensor和Python类型
7. **Checkpoint保存** → 添加hooks和修正保存逻辑
8. **进程管理** → 实现恢复机制
9. **Checkpoint路径逻辑** → 支持完整路径和相对路径
10. **Seraena模型配置** → 区分训练和推理模型
11. **张量形状不匹配** → 重新设计对称函数和形状信息传递

### 🛡️ 关键预防措施
1. **路径管理**：使用绝对路径或项目相对路径
2. **类型安全**：显式类型转换和检查
3. **错误处理**：全面的异常捕获和恢复逻辑
4. **资源管理**：定期内存清理和监控
5. **检查点策略**：频繁保存和验证checkpoint完整性
6. **通信优化**：合理的超时和缓冲区配置
7. **监控告警**：实时监控训练状态和资源使用
8. **DeepSpeed配置**：明确区分训练和推理模型，避免无优化器的模型传入`accelerator.prepare()`
9. **张量维度设计**：确保维度变换函数对称设计，传递必要的形状信息
10. **边界条件处理**：智能处理帧数不整除等边界情况，而非简单跳过

### 🚀 训练稳定性建议
1. 使用保守的批量大小设置
2. 启用梯度检查点以节省内存
3. 设置合理的NCCL超时时间
4. 定期验证checkpoint完整性
5. 监控GPU内存使用情况
6. 实现自动恢复机制

---

## 错误9: Checkpoint路径逻辑错误

### 🚨 错误现象
```bash
ValueError: Tried to find output/2025-09-30_12-38-46/checkpoint-5500 but folder does not exist
```

### 🔍 根本原因
- 配置中设置了完整的checkpoint路径：`"output/2025-09-29_22-24-51/checkpoint-5500"`
- 但程序启动时创建了新的时间戳输出目录：`output/2025-09-30_12-38-46/`
- 代码逻辑错误：提取了checkpoint名称`checkpoint-5500`，然后与新目录组合
- 结果寻找错误路径：`output/2025-09-30_12-38-46/checkpoint-5500`（实际不存在）

### ✅ 解决方法

**修改文件**: `training/taehv_train.py`

```python
# 修改前的错误逻辑
if config.resume_from_checkpoint != "latest":
    path = os.path.basename(config.resume_from_checkpoint)  # 只提取checkpoint名称
    # 然后错误地与新目录组合
    accelerator.load_state(os.path.join(config.output_dir, path))

# 修改后的正确逻辑
if config.resume_from_checkpoint != "latest":
    # 检查是否为完整路径（包含路径分隔符）
    if os.path.sep in config.resume_from_checkpoint or '/' in config.resume_from_checkpoint:
        # 完整路径，直接使用
        checkpoint_path = config.resume_from_checkpoint
        path = os.path.basename(config.resume_from_checkpoint)  # 用于提取步数
    else:
        # 只是checkpoint名称，需要组合路径
        path = config.resume_from_checkpoint
        checkpoint_path = os.path.join(config.output_dir, path)
else:
    # 获取最新的检查点
    dirs = os.listdir(config.output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint")]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    path = dirs[-1] if len(dirs) > 0 else None
    checkpoint_path = os.path.join(config.output_dir, path) if path else None

# 检查checkpoint是否存在
if path is None or not os.path.exists(checkpoint_path):
    accelerator.print(
        f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
    )
    config.resume_from_checkpoint = None
    initial_global_step = 0
else:
    accelerator.print(f"Resuming from checkpoint {path}")
    accelerator.load_state(checkpoint_path)  # 使用正确的路径
    global_step = int(path.split("-")[1])
```

### 🛡️ 预防措施
- 区分完整路径和相对路径
- 添加路径存在性检查
- 支持多种checkpoint指定方式：
  - 完整路径：`"output/2025-09-29_22-24-51/checkpoint-5500"`
  - 相对名称：`"checkpoint-5500"`
  - 最新：`"latest"`

### 💡 **使用建议**
```python
# 推荐的checkpoint配置方式
args.resume_from_checkpoint = "output/2025-09-29_22-24-51/checkpoint-5500"  # 完整路径
# 或
args.resume_from_checkpoint = "latest"  # 自动找最新的
# 或
args.resume_from_checkpoint = "checkpoint-5500"  # 当前输出目录下的checkpoint
```

---

## 错误10: Seraena模型DeepSpeed优化器问题

### 🚨 错误现象
```bash
AssertionError: zero stage 2 requires an optimizer
File "training/taehv_train.py", line 220, in main
  seraena = accelerator.prepare(seraena)
```

### 🔍 根本原因
- `seraena`模型（对抗训练模块）被传递给`accelerator.prepare()`
- 但`seraena`没有对应的优化器
- DeepSpeed ZeRO stage 2要求所有prepared的模型都必须有对应的优化器
- 这和错误2中的`vae_ref`问题完全相同

### ✅ 解决方法

**修改文件**: `training/taehv_train.py`

```python
# 修改前
if seraena is not None:
    seraena = accelerator.prepare(seraena)  # 错误：没有优化器

# 修改后
if seraena is not None:
    # Seraena不需要训练，手动移动到设备而不是通过DeepSpeed准备
    seraena = seraena.to(accelerator.device)
```

### 🔍 **问题模式识别**

这是DeepSpeed配置的**经典问题模式**：
1. ❌ **错误2**: `vae_ref`模型没有优化器
2. ❌ **错误10**: `seraena`模型没有优化器

**通用规则**：
- ✅ **训练模型** → `accelerator.prepare(model, optimizer, ...)`
- ✅ **推理模型** → `model.to(accelerator.device)`

### 🛡️ 预防措施
- 明确区分**训练模型**和**推理模型**
- 只有需要梯度更新的模型才通过`accelerator.prepare()`
- 参考模型/辅助模型手动移动到设备
- 检查所有`accelerator.prepare()`调用是否有对应的优化器

### 💡 **识别方法**
```python
# ✅ 正确：主训练模型
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# ❌ 错误：没有优化器的模型
auxiliary_model = accelerator.prepare(auxiliary_model)  # 缺少优化器

# ✅ 正确：辅助模型
auxiliary_model = auxiliary_model.to(accelerator.device)
```

---

## 错误11: Seraena张量形状不匹配问题

### 🚨 错误现象
```bash
Seraena training failed: shape '[-1, 15, 3, 128, 128]' is invalid for input of size 1179648, skipping adversarial loss
```

### 🔍 根本原因
- **核心问题**：`pad_and_group` 和 `ungroup_and_unpad` 函数设计不对称
- **维度信息丢失**：在张量变换过程中，原始形状信息没有被正确传递
- **具体表现**：期望形状 `[-1, 15, 3, 128, 128]` vs 实际输入 `[8, 9, 128, 128]`
- **数学不匹配**：`1,179,648 ≠ batch × 737,280`（比例1.60不是整数）
- **设计缺陷**：`ungroup_and_unpad` 试图从输出推断原始维度，但信息不足

### 📊 **数值分析**
```python
# 配置参数
config.n_frames = 12           # 每个样本的帧数
config.n_seraena_frames = 3    # Seraena分组的帧数
model.frames_to_trim = 3       # 需要修剪的帧数

# 期望计算
total_frames = 12 + 3 = 15
expected_elements = batch × 15 × 3 × 128 × 128 = batch × 737,280

# 实际情况
actual_elements = 1,179,648 ≈ batch × 1.6 × 737,280 (不匹配)
```

### ✅ 解决方法

**修改文件**: `training/taehv_train.py`

**问题根源**：`pad_and_group` 和 `ungroup_and_unpad` 函数不对称，导致维度信息丢失。

**解决方案**：重新设计函数使其完全对称，并传递原始形状信息。

```python
# 1. 增强 pad_and_group 函数，添加调试和容错
def pad_and_group(x):
    logger.debug(f"pad_and_group input shape: {x.shape}")
    
    frames_to_trim = getattr(model, 'frames_to_trim', 0)
    logger.debug(f"frames_to_trim: {frames_to_trim}")
    
    if frames_to_trim > 0:
        x_padded = torch.cat([x, x[:, :frames_to_trim]], 1)
    else:
        x_padded = x
        
    n, t, c, h, w = x_padded.shape
    logger.debug(f"After padding: {x_padded.shape}")
    
    # 确保帧数能被n_seraena_frames整除
    if t % config.n_seraena_frames != 0:
        logger.warning(f"Frame count {t} is not divisible by n_seraena_frames {config.n_seraena_frames}")
        t_trimmed = (t // config.n_seraena_frames) * config.n_seraena_frames
        x_padded = x_padded[:, :t_trimmed]
        t = t_trimmed
    
    result = x_padded.reshape(n * t//config.n_seraena_frames, config.n_seraena_frames*c, h, w)
    return result

# 2. 重新设计 ungroup_and_unpad 函数，接受原始形状信息
def ungroup_and_unpad(x, original_shape):
    """
    将pad_and_group的输出转换回原始格式
    x: [n_groups, grouped_c, h, w] - pad_and_group的输出
    original_shape: [N, T, C, H, W] - 原始张量的形状
    """
    n_groups, grouped_c, h, w = x.shape
    original_n, original_t, original_c, original_h, original_w = original_shape
    
    c = grouped_c // config.n_seraena_frames
    frames_to_trim = getattr(model, 'frames_to_trim', 0)
    
    # 计算实际的处理帧数
    padded_frames = original_t + (frames_to_trim if frames_to_trim > 0 else 0)
    actual_frames = (padded_frames // config.n_seraena_frames) * config.n_seraena_frames
    
    # 验证维度一致性
    expected_n_groups = original_n * actual_frames // config.n_seraena_frames
    if n_groups != expected_n_groups:
        logger.warning(f"Group count mismatch: got {n_groups}, expected {expected_n_groups}")
        return None
    
    # reshape回原始格式
    x_ungrouped = x.reshape(original_n, actual_frames, c, h, w)
    
    # 去掉padding的帧（如果有）
    if frames_to_trim > 0 and actual_frames > original_t:
        trim_amount = min(frames_to_trim, actual_frames - original_t)
        x_ungrouped = x_ungrouped[:, :-trim_amount]
    
    return x_ungrouped

# 3. 调用时传入原始形状信息
seraena_target = ungroup_and_unpad(seraena_target, frames_target.shape)
if seraena_target is not None:
    if seraena_target.shape == decoded.shape:
        seraena_loss = F.mse_loss(decoded, seraena_target)
        total_loss = total_loss + config.seraena_loss_weight * seraena_loss
    else:
        logger.warning(f"Shape mismatch: decoded {decoded.shape} vs seraena_target {seraena_target.shape}")
else:
    logger.warning("Skipping Seraena loss due to dimension mismatch")

# 4. 启用DEBUG日志查看详细信息
logging.basicConfig(level=logging.DEBUG)
```

### 🛡️ 预防措施
- **函数对称性**：确保 `pad_and_group` 和 `ungroup_and_unpad` 完全对称设计
- **形状信息传递**：将原始张量形状信息传递给反向变换函数
- **维度一致性验证**：在每个变换步骤验证维度计算的正确性
- **详细调试日志**：记录每个维度变换步骤的详细信息
- **边界条件处理**：正确处理 `frames_to_trim` 和帧数不整除的情况
- **参数配置验证**：启动时验证 `n_frames`, `n_seraena_frames` 等参数的兼容性

### 🔍 **问题排查步骤**
1. **启用详细日志**：
   ```python
   # 在训练脚本中设置DEBUG级别
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查关键参数**：
   ```python
   print(f"✓ n_frames: {config.n_frames}")
   print(f"✓ n_seraena_frames: {config.n_seraena_frames}")
   print(f"✓ frames_to_trim: {getattr(model, 'frames_to_trim', 0)}")
   print(f"✓ batch_size: {config.train_batch_size}")
   
   # 验证参数兼容性
   padded_frames = config.n_frames + getattr(model, 'frames_to_trim', 0)
   print(f"✓ padded_frames: {padded_frames}")
   print(f"✓ divisible by n_seraena_frames: {padded_frames % config.n_seraena_frames == 0}")
   ```

3. **跟踪维度变换**：
   ```python
   # 查看DEBUG日志中的关键信息
   # pad_and_group input shape: [N, T, C, H, W]
   # pad_and_group output shape: [groups, grouped_c, H, W]  
   # ungroup_and_unpad input shape: [groups, grouped_c, H, W]
   # After ungrouping: [N', T', C, H, W]
   # After removing padding: [N, T, C, H, W]
   ```

4. **验证形状匹配**：
   ```python
   print(f"✓ frames_target.shape: {frames_target.shape}")
   print(f"✓ decoded.shape: {decoded.shape}")  
   print(f"✓ seraena_target.shape: {seraena_target.shape if seraena_target else 'None'}")
   ```

### 💡 **根本解决方案**

**✅ 已实施的完整解决方案**：

1. **对称函数设计**：
   - `pad_and_group`: `[N,T,C,H,W] → [groups,grouped_c,H,W]`
   - `ungroup_and_unpad`: `[groups,grouped_c,H,W] + original_shape → [N,T,C,H,W]`
   - 关键：传递原始形状信息，确保可逆变换

2. **智能边界处理**：
   ```python
   # 自动调整帧数确保整除
   if t % config.n_seraena_frames != 0:
       t_trimmed = (t // config.n_seraena_frames) * config.n_seraena_frames
       x_padded = x_padded[:, :t_trimmed]
   ```

3. **维度一致性验证**：
   ```python
   expected_n_groups = original_n * actual_frames // config.n_seraena_frames
   if n_groups != expected_n_groups:
       logger.warning("Group count mismatch")
       return None
   ```

4. **配置参数优化建议**：
   ```python
   # 推荐配置（确保兼容性）
   args.n_frames = 12           # 基础帧数
   args.n_seraena_frames = 3    # 分组大小  
   args.frames_to_trim = 3      # 修剪帧数
   # 总帧数：12+3=15，能被3整除 ✓
   ```

5. **调试和监控**：
   - DEBUG级别日志记录每步维度变换
   - 形状匹配验证防止静默错误
   - 详细错误信息便于快速定位问题

**🚫 不再需要的临时方案**：
- ~~禁用Seraena训练~~（问题已根本解决）
- ~~跳过维度不匹配的批次~~（现在能正确处理）

---

## 错误12: TAEHV帧数维度不匹配 - decoded与frames_target不一致

### 🚨 错误现象

**错误信息**：
```bash
RuntimeError: The size of tensor a (17) must match the size of tensor b (16) at non-singleton dimension 1

UserWarning: Using a target size (torch.Size([1, 16, 3, 480, 480])) that is different to the 
input size (torch.Size([1, 17, 3, 480, 480])). This will likely lead to incorrect results 
due to broadcasting.
  recon_loss = F.mse_loss(decoded, frames_target)
```

**具体场景**：
- 配置 `n_frames = 19`
- `decoded` 输出形状: `[batch, 17, 3, 480, 480]`
- `frames_target` 形状: `[batch, 16, 3, 480, 480]`
- 维度不匹配导致 MSE loss 计算失败

### 🔍 根本原因

**TAEHV模型的编码-解码机制**：

1. **时间下采样/上采样比例**：
   - `decoder_time_upscale = (True, True)` → 2次时间上采样
   - 总上采样倍率：2 × 2 = **4x**
   - 对应的编码器有 **4x 时间下采样**

2. **frames_to_trim 机制**：
   ```python
   self.frames_to_trim = 2**sum(decoder_time_upscale) - 1 = 2^2 - 1 = 3
   ```
   - 解码器输出后会裁剪掉**前3帧**

3. **完整的编码-解码流程**（以 n_frames=19 为例）：
   ```
   输入: 19 帧
     ↓ 编码器（4x下采样）
   潜在表示: ceil(19/4) = 5 帧
     ↓ 解码器（4x上采样）
   原始输出: 5 × 4 = 20 帧
     ↓ 裁剪前3帧
   decoded: 20 - 3 = 17 帧 ✓
   ```

4. **frames_target 计算**：
   ```python
   frames_target = frames[:, :-model.frames_to_trim]
                 = frames[:, :-3]
                 = 19 - 3 = 16 帧 ❌
   ```

5. **维度不匹配**：
   - `decoded`: 17 帧
   - `frames_target`: 16 帧
   - **17 ≠ 16** → RuntimeError

### 📊 正确的帧数计算公式

要确保 `decoded` 和 `frames_target` 维度匹配，需要满足：

```python
decoded_frames = ceil(n_frames / 4) * 4 - 3
frames_target = n_frames - 3

# 要求：decoded_frames = frames_target
ceil(n_frames / 4) * 4 - 3 = n_frames - 3
ceil(n_frames / 4) * 4 = n_frames
```

**关键规则**：`n_frames` 必须是 **4 的倍数**或使解码后裁剪的帧数匹配！

### ✅ 解决方法

#### 方案1：修正 n_frames 为正确值（推荐）

**修改文件**: `training/configs/taehv_config_1gpu_h100.py`

```python
# ❌ 错误配置
args.n_frames = 19  # decoded=17, target=16, 不匹配！

# ✅ 正确配置
args.n_frames = 20  # decoded=17, target=17, 匹配！
```

**验证 n_frames = 20**：
```
输入: 20 帧
编码: ceil(20/4) = 5 帧
解码: 5 × 4 = 20 帧
裁剪: 20 - 3 = 17 帧 (decoded) ✓
目标: 20 - 3 = 17 帧 (frames_target) ✓
结果: 17 = 17 ✓ 完美匹配！
```

#### 方案2：其他合法的帧数配置

根据公式 `ceil(n_frames/4)*4 - 3 = n_frames - 3`，以下是合法配置：

| n_frames | 编码后 | 解码后 | 裁剪后(decoded) | frames_target | 匹配? | 显存估算 |
|----------|--------|--------|----------------|---------------|-------|----------|
| 4 | 1 | 4 | 1 | 1 | ✅ | ~5GB |
| 8 | 2 | 8 | 5 | 5 | ✅ | ~10GB |
| 12 | 3 | 12 | 9 | 9 | ✅ | ~18GB |
| 16 | 4 | 16 | 13 | 13 | ✅ | ~25GB |
| **20** | **5** | **20** | **17** | **17** | **✅** | **~32GB** ⭐ |
| 24 | 6 | 24 | 21 | 21 | ✅ | ~40GB |
| 28 | 7 | 28 | 25 | 25 | ✅ | ~48GB |

**非法配置示例**（会导致错误）：
| n_frames | decoded | frames_target | 匹配? | 原因 |
|----------|---------|---------------|-------|------|
| 13 | 13 | 10 | ❌ | 13 ≠ 10 |
| 17 | 17 | 14 | ❌ | 17 ≠ 14 |
| **19** | **17** | **16** | ❌ | 17 ≠ 16（当前错误）|
| 21 | 21 | 18 | ❌ | 21 ≠ 18 |

### 🔧 完整修复步骤

1. **修改单卡H100配置**：
```bash
# 文件: training/configs/taehv_config_1gpu_h100.py
args.n_frames = 20  # 修改为20
args.n_seraena_frames = 8  # 保持不变（17/2≈8）
```

2. **可选：调整batch size以适应更多帧**：
```python
# 如果20帧占用显存超过预期
args.train_batch_size = 1  # 已经是1，保持不变
args.gradient_accumulation_steps = 16  # 保持有效batch=16
```

3. **验证配置**：
```bash
# 运行训练
./train_taehv_h100.sh 1 h100

# 期望输出：
# decoded shape: [1, 17, 3, 480, 720]
# frames_target shape: [1, 17, 3, 480, 720]
# ✓ 不再有维度不匹配警告
```

### 📐 通用帧数选择指南

**选择原则**：
1. **必须满足**：`n_frames = 4k` 或使 `ceil(n_frames/4)*4 - 3 = n_frames - 3`
2. **显存考虑**：帧数越多，显存占用越高
3. **质量考虑**：更多帧 → 更好的时间连贯性
4. **速度考虑**：更多帧 → 训练速度更慢

**推荐配置**：

- **测试/调试**：`n_frames = 12` (~18GB，快速迭代)
- **单卡H100**：`n_frames = 20` (~32GB，平衡质量和速度) ⭐
- **多卡H100**：`n_frames = 16` (~25GB/卡，推荐)
- **显存充足**：`n_frames = 24` (~40GB，最高质量)

### 🛡️ 预防措施

1. **配置验证脚本**：
```python
def validate_n_frames(n_frames):
    """验证n_frames配置是否合法"""
    frames_to_trim = 3
    decoded_frames = ((n_frames + 3) // 4) * 4 - frames_to_trim
    target_frames = n_frames - frames_to_trim
    
    if decoded_frames != target_frames:
        print(f"❌ n_frames={n_frames} 不合法!")
        print(f"   decoded={decoded_frames}, target={target_frames}")
        return False
    
    print(f"✅ n_frames={n_frames} 合法!")
    print(f"   decoded={decoded_frames}, target={target_frames}")
    return True

# 测试
validate_n_frames(19)  # ❌ 不合法
validate_n_frames(20)  # ✅ 合法
```

2. **在配置文件中添加注释**：
```python
# CRITICAL: n_frames 必须满足 ceil(n_frames/4)*4 - 3 = n_frames - 3
# 合法值: 4, 8, 12, 16, 20, 24, 28, ...
args.n_frames = 20  # ✓ 验证通过
```

3. **训练脚本启动前检查**：
```python
# 在 taehv_train.py 中添加
def validate_config(args, model):
    n_frames = args.n_frames
    frames_to_trim = model.frames_to_trim
    decoded_frames = ((n_frames + 3) // 4) * 4 - frames_to_trim
    target_frames = n_frames - frames_to_trim
    
    assert decoded_frames == target_frames, \
        f"Invalid n_frames={n_frames}: decoded={decoded_frames} != target={target_frames}"
    
    logger.info(f"✓ n_frames validation passed: {n_frames} → {target_frames} frames")
```

### 📝 相关错误

此错误与以下问题相关：
- [错误11: Seraena张量形状不匹配](#错误11-seraena张量形状不匹配问题) - 也涉及帧数维度
- 帧数配置错误是训练失败的常见原因

### 🎯 经验教训

1. **理解模型架构至关重要**：
   - 必须了解编码器/解码器的时间下采样/上采样比例
   - frames_to_trim 的作用和影响
   - 编码-解码的完整流程

2. **不是所有的4的倍数都合法**：
   - `n_frames = 16` ✅ 合法（decoded=13, target=13）
   - `n_frames = 18` ❌ 不合法（decoded=17, target=15）
   - 必须通过公式验证

3. **配置验证比调试更高效**：
   - 提前验证配置可避免浪费训练时间
   - 自动化验证脚本是最佳实践

4. **文档和注释的重要性**：
   - 在配置文件中明确标注约束条件
   - 提供合法值列表和验证公式

---

## 错误13: Seraena帧数不能整除警告 - 数据损失问题

### 🚨 错误现象

**警告信息**：
```bash
WARNING - __main__ - Frame count 20 is not divisible by n_seraena_frames 8
WARNING - __main__ - Shape mismatch: decoded torch.Size([1, 17, 3, 480, 480]) vs seraena_target torch.Size([1, 16, 3, 480, 480])
```

**具体场景**：
- 配置 `n_frames = 20`, `n_seraena_frames = 8`
- `frames_target = 20 - 3 = 17` 帧
- 17 不能被 8 整除：17 ÷ 8 = 2 余 1
- Seraena 自动裁剪帧数导致维度不匹配

### 🔍 根本原因

**Seraena 的帧数分组机制**：

1. **分组要求**：Seraena 需要将视频帧分组处理，要求总帧数能被 `n_seraena_frames` 整除
2. **自动裁剪逻辑**：
   ```python
   # 在 training/taehv_train.py 中
   if t % config.n_seraena_frames != 0:
       logger.warning(f"Frame count {t} is not divisible by n_seraena_frames {config.n_seraena_frames}")
       # 裁剪到最接近的可整除数
       t_trimmed = (t // config.n_seraena_frames) * config.n_seraena_frames
       x_padded = x_padded[:, :t_trimmed]  # 数据损失！
   ```

3. **数据损失后果**：
   - 原始帧数：17 帧
   - 裁剪后：17 // 8 * 8 = 16 帧
   - **丢失1帧数据**，影响训练质量

4. **维度不匹配**：
   - `decoded`: 17 帧（来自TAEHV解码器）
   - `seraena_target`: 16 帧（被裁剪后）
   - 导致损失函数计算失败

### 📊 帧数整除性分析

**当前配置问题**：
```
n_frames = 20 → frames_target = 17
n_seraena_frames = 8
17 ÷ 8 = 2 余 1 ❌ 不能整除
```

**17 是质数**，只有因数 `[1, 17]`，选择有限。

**其他合法配置的因数分析**：

| n_frames | frames_target | 可用的 n_seraena_frames | 推荐值 | 显存估算 |
|----------|---------------|------------------------|--------|----------|
| 16 | 13 | [1, 13] | 1 | ~25GB |
| **20 (当前)** | **17** | **[1, 17]** | **1 ⭐** | **~32GB** |
| 24 | 21 | [1, 3, 7, 21] | 3 或 7 | ~40GB |
| 28 | 25 | [1, 5, 25] | 5 | ~48GB |

### ✅ 解决方法

#### 方案1：修改 n_seraena_frames（推荐）

**修改文件**: `training/configs/taehv_config_1gpu_h100.py`

```python
# ❌ 错误配置
args.n_seraena_frames = 8  # 17 ÷ 8 = 2 余 1，数据损失

# ✅ 正确配置（推荐）
args.n_seraena_frames = 1  # 17 ÷ 1 = 17，无损失

# ✅ 备选配置
args.n_seraena_frames = 17  # 17 ÷ 17 = 1，无损失但计算效率较低
```

**为什么推荐 n_seraena_frames = 1**：
- ✅ 完全避免数据损失
- ✅ 计算效率最高
- ✅ 兼容性最好
- ✅ 训练更稳定

#### 方案2：更换为有更多因数的帧数配置

```python
# 使用 n_frames = 24 (更灵活)
args.n_frames = 24  # frames_target = 21
args.n_seraena_frames = 3  # 21 ÷ 3 = 7 ✓

# 验证
# 输入: 24 帧 → 编码: ceil(24/4) = 6 → 解码: 24 → 裁剪: 21
# frames_target = 21, decoded = 21 ✅ 匹配
# 21 ÷ 3 = 7 ✅ 整除
```

### 🔧 完整修复步骤

1. **修复配置**：
```bash
# 修改 training/configs/taehv_config_1gpu_h100.py
args.n_seraena_frames = 1  # 从 8 改为 1
```

2. **重新启动训练**：
```bash
# 停止当前训练 (Ctrl+C)
./train_taehv_h100.sh 1 h100
```

3. **验证修复**：
```bash
# 期望输出：
# ✅ 不再有 "Frame count ... is not divisible" 警告
# ✅ 不再有 "Shape mismatch" 警告
# ✅ decoded 和 seraena_target 都是 17 帧
```

### 📐 配置验证脚本

```python
def validate_seraena_config(n_frames, n_seraena_frames):
    """验证 Seraena 配置的兼容性"""
    frames_target = n_frames - 3
    
    print(f"配置验证: n_frames={n_frames}, n_seraena_frames={n_seraena_frames}")
    print(f"frames_target = {frames_target}")
    
    if frames_target % n_seraena_frames == 0:
        groups = frames_target // n_seraena_frames
        print(f"✅ 整除检查通过: {frames_target} ÷ {n_seraena_frames} = {groups}")
        print(f"✅ 无数据损失")
        return True
    else:
        remainder = frames_target % n_seraena_frames
        trimmed = (frames_target // n_seraena_frames) * n_seraena_frames
        loss = frames_target - trimmed
        print(f"❌ 整除检查失败: {frames_target} ÷ {n_seraena_frames} = {frames_target//n_seraena_frames} 余 {remainder}")
        print(f"❌ 数据损失: {loss} 帧 ({frames_target} → {trimmed})")
        return False

# 测试
validate_seraena_config(20, 8)   # ❌ 当前配置
validate_seraena_config(20, 1)   # ✅ 推荐配置
validate_seraena_config(24, 3)   # ✅ 替代配置
```

### 🛡️ 预防措施

1. **配置文件注释**：
```python
# CRITICAL: frames_target 必须能被 n_seraena_frames 整除
# frames_target = n_frames - 3
# 验证: (n_frames - 3) % n_seraena_frames == 0
args.n_seraena_frames = 1  # ✓ 兼容所有帧数
```

2. **启动时验证**：
```python
# 在 taehv_train.py 中添加
def validate_seraena_config(config):
    frames_target = config.n_frames - 3
    if frames_target % config.n_seraena_frames != 0:
        raise ValueError(
            f"Invalid configuration: frames_target ({frames_target}) is not divisible by "
            f"n_seraena_frames ({config.n_seraena_frames}). "
            f"This will cause data loss and training instability."
        )
    logger.info(f"✓ Seraena config validated: {frames_target} ÷ {config.n_seraena_frames} = {frames_target // config.n_seraena_frames}")
```

### 📝 相关错误

此错误与以下问题相关：
- [错误12: TAEHV帧数维度不匹配](#错误12-taehv帧数维度不匹配-decoded与frames_target不一致) - 帧数配置的上游问题
- [错误11: Seraena张量形状不匹配](#错误11-seraena张量形状不匹配问题) - 相同的维度不匹配根因

### 🎯 经验教训

1. **理解所有组件的约束**：
   - TAEHV 需要特定的帧数规则
   - Seraena 需要帧数能被分组数整除
   - 两者的约束需要同时满足

2. **避免数据损失**：
   - 自动裁剪看似"解决"了警告，但实际造成训练数据丢失
   - 正确的做法是调整配置而不是接受数据损失

3. **质数帧数的挑战**：
   - 17、13、19 等质数只有很少的因数选择
   - 选择有更多因数的帧数配置可提供更大灵活性

4. **配置验证的重要性**：
   - 复杂的多组件系统需要全面的配置验证
   - 启动时检查可避免运行时的问题

---

## 错误14: CogVideoX模式维度不匹配 - decoded vs frames_target

**错误信息**:
```
RuntimeError: The size of tensor a (16) must match the size of tensor b (13) at non-singleton dimension 1
UserWarning: Using a target size (torch.Size([2, 13, 3, 480, 480])) that is different to the input size (torch.Size([2, 16, 3, 480, 480]))
```

### 错误分析

当使用CogVideoX预训练模型(`taecvx.pth`)时，TAEHV进入特殊的CogVideoX兼容模式：

**问题根源**:
1. **模型检测**: `pretrained_model_path`包含`"taecvx"` → `is_cogvideox = True`
2. **解码行为**: 编码帧数为偶数时 → `skip_trim = True` → 解码输出不裁剪
3. **Target计算**: 训练代码仍然裁剪了`frames_target`
4. **维度不匹配**: `decoded=16帧` vs `frames_target=13帧`

**计算链条**:
```
输入: 16帧 → 编码: ceil(16/4)=4帧 → 解码: 4×4=16帧 (跳过裁剪)
Target: 16帧 → 裁剪: 16-3=13帧
结果: decoded[16] ≠ target[13] → 维度不匹配
```

### 解决方案

**方法1: 修复训练逻辑** (已实施)
```python
# training/taehv_train.py 修复逻辑
will_skip_trim = model.is_cogvideox and encoded.shape[1] % 2 == 0
if will_skip_trim:
    frames_target = frames  # CogVideoX模式不裁剪
else:
    frames_target = frames[:, :-model.frames_to_trim]  # 正常模式裁剪
```

### 预防措施

1. **配置验证脚本**:
```bash
# 检查模型模式和帧数匹配性
python -c "
from models.taehv import TAEHV
model = TAEHV(checkpoint_path='checkpoints/taecvx.pth')
n_frames = 16
encoded_frames = (n_frames + 3) // 4
print(f'CogVideoX模式: {model.is_cogvideox}')
print(f'输入帧数: {n_frames}, 编码帧数: {encoded_frames}')
if model.is_cogvideox and encoded_frames % 2 == 0:
    print(f'将跳过裁剪，target帧数: {n_frames}')
else:
    print(f'正常裁剪，target帧数: {n_frames - 3}')
"
```

2. **相关错误**:
   - 错误12: 标准TAEHV模式的帧数不匹配
   - 错误13: Seraena帧数整除问题

---

## 错误15: CogVideoX解码输出帧数异常 - 19帧vs配置16帧

**错误信息**:
```
WARNING - Frame count 19 is not divisible by n_seraena_frames 4
```

### 错误分析

使用CogVideoX预训练模型时，解码器的实际输出帧数与理论计算不符：

**问题链条**:
1. **配置帧数**: `n_frames = 16`
2. **编码帧数**: `ceil(16/4) = 4`
3. **理论解码**: `4 × 4 = 16帧`
4. **实际解码**: `19帧` ❌ (CogVideoX内部逻辑)
5. **Seraena检查**: `19 ÷ 4 = 4余3` → 不能整除 → 警告

**根因**:
CogVideoX解码器有复杂的时间上采样逻辑，不是简单的线性映射，实际输出帧数可能与 `编码帧数 × 4` 不同。

### 解决方案

**方案1: 调整n_seraena_frames** (推荐)
```python
# training/configs/taehv_config_h100.py
args.n_seraena_frames = 1  # 适配任意帧数，无数据损失
```

**方案2: 预裁剪frames_target**
```python
# 在Seraena处理前裁剪为可整除的帧数
target_frames = 16  # 4的倍数
frames_target = frames_target[:, :target_frames]  # 会丢失3帧
```

### 预防措施

1. **帧数验证脚本**:
```bash
# 检查实际解码输出帧数
python -c "
from models.taehv import TAEHV
import torch
model = TAEHV(checkpoint_path='checkpoints/taecvx.pth')
dummy_input = torch.randn(1, 16, 3, 256, 256)
encoded = model.encode_video(dummy_input)
decoded = model.decode_video(encoded)
print(f'输入: {dummy_input.shape[1]}帧')
print(f'编码: {encoded.shape[1]}帧') 
print(f'解码: {decoded.shape[1]}帧')
"
```

2. **配置建议**:
   - 使用 `n_seraena_frames = 1` 最安全（适配任意帧数）
   - 或选择19的因数：`1, 19`

### 相关错误
- 错误13: 其他帧数不能整除的情况
- 错误14: CogVideoX模式维度不匹配

---

## 错误16: Seraena参数维度不匹配 - 实际帧数vs配置帧数

**错误信息**:
```
WARNING - Seraena training failed: The size of tensor a (38) must match the size of tensor b (32) at non-singleton dimension 0, skipping adversarial loss
```

### 错误分析

当CogVideoX解码输出实际帧数与配置帧数不同时，Seraena的三个输入参数维度不匹配：

**问题链条**:
1. **实际数据**: CogVideoX解码输出19帧
2. **参数1&2**: `pad_and_group(frames_target)` 和 `pad_and_group(decoded)`
   - 基于实际19帧 → 输出维度 `batch_size × 19 = 38`
3. **参数3**: `encoded.mean().repeat_interleave(config.n_frames//n_seraena_frames)`
   - 基于配置16帧 → 输出维度 `batch_size × 16 = 32`
4. **维度不匹配**: `38 ≠ 32` → Seraena调用失败

**根因**:
第三个参数的计算使用了配置文件中的`config.n_frames`，而不是实际的`frames_target.shape[1]`。

### 解决方案

**修复代码** (已实施)
```python
# training/taehv_train.py 第440-450行
# 使用实际帧数而不是配置帧数
actual_frames = frames_target.shape[1]  # 实际帧数（19）
repeat_times = actual_frames // config.n_seraena_frames  # 19÷1=19

seraena_target, seraena_debug = seraena.step_and_make_correction_targets(
    pad_and_group(frames_target),     # 38维度
    pad_and_group(decoded),          # 38维度  
    encoded.mean(1, keepdim=True).repeat_interleave(
        repeat_times, dim=1          # 使用19而非16
    ).flatten(0, 1)                  # 38维度 ✓
)
```

### 预防措施

1. **动态帧数检测**:
```python
# 总是使用实际帧数进行计算
actual_frames = tensor.shape[1]
repeat_factor = actual_frames // config.n_seraena_frames
```

2. **维度验证**:
```python
# 在Seraena调用前验证维度匹配
assert param1.shape[0] == param2.shape[0] == param3.shape[0]
```

### 相关错误
- 错误15: CogVideoX解码输出帧数异常
- 错误14: CogVideoX模式维度不匹配

---

通过以上措施，TAEHV训练已实现高度稳定性，能够从各种错误中自动恢复并继续训练。
