from easydict import EasyDict
import datetime
import pytz
import os

args = EasyDict()

# ============ Model Arguments ============
args.pretrained_model_path = "checkpoints/taecvx.pth"  # str: Path to TAEHV pretrained checkpoint
args.patch_size = 1  # int: TAEHV patch size
args.latent_channels = 16  # int: Number of latent channels
args.model_dtype = "bfloat16"  # str: Model data type - H100 native support

# ============ Dataset Arguments ============
# 生产环境配置 - 几百万mp4数据
args.data_root = "/mnt/project_modelware/zhaojian/matrix_rfl/matrix_dataset_5M_8M/the_matrix_dataset_5M_1600_900/data"  # str: Production training data folder
args.annotation_file = "/mnt/project_modelware/zhaojian/matrix_rfl/matrix_dataset_5M_8M/the_matrix_dataset_5M_1600_900/stage1_annotations_cleaned.json"  # str: Full annotation file
# args.data_root = "/data/matrix-project/MiniDataset/data"  # str: Training data folder
# args.annotation_file = "/data/matrix-project/MiniDataset/stage1_annotations_500.json"  # str: Annotation file
args.train_split = 0.95  # float: Training data split ratio (more data for training)
args.dataloader_num_workers = 11  # int: ✅ 48卡时适中workers
args.pin_memory = True  # bool: Whether to use pinned memory
args.prefetch_factor = 3  # int: 适中prefetch
args.persistent_workers = True  # bool: Keep workers alive between epochs

# ============ Video Processing Arguments ============
# 针对H100大显存，提升视频质量
args.height = 480  # int: Ultra high resolution for H100
args.width = 720  # int: Ultra high resolution for H100
args.n_frames = 12  # int: Reduced frames for high resolution (720x480)
args.min_frame_delta = 1  # int: Minimum frame interval
args.max_frame_delta = 2  # int: Reduced for better temporal consistency

# ============ Data Augmentation Arguments ============
args.random_crop = True  # bool: Whether to use random crop
args.random_flip = True  # bool: Whether to use random flip
args.color_jitter = True  # bool: Whether to use color jitter
args.color_jitter_params = {
    'brightness': 0.15,  # Increased for more robust training
    'contrast': 0.15,
    'saturation': 0.15,
    'hue': 0.08
}
args.gaussian_blur = True  # bool: Add Gaussian blur augmentation
args.gaussian_blur_sigma = (0.1, 2.0)  # tuple: Blur sigma range

# ============ Training Arguments ============
args.seed = 42  # int: Random seed
args.mixed_precision = "bf16"  # str: H100 native bf16 support
args.output_dir = "output/" + datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")  # str: Output directory
args.train_batch_size = 5  # int: ✅ 48卡时设为5（每卡），总有效batch=5×48×1=240
args.gradient_accumulation_steps = 1  # int: ✅ 48卡时无需累积
args.gradient_checkpointing = True  # bool: Whether to use gradient checkpointing
args.max_train_steps = 100000  # int: Extended training for production
args.checkpointing_steps = 750  # int: ✅ 改为750（64卡是500，16卡是1000）- 48卡适中
args.checkpoints_total_limit = 100  # int: Keep more checkpoints for production
args.resume_from_checkpoint = None  # str: Resume from checkpoint path
args.save_model_card = True  # bool: Save model card for production

# ============ Learning Rate Arguments ============
# ✅ 48卡训练：有效batch size = 5×48×1 = 240
# 介于16卡的128和64卡的256之间，学习率线性插值
args.learning_rate = 1.8e-4  # float: ✅ 介于1e-4和2e-4之间（线性插值）
args.lr_scheduler = "cosine"  # str: Better scheduler for long training
args.lr_warmup_steps = 1200  # int: ✅ 介于1500和1000之间 - 48卡适中
args.lr_num_cycles = 1  # int: Multiple cycles for long training
args.lr_power = 1.0  # float: Power for polynomial scheduler

# ============ Optimizer Arguments ============
args.optimizer = "adamw"  # str: Optimizer type
args.beta1 = 0.9  # float: Adam beta1
args.beta2 = 0.95  # float: Adam beta2
args.weight_decay = 1e-2  # float: Increased weight decay for regularization
args.epsilon = 1e-8  # float: Adam epsilon
args.max_grad_norm = 1.0  # float: Max gradient norm

# ============ Loss Function Arguments ============
# ✅✅✅ 关键修复：调整loss权重，重建优先！
args.reconstruction_loss_weight = 20.0  # ✅ 提高到20.0 - 最重要的修复！
args.encoder_loss_weight = 0.1         # ✅ 降低到0.1
args.decoder_loss_weight = 0.1         # ✅ 降低到0.1
args.perceptual_loss_weight = 0.1     # float: Add perceptual loss for better quality

# ============ Seraena Arguments ============
args.use_seraena = False  # bool: ✅ 禁用Seraena，避免帧数不匹配问题
args.seraena_loss_weight = 0.0  # ✅ 权重设为0
args.n_seraena_frames = 0  # int: 设为0（不使用）

# ============ Validation Arguments ============
# ✅ 验证系统配置（配合新实现的验证系统）
args.validation_steps = 100000  # int: ✅ 禁用验证（避免超时问题，专注训练）
args.num_validation_samples = 6  # ✅ 降低到6（减少验证时间）
args.save_validation_videos = False  # ✅ 关闭，节省时间
args.validation_batch_size = 1  # ✅ 降低到1，避免OOM和超时

# ============ Reference VAE Arguments ============
args.use_ref_vae = True  # bool: Whether to use reference VAE
args.ref_vae_dtype = "bfloat16"  # str: Reference VAE data type - H100 optimization
args.cogvideox_model_path = "CogVideoX-2b"  # str: Path to local CogVideoX-2b model

# ============ Logging Arguments ============
args.logging_dir = "logs"  # str: Logging directory
args.log_every = 15  # int: ✅ 改为15（64卡是10，16卡是20）- 48卡适中
args.report_to = "tensorboard"  # str: Reporting platform
args.tracker_name = "taehv_48gpu_h100_cloud"  # ✅ 修改tracker名称标识48卡
args.log_predictions = True  # bool: Log prediction samples

# ============ System Arguments ============
args.allow_tf32 = True  # bool: Allow TF32 - H100 optimization
args.nccl_timeout = 10800  # int: ✅ 3小时超时（16卡是7200，64卡是14400）- 48卡适中
args.device_type = "cuda"  # str: Device type
args.dataloader_pin_memory_device = "cuda"  # str: Pin memory to CUDA

# ============ Environment Arguments ============
# ✅✅✅ 48卡关键配置 - 云平台多机多卡
# 这些参数由云平台环境变量自动设置，无需手动修改
args.world_size = 48  # int: Number of GPUs (6机器×8GPU)
args.rank = 0  # int: Current process rank (由平台RANK环境变量设置)
args.local_rank = 0  # int: Local rank (由平台LOCAL_RANK环境变量设置)
args.master_addr = "localhost"  # str: Master address (由平台MASTER_ADDR环境变量覆盖)
args.master_port = "29500"  # str: Master port (由平台MASTER_PORT环境变量覆盖)

# ============ DeepSpeed Arguments ============
args.use_deepspeed = True  # bool: Whether to use DeepSpeed
args.deepspeed_config_file = "accelerate_configs/deepspeed_48gpu.yaml"  # str: ✅ 48 H100 DeepSpeed config (ZeRO-3)

# ============ H100 Specific Optimizations ============
args.use_flash_attention = True  # bool: flash-attn 2.8.3已安装，启用最佳性能
args.compile_model = True  # bool: torch.compile优化
args.compile_mode = "reduce-overhead"  # str: Compile mode for production
args.enable_fused_adam = True  # bool: Use fused Adam optimizer
args.enable_xformers = True  # bool: Enable xformers memory efficient attention
args.gradient_as_bucket_view = True  # bool: Memory optimization - 48卡重要
args.static_loss_scale = 65536.0  # float: Fixed loss scale for stability

# ============ Production Monitoring ============
args.enable_profiler = False  # bool: Disable profiler in production
args.log_system_stats = True  # bool: Log GPU/CPU/Memory stats
args.early_stopping_patience = 100  # ✅ 保持100
args.metric_for_best_model = "psnr"  # ✅ 使用psnr作为最佳模型指标

# ============ 48-GPU Multi-Node Specific Optimizations ============
# ✅ 48卡多机训练的额外优化
args.use_zero3_save = True  # bool: Use ZeRO-3 model saving (48卡使用ZeRO-3)
args.zero3_save_16bit_model = True  # bool: Save 16bit model with ZeRO-3
args.communication_overlap = True  # bool: 通信计算重叠 - 48卡跨节点重要
args.bucket_size_mb = 750  # int: ✅ 适中bucket size for 48-GPU (750MB)

# ============ Multi-Node Network Optimizations ============
# ✅ 跨节点通信优化
args.find_unused_parameters = False  # bool: 关闭未使用参数检测（提升性能）
args.broadcast_buffers = True  # bool: 广播buffers（保证一致性）
args.ddp_bucket_cap_mb = 750  # int: DDP bucket大小（750MB）
args.gradient_compression = False  # bool: 可选：梯度压缩（视网络带宽而定）

# ============ Data Loading Optimizations for Multi-Node ============
# ✅ 多机数据加载优化
args.dataloader_drop_last = True  # bool: Drop last incomplete batch (48卡避免不均衡)
args.dataloader_timeout = 600  # int: 10分钟超时（跨节点数据加载可能较慢）

# ============ Checkpoint Strategy for Multi-Node ============
# ✅ 多机检查点策略
args.save_on_each_node = False  # bool: 只在主节点保存（避免重复）
args.sync_each_batch = False  # bool: 关闭每batch同步（提升性能）
args.save_optimizer_states = True  # bool: 保存优化器状态（支持完整恢复）

# ============ Advanced Multi-Node Settings ============
# ✅ 高级多机设置
args.use_distributed_sampler = True  # bool: 使用分布式采样器
args.shuffle_data = True  # bool: Shuffle数据
args.dist_backend = "nccl"  # str: 分布式后端（NCCL for GPU）
args.init_method = "env://"  # str: 初始化方法（使用环境变量）

# ============ Monitoring and Debugging ============
# ✅ 监控和调试（可选，调试时启用）
args.monitor_nccl = False  # bool: 监控NCCL通信（调试用）
args.profile_communication = False  # bool: 分析通信性能（调试用）
args.check_nan_inf = False  # bool: 检查NaN/Inf（会降低性能）

# ============ Comments and Notes ============
"""
48-GPU (6×8) Multi-Node Training Configuration for Cloud Platform

关键配置说明：
1. 有效Batch Size = 5×48×1 = 240
   - 介于16卡的128和64卡的256之间
   - 学习率线性插值到1.8e-4
   
2. ZeRO-3必须启用
   - 48卡训练模型参数分片
   - stage3_max_live_parameters设为8e8（适中）
   
3. 通信优化
   - bucket_size设为750MB（适中）
   - NCCL超时设为3小时（适中）
   - 启用communication_overlap
   
4. 检查点策略
   - 适中保存频率（每750步）
   - 只在主节点保存
   
5. 云平台特性
   - 环境变量由平台自动设置
   - WORLD_SIZE=6, RANK=0-5
   - MASTER_ADDR/PORT由平台提供
   
6. 预期性能
   - 训练速度约为16卡的1.4-1.5倍
   - 每步约12-17秒
   - 100K步约22-25小时
"""

