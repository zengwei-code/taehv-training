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
args.dataloader_num_workers = 10  # int: ✅ 降低到8（避免I/O瓶颈导致超时）
args.pin_memory = True  # bool: Whether to use pinned memory
args.prefetch_factor = 3  # int: ✅ 降低到2（避免内存压力和超时）
args.persistent_workers = True  # bool: Keep workers alive between epochs

# ============ Video Processing Arguments ============
# 针对H100大显存，提升视频质量
args.height = 480  # int: Ultra high resolution for H100
args.width = 720  # int: Ultra high resolution for H100
args.n_frames = 12  # int: Reduced frames for high resolution (720x480) 降低帧数以节省内存（关键优化！）
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
args.train_batch_size = 8  # int: ✅ 保持8（每卡），16卡总batch=8×16×1=128
args.gradient_accumulation_steps = 1  # int: ✅ 16卡时不需要累积（8卡是2）
args.gradient_checkpointing = True  # bool: Whether to use gradient checkpointing
args.max_train_steps = 100000  # int: Extended training for production
args.checkpointing_steps = 1000  # int: ✅ 改为1000（8卡是2000）- 16卡训练更快
args.checkpoints_total_limit = 100  # int: Keep more checkpoints for production
args.resume_from_checkpoint = None  # str: Resume from checkpoint path
args.save_model_card = True  # bool: Save model card for production

# ============ Learning Rate Arguments ============
# ✅ 16卡训练：有效batch size = 8×16×1 = 128（8卡是8×8×2=128）
# 有效batch size相同，学习率保持不变
args.learning_rate = 1e-4  # float: 与8卡保持一致
args.lr_scheduler = "cosine"  # str: Better scheduler for long training
args.lr_warmup_steps = 1500  # int: ✅ 稍微缩短（8卡是2000）- 16卡收敛更快
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
args.num_validation_samples = 8  # ✅ 降低到8（减少验证时间）
args.save_validation_videos = False  # ✅ 关闭，节省时间
args.validation_batch_size = 1  # ✅ 降低到1，避免OOM和超时

# ============ Reference VAE Arguments ============
args.use_ref_vae = True  # bool: Whether to use reference VAE
args.ref_vae_dtype = "bfloat16"  # str: Reference VAE data type - H100 optimization
args.cogvideox_model_path = "CogVideoX-2b"  # str: Path to local CogVideoX-2b model

# ============ Logging Arguments ============
args.logging_dir = "logs"  # str: Logging directory
args.log_every = 20  # int: ✅ 改为20（8卡是25）- 16卡更频繁日志
args.report_to = "tensorboard"  # str: Reporting platform
args.tracker_name = "taehv_16gpu_h100_production"  # ✅ 修改tracker名称标识16卡
args.log_predictions = True  # bool: Log prediction samples

# ============ System Arguments ============
args.allow_tf32 = True  # bool: Allow TF32 - H100 optimization
args.nccl_timeout = 7200  # int: ✅ 增加超时（8卡是3600）- 16卡通信更复杂
args.device_type = "cuda"  # str: Device type
args.dataloader_pin_memory_device = "cuda"  # str: Pin memory to CUDA

# ============ Environment Arguments ============
# ✅✅✅ 16卡关键配置
args.world_size = 16  # int: ✅ Number of GPUs (H100 x16)
args.rank = 0  # int: Current process rank
args.local_rank = 0  # int: Local rank
args.master_addr = "localhost"  # str: Master address
args.master_port = "29500"  # str: Master port

# ============ DeepSpeed Arguments ============
args.use_deepspeed = True  # bool: Whether to use DeepSpeed
args.deepspeed_config_file = "accelerate_configs/deepspeed_16gpu.yaml"  # str: ✅ 16 H100 DeepSpeed config (ZeRO-3)

# ============ H100 Specific Optimizations ============
args.use_flash_attention = True  # bool: flash-attn 2.8.3已安装，启用最佳性能
args.compile_model = True  # bool: torch.compile优化
args.compile_mode = "reduce-overhead"  # str: Compile mode for production
args.enable_fused_adam = True  # bool: Use fused Adam optimizer
args.enable_xformers = True  # bool: Enable xformers memory efficient attention
args.gradient_as_bucket_view = True  # bool: Memory optimization - 16卡特别重要
args.static_loss_scale = 65536.0  # float: Fixed loss scale for stability

# ============ Production Monitoring ============
args.enable_profiler = False  # bool: Disable profiler in production
args.log_system_stats = True  # bool: Log GPU/CPU/Memory stats
args.early_stopping_patience = 100  # ✅ 保持100
args.metric_for_best_model = "psnr"  # ✅ 使用psnr作为最佳模型指标

# ============ 16-GPU Specific Optimizations ============
# ✅ 16卡训练的额外优化
args.use_zero3_save = True  # bool: Use ZeRO-3 model saving (16卡使用ZeRO-3)
args.zero3_save_16bit_model = True  # bool: Save 16bit model with ZeRO-3
args.communication_overlap = True  # bool: 通信计算重叠 - 16卡特别重要
args.bucket_size_mb = 500  # int: 更大的bucket size for 16-GPU (MB)


