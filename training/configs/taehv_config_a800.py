from easydict import EasyDict
import datetime
import pytz
import os

args = EasyDict()

# ============ Model Arguments ============
args.pretrained_model_path = "checkpoints/taecvx.pth"  # str: Path to TAEHV pretrained checkpoint
args.patch_size = 1  # int: TAEHV patch size
args.latent_channels = 16  # int: Number of latent channels
args.model_dtype = "bfloat16"  # str: Model data type - A800 support

# ============ Dataset Arguments ============
# A800测试配置，使用MiniDataset
args.data_root = "/data/matrix-project/MiniDataset/data"  # str: MiniDataset for A800 testing
args.annotation_file = "/data/matrix-project/MiniDataset/stage1_annotations_500.json"  # str: MiniDataset annotation
args.train_split = 0.95  # float: Training data split ratio
args.dataloader_num_workers = 4  # int: 减少workers数量节省内存
args.pin_memory = True  # bool: Whether to use pinned memory
args.prefetch_factor = 2  # int: 减少预取节省内存
args.persistent_workers = True  # bool: Keep workers alive between epochs

# ============ Video Processing Arguments ============
# A800显存优化设置 - 降低分辨率避免OOM
args.height = 256  # int: A800适中分辨率 (从480降低)
args.width = 384  # int: A800适中分辨率 (从720降低)  
args.n_frames = 11  # int: 确保frames_target = 11-3 = 8帧 (考虑frames_to_trim)
args.min_frame_delta = 1  # int: Minimum frame interval
args.max_frame_delta = 2  # int: Frame interval

# ============ Data Augmentation Arguments ============
args.random_crop = True  # bool: Whether to use random crop
args.random_flip = True  # bool: Whether to use random flip
args.color_jitter = True  # bool: Whether to use color jitter
args.color_jitter_params = {
    'brightness': 0.1,   # 相比H100更保守
    'contrast': 0.1,
    'saturation': 0.1,
    'hue': 0.05
}
args.gaussian_blur = False  # bool: A800关闭高级增强以节省计算

# ============ Training Arguments ============
args.seed = 42  # int: Random seed
args.mixed_precision = "bf16"  # str: A800 bf16 support (可能不如H100高效)
args.output_dir = "output/a800_test_" + datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")
args.train_batch_size = 1  # int: A800最小batch size
args.gradient_accumulation_steps = 4  # int: 减少累积步数，降低显存峰值
args.gradient_checkpointing = True  # bool: Whether to use gradient checkpointing
args.max_train_steps = 1000  # int: A800测试用的短训练步数
args.checkpointing_steps = 500  # int: 更频繁的检查点
args.checkpoints_total_limit = 5  # int: 保持少量检查点
args.resume_from_checkpoint = None  # str: Resume from checkpoint path
args.save_model_card = True  # bool: Save model card

# ============ Learning Rate Arguments ============
# A800保守的学习率设置
args.learning_rate = 3e-5  # float: A800保守LR
args.lr_scheduler = "cosine"  # str: 简单调度器
args.lr_warmup_steps = 200  # int: 短warmup适应测试
args.lr_num_cycles = 1  # int: 单周期测试
args.lr_power = 1.0  # float: Power for polynomial scheduler

# ============ Optimizer Arguments ============
args.optimizer = "adamw"  # str: Optimizer type
args.beta1 = 0.9  # float: Adam beta1
args.beta2 = 0.95  # float: Adam beta2
args.weight_decay = 1e-2  # float: Weight decay
args.epsilon = 1e-8  # float: Adam epsilon
args.max_grad_norm = 1.0  # float: Max gradient norm

# ============ Loss Function Arguments ============
args.encoder_loss_weight = 1.0  # float: Encoder loss weight
args.decoder_loss_weight = 1.0  # float: Decoder loss weight
args.reconstruction_loss_weight = 0.1  # float: Reconstruction loss weight
args.perceptual_loss_weight = 0.02  # float: 降低感知损失权重

# ============ Seraena Arguments ============
args.use_seraena = False  # bool: 关闭Seraena节省显存
args.seraena_loss_weight = 0.0  # float: 关闭Seraena
args.n_seraena_frames = 0  # int: 关闭Seraena

# ============ Validation Arguments ============
args.validation_steps = 1000  # int: 减少验证频率节省显存
args.num_validation_samples = 2  # int: 最少验证样本
args.save_validation_videos = False  # bool: 关闭视频保存节省显存
args.validation_batch_size = 1  # int: A800保守的验证batch size

# ============ Reference VAE Arguments ============
args.use_ref_vae = False  # bool: 关闭参考VAE节省显存
args.ref_vae_dtype = "bfloat16"  # str: Reference VAE data type
args.cogvideox_model_path = "CogVideoX-2b"  # str: Path to local CogVideoX-2b model

# ============ Logging Arguments ============
args.logging_dir = "logs"  # str: Logging directory
args.log_every = 50  # int: Logging frequency
args.report_to = "tensorboard"  # str: Reporting platform
args.tracker_name = "taehv_a800_test"  # str: A800测试标识
args.log_predictions = True  # bool: Log prediction samples

# ============ System Arguments ============
args.allow_tf32 = True  # bool: Allow TF32
args.nccl_timeout = 1800  # int: A800保守的超时设置
args.device_type = "cuda"  # str: Device type
args.dataloader_pin_memory_device = "cuda"  # str: Pin memory to CUDA

# ============ Environment Arguments ============
args.world_size = 8  # int: Number of GPUs (A800 x8)
args.rank = 0  # int: Current process rank
args.local_rank = 0  # int: Local rank
args.master_addr = "localhost"  # str: Master address
args.master_port = "29500"  # str: Master port

# ============ DeepSpeed Arguments ============
args.use_deepspeed = True  # bool: Whether to use DeepSpeed
args.deepspeed_config_file = "accelerate_configs/deepspeed_8gpu.yaml"  # str: 与H100共享DeepSpeed配置

# ============ A800 Specific Settings ============
args.use_flash_attention = True  # bool: A800 (8.0架构) 完全支持flash-attn 2.8.3
args.compile_model = False  # bool: A800关闭torch.compile避免问题  
args.compile_mode = "default"  # str: Default compile mode
args.enable_fused_adam = False  # bool: A800保守设置，关闭fused Adam
args.gradient_as_bucket_view = True  # bool: Memory optimization
args.static_loss_scale = 32768.0  # float: A800更保守的loss scale

# ============ Testing Configuration ============
args.enable_profiler = True  # bool: A800测试时启用profiler分析性能
args.log_system_stats = True  # bool: Log GPU/CPU/Memory stats
args.early_stopping_patience = 1000  # int: 测试用的早停
args.metric_for_best_model = "reconstruction_loss"  # str: Metric for model selection
