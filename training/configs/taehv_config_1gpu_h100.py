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
# 单卡测试配置 - 使用MiniDataset快速验证
args.data_root = "/data/matrix-project/MiniDataset/data"  # str: Test data folder for quick iteration
args.annotation_file = "/data/matrix-project/MiniDataset/stage1_annotations_500.json"  # str: Small test annotation
args.train_split = 0.95  # float: Training data split ratio (more data for training)
args.dataloader_num_workers = 8  # int: Moderate workers for single GPU
args.pin_memory = True  # bool: Whether to use pinned memory
args.prefetch_factor = 4  # int: Increased prefetch for faster loading
args.persistent_workers = True  # bool: Keep workers alive between epochs

# ============ Video Processing Arguments ============
# 单卡H100配置 - 利用80GB显存优势使用高分辨率
args.height = 480  # int: Ultra high resolution for H100
args.width = 720  # int: Ultra high resolution for H100
args.n_frames = 20  # int: Frames for high resolution (20-3=17 target frames, ceil(20/4)*4-3=17)
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
args.train_batch_size = 1  # int: Reduced for 720x480x19 frames on single GPU
args.gradient_accumulation_steps = 16  # int: Increased to maintain effective batch size of 16
args.gradient_checkpointing = True  # bool: Whether to use gradient checkpointing
args.max_train_steps = 500  # int: Short training for quick testing
args.checkpointing_steps = 500  # int: Save checkpoint every N steps
args.checkpoints_total_limit = 5  # int: Keep limited checkpoints for testing
args.resume_from_checkpoint = None  # str: Resume from checkpoint path
args.save_model_card = True  # bool: Save model card for production

# ============ Learning Rate Arguments ============
# 单卡测试学习率配置
args.learning_rate = 5e-4  # float: Higher LR for faster convergence in testing
args.lr_scheduler = "cosine_with_restarts"  # str: Better scheduler for long training
args.lr_warmup_steps = 100  # int: Short warmup for quick testing
args.lr_num_cycles = 2  # int: Multiple cycles for testing
args.lr_power = 1.0  # float: Power for polynomial scheduler

# ============ Optimizer Arguments ============
args.optimizer = "adamw"  # str: Optimizer type
args.beta1 = 0.9  # float: Adam beta1
args.beta2 = 0.95  # float: Adam beta2
args.weight_decay = 1e-2  # float: Increased weight decay for regularization
args.epsilon = 1e-8  # float: Adam epsilon
args.max_grad_norm = 1.0  # float: Max gradient norm

# ============ Loss Function Arguments ============
args.encoder_loss_weight = 1.0  # float: Encoder loss weight
args.decoder_loss_weight = 1.0  # float: Decoder loss weight
args.reconstruction_loss_weight = 0.1  # float: Reconstruction loss weight
args.perceptual_loss_weight = 0.05  # float: Add perceptual loss for better quality

# ============ Seraena Arguments ============
args.use_seraena = True  # bool: Whether to use Seraena adversarial training
args.seraena_loss_weight = 0.3  # float: Reduced for stability in long training
args.n_seraena_frames = 1  # int: Seraena frames (17 target frames ÷ 1 = 17, avoiding data loss)

# ============ Validation Arguments ============
args.validation_steps = 500  # int: Frequent validation for testing
args.num_validation_samples = 4  # int: Limited validation samples for speed
args.save_validation_videos = True  # bool: Whether to save validation videos
args.validation_batch_size = 2  # int: Separate validation batch size

# ============ Reference VAE Arguments ============
args.use_ref_vae = True  # bool: Whether to use reference VAE
args.ref_vae_dtype = "bfloat16"  # str: Reference VAE data type - H100 optimization
args.cogvideox_model_path = "CogVideoX-2b"  # str: Path to local CogVideoX-2b model

# ============ Logging Arguments ============
args.logging_dir = "logs"  # str: Logging directory
args.log_every = 10  # int: Frequent logging for testing
args.report_to = "tensorboard"  # str: Reporting platform
args.tracker_name = "taehv_1gpu_h100_test"  # str: Tracker name for single GPU testing
args.log_predictions = True  # bool: Log prediction samples

# ============ System Arguments ============
args.allow_tf32 = True  # bool: Allow TF32 - H100 optimization
args.nccl_timeout = 1800  # int: Timeout for single GPU (shorter than multi-GPU)
args.device_type = "cuda"  # str: Device type
args.dataloader_pin_memory_device = "cuda"  # str: Pin memory to CUDA

# ============ Environment Arguments ============
# 单卡训练环境配置
args.world_size = 1  # int: Single GPU
args.rank = 0  # int: Current process rank
args.local_rank = 0  # int: Local rank
args.master_addr = "localhost"  # str: Master address
args.master_port = "29500"  # str: Master port (changed from 8000)

# ============ DeepSpeed Arguments ============
args.use_deepspeed = True  # bool: Whether to use DeepSpeed
args.deepspeed_config_file = "accelerate_configs/deepspeed_1gpu.yaml"  # str: DeepSpeed config file for single H100

# ============ H100 Specific Optimizations ============
args.use_flash_attention = True  # bool: flash-attn 2.8.3已安装，启用最佳性能
args.compile_model = False  # bool: torch.compile可能有兼容性问题，暂时关闭
args.compile_mode = "reduce-overhead"  # str: Compile mode for production
args.enable_fused_adam = True  # bool: Use fused Adam optimizer
args.enable_xformers = True  # bool: Enable xformers memory efficient attention
args.gradient_as_bucket_view = True  # bool: Memory optimization
args.static_loss_scale = 65536.0  # float: Fixed loss scale for stability

# ============ Production Monitoring ============
args.enable_profiler = False  # bool: Disable profiler in production
args.log_system_stats = True  # bool: Log GPU/CPU/Memory stats
args.early_stopping_patience = 5000  # int: Early stopping patience (shorter for testing)
args.metric_for_best_model = "reconstruction_loss"  # str: Metric for model selection
