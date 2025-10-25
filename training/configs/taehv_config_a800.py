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
args.data_root = "/data/matrix-project/MiniDataset/data"  # str: Training data folder
args.annotation_file = "/data/matrix-project/MiniDataset/stage1_annotations_500.json"  # str: Annotation file
args.train_split = 0.95  # float: Training data split ratio
args.dataloader_num_workers = 8  # int: Number of dataloader workers
args.pin_memory = True  # bool: Whether to use pinned memory
args.prefetch_factor = 2  # int: Prefetch factor for faster loading
args.persistent_workers = True  # bool: Keep workers alive between epochs

# ============ Video Processing Arguments ============
args.height = 480  # int: Video frame height
args.width = 720  # int: Video frame width
args.n_frames = 12  # int: Number of frames per video sample
args.min_frame_delta = 1  # int: Minimum frame interval
args.max_frame_delta = 2  # int: Frame interval

# ============ Data Augmentation Arguments ============
args.random_crop = True  # bool: Whether to use random crop
args.random_flip = True  # bool: Whether to use random flip
args.color_jitter = True  # bool: Whether to use color jitter
args.color_jitter_params = {
    'brightness': 0.15,
    'contrast': 0.15,
    'saturation': 0.15,
    'hue': 0.08
}
args.gaussian_blur = True  # bool: Add Gaussian blur augmentation
args.gaussian_blur_sigma = (0.1, 2.0)  # tuple: Blur sigma range

# ============ Training Arguments ============
args.seed = 42  # int: Random seed
args.mixed_precision = "bf16"  # str: Mixed precision training mode
args.output_dir = "output/a800_" + datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")
args.train_batch_size = 2  # int: Training batch size per device
args.gradient_accumulation_steps = 8  # int: Gradient accumulation steps
args.gradient_checkpointing = True  # bool: Whether to use gradient checkpointing
args.max_train_steps = 2000  # int: Maximum training steps
args.checkpointing_steps = 1000  # int: Save checkpoint frequency
args.checkpoints_total_limit = 5  # int: Maximum number of checkpoints to keep
args.resume_from_checkpoint = None  # str: Resume from checkpoint path
args.save_model_card = True  # bool: Save model card

# ============ Learning Rate Arguments ============
args.learning_rate = 1e-4  # float: Initial learning rate
args.lr_scheduler = "cosine"  # str: Learning rate scheduler type (cosine更稳定)
args.lr_warmup_steps = 200  # int: Number of warmup steps (减少warmup，更快进入有效训练)
args.lr_num_cycles = 1  # int: Number of cosine cycles (单周期，避免过度衰减)
args.lr_power = 1.0  # float: Power for polynomial scheduler

# ============ Optimizer Arguments ============
args.optimizer = "adamw"  # str: Optimizer type
args.beta1 = 0.9  # float: Adam beta1
args.beta2 = 0.95  # float: Adam beta2
args.weight_decay = 1e-2  # float: Weight decay
args.epsilon = 1e-8  # float: Adam epsilon
args.max_grad_norm = 1.0  # float: Max gradient norm

# ============ Loss Function Arguments ============
args.reconstruction_loss_weight = 10.0  # float: Reconstruction loss weight
args.encoder_loss_weight = 0.1          # float: Encoder regularization weight
args.decoder_loss_weight = 0.1          # float: Decoder regularization weight
args.perceptual_loss_weight = 0.1       # float: Perceptual loss weight

# ============ Seraena Arguments ============
args.use_seraena = False  # bool: Whether to use Seraena
args.seraena_loss_weight = 0.0  # float: Seraena loss weight
args.n_seraena_frames = 0  # int: Number of Seraena frames

# ============ Validation Arguments ============
args.validation_steps = 1000  # int: Validation frequency
args.num_validation_samples = 20  # int: Number of validation samples
args.save_validation_videos = False  # bool: Whether to save validation videos
args.validation_batch_size = 2  # int: Validation batch size

# ============ Reference VAE Arguments ============
args.use_ref_vae = True  # bool: Whether to use reference VAE
args.ref_vae_dtype = "bfloat16"  # str: Reference VAE data type
args.cogvideox_model_path = "CogVideoX-2b"  # str: Path to local CogVideoX-2b model

# ============ Logging Arguments ============
args.logging_dir = "logs"  # str: Logging directory
args.log_every = 25  # int: Logging frequency
args.report_to = "tensorboard"  # str: Reporting platform
args.tracker_name = "taehv_a800"  # str: Tracker name
args.log_predictions = True  # bool: Log prediction samples

# ============ System Arguments ============
args.allow_tf32 = True  # bool: Allow TF32
args.nccl_timeout = 3600  # int: NCCL timeout
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
args.deepspeed_config_file = "accelerate_configs/deepspeed_8gpu.yaml"  # str: DeepSpeed configuration file

# ============ A800 Specific Settings ============
args.use_flash_attention = True  # bool: Use FlashAttention
args.compile_model = False  # bool: Use torch.compile
args.compile_mode = "reduce-overhead"  # str: Compile mode
args.enable_fused_adam = True  # bool: Use fused Adam optimizer
args.enable_xformers = True  # bool: Enable xformers memory efficient attention
args.gradient_as_bucket_view = True  # bool: Memory optimization
args.static_loss_scale = 65536.0  # float: Fixed loss scale for stability

# ============ Monitoring and Early Stopping ============
args.enable_profiler = False  # bool: Enable profiler
args.log_system_stats = True  # bool: Log GPU/CPU/Memory stats
args.early_stopping_patience = 10  # int: Early stopping patience (增加耐心，避免过早停止)
args.metric_for_best_model = "psnr"  # str: Metric for best model selection

