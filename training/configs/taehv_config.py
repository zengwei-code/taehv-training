from easydict import EasyDict
import datetime
import pytz
import os

args = EasyDict()

# ============ Model Arguments ============
args.pretrained_model_path = "checkpoints/taecvx.pth"  # str: Path to TAEHV pretrained checkpoint
args.patch_size = 1  # int: TAEHV patch size
args.latent_channels = 16  # int: Number of latent channels
args.model_dtype = "float16"  # str: Model data type

# ============ Dataset Arguments ============
args.data_root = "/data/matrix-project/MiniDataset/data"  # str: Training data folder
args.annotation_file = "/data/matrix-project/MiniDataset/stage1_annotations_500.json"  # str: Annotation file
args.train_split = 0.9  # float: Training data split ratio
args.dataloader_num_workers = 8  # int: Number of dataloader workers
args.pin_memory = True  # bool: Whether to use pinned memory
args.prefetch_factor = 2  # int: Prefetch factor for dataloader

# ============ Video Processing Arguments ============
args.height = 128  # int: Video height (patch size)
args.width = 128  # int: Video width (patch size)
args.n_frames = 12  # int: Number of frames per sample
args.min_frame_delta = 1  # int: Minimum frame interval
args.max_frame_delta = 3  # int: Maximum frame interval

# ============ Data Augmentation Arguments ============
args.random_crop = True  # bool: Whether to use random crop
args.random_flip = True  # bool: Whether to use random flip
args.color_jitter = True  # bool: Whether to use color jitter
args.color_jitter_params = {
    'brightness': 0.1,
    'contrast': 0.1,
    'saturation': 0.1,
    'hue': 0.05
}

# ============ Training Arguments ============
args.seed = 42  # int: Random seed
args.mixed_precision = "bf16"  # str: Mixed precision type
args.output_dir = "output/" + datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d_%H-%M-%S")  # str: Output directory
args.train_batch_size = 2  # int: Batch size per device (reduced from 4 to avoid NCCL timeout)
args.gradient_accumulation_steps = 2  # int: Gradient accumulation steps (increased to maintain effective batch size)
args.gradient_checkpointing = True  # bool: Whether to use gradient checkpointing
args.max_train_steps = 1000  # int: Maximum training steps
args.checkpointing_steps = 500  # int: Save checkpoint every N steps
args.checkpoints_total_limit = 5  # int: Maximum number of checkpoints to keep
args.resume_from_checkpoint = None  # str: Resume from checkpoint path

# ============ Learning Rate Arguments ============
args.learning_rate = 3e-4  # float: Learning rate
args.lr_scheduler = "cosine"  # str: LR scheduler type
args.lr_warmup_steps = 1000  # int: Warmup steps
args.lr_num_cycles = 1  # int: Number of cycles for cosine scheduler

# ============ Optimizer Arguments ============
args.optimizer = "adamw"  # str: Optimizer type
args.beta1 = 0.9  # float: Adam beta1
args.beta2 = 0.95  # float: Adam beta2
args.weight_decay = 1e-4  # float: Weight decay
args.epsilon = 1e-8  # float: Adam epsilon
args.max_grad_norm = 1.0  # float: Max gradient norm

# ============ Loss Function Arguments ============
args.encoder_loss_weight = 1.0  # float: Encoder loss weight
args.decoder_loss_weight = 1.0  # float: Decoder loss weight
args.reconstruction_loss_weight = 0.1  # float: Reconstruction loss weight

# ============ Seraena Arguments ============
args.use_seraena = True  # bool: Whether to use Seraena adversarial training
args.seraena_loss_weight = 0.5  # float: Seraena loss weight
args.n_seraena_frames = 3  # int: Number of frames to group for Seraena

# ============ Validation Arguments ============
args.validation_steps = 500  # int: Run validation every N steps
args.num_validation_samples = 5  # int: Number of validation samples
args.save_validation_videos = True  # bool: Whether to save validation videos

# ============ Reference VAE Arguments ============
args.use_ref_vae = True  # bool: Whether to use reference VAE (set to False if you don't want to use CogVideoX-2b)
args.ref_vae_dtype = "bfloat16"  # str: Reference VAE data type
args.cogvideox_model_path = "CogVideoX-2b"  # str: Path to local CogVideoX-2b model (relative to training directory)

# ============ Logging Arguments ============
args.logging_dir = "logs"  # str: Logging directory
args.log_every = 50  # int: Log every N steps
args.report_to = "tensorboard"  # str: Reporting platform
args.tracker_name = "taehv_training"  # str: Tracker name

# ============ System Arguments ============
args.allow_tf32 = True  # bool: Allow TF32
args.nccl_timeout = 1800  # int: NCCL timeout
args.device_type = "cuda"  # str: Device type

# ============ Environment Arguments ============
# These will be set by the launch script
args.world_size = 8  # int: Number of GPUs
args.rank = 0  # int: Current process rank
args.local_rank = 0  # int: Local rank
args.master_addr = "localhost"  # str: Master address
args.master_port = "8000"  # str: Master port

# ============ DeepSpeed Arguments ============
args.use_deepspeed = True  # bool: Whether to use DeepSpeed
args.deepspeed_config_file = "accelerate_configs/deepspeed.yaml"  # str: DeepSpeed config file
