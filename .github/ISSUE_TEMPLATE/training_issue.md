---
name: Training Issue
about: Report problems related to model training
title: '[TRAINING] '
labels: 'training'
assignees: ''

---

**Training Problem Description**
Clearly describe the training issue you're experiencing.

**Training Configuration**
```python
# Paste your complete training configuration here
# training/configs/taehv_config_h100.py
```

**Training Command**
```bash
# The exact command you used to start training
```

**Training Logs**
```
# Paste relevant training logs, especially error messages
# Include at least the last 50 lines before the error
```

**Evaluation Results (if applicable)**
If you completed training, what were your evaluation results?
- PSNR: [e.g. 4.2 dB - too low]
- SSIM: [e.g. 0.01 - too low]  
- LPIPS: [e.g. 0.8 - too high]

**Hardware Setup**
- GPU: [e.g. 8x H100, RTX 4090]
- VRAM per GPU: [e.g. 80GB, 24GB]
- System RAM: [e.g. 512GB]
- Storage: [e.g. NVMe SSD, HDD]

**Dataset Information**
- Dataset size: [e.g. 500 videos]
- Video resolution: [e.g. 720x480]
- Video length: [e.g. 12 frames]
- Data format: [e.g. MP4, AVI]

**Training Progress**
- How many steps did you train? [e.g. 100000]
- How long did training take? [e.g. 6 hours]
- Did loss decrease? [Yes/No]
- At what step did the problem occur?

**Expected Behavior**
What training results were you expecting?

**Additional Context**
- Is this your first time training this model?
- Have you tried different configurations?
- Any other relevant information

**Checklist**
- [ ] I have read the troubleshooting guide
- [ ] I have checked my loss weights configuration
- [ ] I have verified my data paths are correct
- [ ] I have tried the recommended configuration settings
