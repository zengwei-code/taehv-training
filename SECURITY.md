# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| v1.x.x  | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of the TAEHV Training Project seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **DO NOT** create a public GitHub issue for security vulnerabilities
2. **Email** us directly at: [INSERT YOUR EMAIL]
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Possible impact
   - Suggested fix (if you have one)

### What to Expect

- **Acknowledgment**: We'll acknowledge receipt of your report within 48 hours
- **Investigation**: We'll investigate and validate the vulnerability
- **Timeline**: We aim to provide an initial assessment within 5 business days
- **Updates**: We'll keep you informed of our progress
- **Credit**: We'll credit you in our security advisories (unless you prefer anonymity)

### Security Best Practices

When using this project, please follow these security guidelines:

#### Model Weights and Checkpoints
- **Never share trained models** containing sensitive data
- **Verify checkpoints** before loading from untrusted sources
- **Use `weights_only=True`** when loading PyTorch models:
  ```python
  checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
  ```

#### Data Security
- **Protect training data** - ensure your video datasets don't contain sensitive information
- **Secure data paths** - use absolute paths and proper file permissions
- **Data anonymization** - remove or anonymize sensitive content before training

#### Environment Security
- **Use virtual environments** to isolate dependencies
- **Keep dependencies updated** regularly run `pip install --upgrade -r requirements.txt`
- **Monitor GPU access** - ensure only authorized users can access training GPUs

#### Configuration Security
- **No hardcoded credentials** - never commit API keys, passwords, or tokens
- **Environment variables** - use environment variables for sensitive configuration
- **Config validation** - validate all configuration inputs

#### Network Security
- **TensorBoard access** - restrict TensorBoard ports to localhost or VPN
- **Model serving** - secure any model serving endpoints
- **Data transfer** - use encrypted connections for data transfer

### Common Security Issues

#### Pickle Security
This project uses PyTorch which relies on pickle for model serialization. Be aware:

- **Only load trusted models** from verified sources
- **Use `weights_only=True`** when possible
- **Validate model architecture** before loading weights

#### Code Injection
When using configuration files:

- **Validate inputs** - sanitize all user inputs
- **Restrict imports** - limit dynamic imports in config files
- **Use safe evaluation** - avoid `eval()` or `exec()` on user data

#### Resource Exhaustion
Training can consume significant resources:

- **Memory limits** - set appropriate memory limits
- **Disk space** - monitor disk usage for large models
- **GPU timeout** - implement training timeouts

### Security Updates

We will:
- Release security patches as soon as possible
- Notify users through GitHub releases and security advisories
- Provide migration guides for breaking security fixes
- Maintain a security changelog

### Responsible Disclosure

We follow responsible disclosure practices:

1. **Private reporting** - vulnerabilities are reported privately first
2. **Coordinated disclosure** - we work with reporters to coordinate public disclosure
3. **Credit** - we credit security researchers who report vulnerabilities
4. **Timeline** - we aim for disclosure within 90 days of the initial report

### Hall of Fame

We thank the following security researchers for their contributions:

<!-- This section will be updated as we receive security reports -->
- No security vulnerabilities have been reported yet

### Contact

For security-related questions or concerns:
- **Security issues**: [INSERT SECURITY EMAIL]
- **General questions**: Create a GitHub issue
- **Private matters**: [INSERT CONTACT EMAIL]

---

Thank you for helping keep the TAEHV Training Project secure! 🔒
