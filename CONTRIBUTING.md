# Contributing to TAEHV Training Project

Thank you for your interest in contributing to the TAEHV Training Project! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Ways to Contribute

- 🐛 **Bug Reports**: Help us identify and fix issues
- 💡 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve docs, guides, and examples
- 🔧 **Code**: Fix bugs, implement features, optimize performance
- 🧪 **Testing**: Add tests, validate on different hardware
- 📊 **Evaluation**: Share training results and model evaluations

### Good First Issues

Look for issues labeled with:
- `good first issue` - Perfect for newcomers
- `help wanted` - We'd love your help
- `documentation` - Improve our docs

## How to Contribute

### 1. Fork the Repository

Click the "Fork" button at the top of the repository page.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/taehv-training.git
cd taehv-training
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

### 4. Make Your Changes

See [Development Setup](#development-setup) for environment setup.

### 5. Test Your Changes

```bash
# Run existing tests
python -m pytest tests/

# Test training pipeline (if applicable)
python training/taehv_train.py --config training/configs/test_config.py

# Test evaluation tools
cd evaluation
python evaluate_vae.py --help
```

### 6. Commit Your Changes

```bash
git add .
git commit -m "feat: add awesome new feature

- Implement feature X
- Fix issue with Y
- Update documentation for Z"
```

Follow [conventional commits](https://www.conventionalcommits.org/) format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for code formatting
- `refactor:` for code refactoring
- `test:` for adding tests
- `chore:` for maintenance tasks

### 7. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 8. Create a Pull Request

1. Go to the original repository
2. Click "New Pull Request"
3. Select your branch
4. Fill out the PR template

## Development Setup

### Environment Setup

```bash
# Create development environment
conda env create -f environment.yml
conda activate tiny-vae

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks (optional)
pre-commit install
```

### Project Structure

```
taehv-training/
├── training/           # Core training logic
├── evaluation/         # Evaluation and testing tools
├── models/            # Model architectures
├── tests/             # Unit and integration tests
├── docs/              # Documentation
└── examples/          # Usage examples
```

### Key Files to Know

- `training/taehv_train.py` - Main training script
- `evaluation/evaluate_vae.py` - Model evaluation
- `models/taehv.py` - TAEHV model implementation
- `training/dataset.py` - Data loading utilities

## Submitting Changes

### Pull Request Guidelines

1. **Clear Description**: Explain what your PR does and why
2. **Link Issues**: Reference related issues with `Closes #123`
3. **Small Changes**: Keep PRs focused and reasonably sized
4. **Tests**: Add tests for new functionality
5. **Documentation**: Update docs if needed

### Review Process

1. **Automated Checks**: CI will run tests and linting
2. **Code Review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Merge**: Once approved, we'll merge your PR

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Use 4 spaces for indentation
# Line length: 100 characters (not 79)
# Use type hints where helpful

def train_model(
    config: TrainingConfig,
    model: torch.nn.Module,
    dataloader: DataLoader
) -> Dict[str, float]:
    """Train the TAEHV model.
    
    Args:
        config: Training configuration
        model: TAEHV model to train
        dataloader: Training data loader
        
    Returns:
        Dict containing training metrics
    """
    # Implementation here
    pass
```

### Code Formatting

We use automated formatting tools:

```bash
# Format code
black . --line-length 100
isort . --profile black

# Check formatting
flake8 . --max-line-length 100
mypy . --ignore-missing-imports
```

### Documentation Style

- Use clear, concise docstrings
- Include type hints
- Add examples for complex functions
- Update README.md for user-facing changes

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests for individual functions
├── integration/       # Integration tests for workflows
├── training/          # Training pipeline tests
└── evaluation/        # Evaluation tool tests
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/unit/
python -m pytest tests/training/

# Run with coverage
python -m pytest tests/ --cov=training --cov=evaluation

# Run tests on specific GPU (if available)
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/training/
```

### Writing Tests

```python
import pytest
import torch
from training.dataset import MiniDataset

def test_dataset_loading():
    """Test that dataset loads correctly."""
    dataset = MiniDataset(
        annotation_file="tests/data/test_annotations.json",
        data_dir="tests/data/videos/",
        patch_hw=64,
        n_frames=8
    )
    
    assert len(dataset) > 0
    
    sample = dataset[0]
    assert sample.shape == (8, 3, 64, 64)  # T, C, H, W
    assert sample.dtype == torch.float32
    assert sample.min() >= 0.0 and sample.max() <= 1.0

@pytest.mark.gpu
def test_model_forward_pass():
    """Test model forward pass on GPU."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")
    
    # Test implementation here
```

## Documentation

### Types of Documentation

1. **Code Comments**: Explain complex logic
2. **Docstrings**: Document functions and classes
3. **README Updates**: User-facing changes
4. **Guides**: New features or workflows
5. **Examples**: Usage examples

### Documentation Guidelines

- Write for your future self and others
- Include code examples
- Test documentation examples
- Update existing docs when making changes

### Building Documentation

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation locally
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Recognition

### Contributors

All contributors will be recognized in:
- `CONTRIBUTORS.md` file
- GitHub contributors page
- Release notes (for significant contributions)

### Types of Contributions We Value

- Code contributions (features, fixes, optimizations)
- Documentation improvements
- Bug reports with detailed information
- Feature requests with clear use cases
- Testing on different hardware configurations
- Performance benchmarks and optimizations
- Example notebooks and tutorials

## Questions?

### Getting Help

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Create an issue for bugs or feature requests
- 📧 **Email**: Contact maintainers for sensitive issues

### Communication Guidelines

- Be respectful and inclusive
- Provide clear, detailed information
- Search existing issues before creating new ones
- Use appropriate labels and templates

---

## Thank You! 🙏

Your contributions help make this project better for everyone. Whether you're fixing a typo, adding a feature, or reporting a bug, every contribution matters!

Happy coding! 🚀
