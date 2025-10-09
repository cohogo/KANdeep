# KAN Module - Kolmogorov-Arnold Networks

A PyTorch implementation of Kolmogorov-Arnold Networks (KAN) with both classic and optimized versions, including Vision Transformer integration.

## Features

- **Classic KAN**: Traditional implementation with B-spline basis functions (`ekan.py`)
- **FasterKAN**: Optimized implementation for better performance (`fasterkan.py`)
- **Vision KAN**: Integration with Vision Transformers using Mixture of Experts (`vision_kan.py`)

## Installation

### From Source
```bash
git clone <repository-url>
cd kan_module
pip install -e .
```

### With Optional Dependencies
```bash
# For Vision KAN functionality
pip install -e .[vision]

# For development and examples
pip install -e .[dev]

# Install all dependencies
pip install -e .[all]
```

## Quick Start

### Basic KAN Usage

```python
import torch
from kan_module import KAN, FasterKAN

# Classic KAN
model = KAN([2, 5, 1])  # 2 inputs, 5 hidden, 1 output
x = torch.randn(100, 2)
y = model(x)

# Faster KAN
model = FasterKAN([2, 5, 1])
x = torch.randn(100, 2)
y = model(x)
```

### Vision KAN Usage

```python
from kan_module import create_kan

# Create a Vision KAN model
model = create_kan('deit_tiny_patch16_224_KAN', pretrained=False, num_classes=10)
x = torch.randn(1, 3, 224, 224)
y = model(x)
```

### MoE KAN MLP Usage

```python
from kan_module import MoE_KAN_MLP

# Create MoE KAN MLP
moe_kan = MoE_KAN_MLP(
    hidden_dim=768,
    ffn_dim=3072,
    num_experts=8,
    top_k=2
)

x = torch.randn(197, 768)  # [seq_len, hidden_dim]
y = moe_kan(x)
```

## Available Models

### Classic KAN Models
- `KAN`: Multi-layer KAN with B-spline basis functions
- `KANLinear`: Single KAN layer

### Faster KAN Models
- `FasterKAN`: Optimized multi-layer KAN
- `FasterKANLayer`: Single optimized KAN layer

### Vision KAN Models
- `deit_tiny_patch16_224_KAN`
- `deit_small_patch16_224_KAN`
- `deit_base_patch16_224_KAN`
- `deit_base_patch16_384_KAN`

## API Reference

### KAN Class
```python
KAN(
    layers_hidden,          # List of layer dimensions
    grid_size=5,           # Grid size for B-splines
    spline_order=3,        # Order of B-splines
    scale_noise=0.1,       # Noise scale for initialization
    scale_base=1.0,        # Base activation scale
    scale_spline=1.0,      # Spline activation scale
    base_activation=torch.nn.SiLU,  # Base activation function
    grid_eps=0.02,         # Grid epsilon
    grid_range=[-1, 1],    # Grid range
)
```

### FasterKAN Class
```python
FasterKAN(
    layers_hidden,              # List of layer dimensions
    grid_min=-2.0,             # Minimum grid value
    grid_max=2.0,              # Maximum grid value
    num_grids=8,               # Number of grid points
    exponent=2,                # Exponent for basis functions
    denominator=0.33,          # Denominator for smoothness
    use_base_update=True,      # Whether to use base updates
    base_activation=F.silu,    # Base activation function
    spline_weight_init_scale=0.667,  # Spline weight initialization scale
)
```

### MoE_KAN_MLP Class
```python
MoE_KAN_MLP(
    ffn_dim,        # Feed-forward dimension
    hidden_dim,     # Hidden dimension
    num_experts=8,  # Number of experts
    top_k=2,        # Top-k experts to use
)
```

## Examples

See the `examples/` directory for detailed usage examples:
- `basic_kan_example.py`: Basic KAN usage
- `faster_kan_example.py`: FasterKAN usage
- `vision_kan_example.py`: Vision KAN usage

## Requirements

- Python >= 3.7
- PyTorch >= 1.13.1
- torchvision >= 0.8.1
- timm >= 0.3.2 (for Vision KAN functionality)

## License

MIT License

## Citation

If you use this code in your research, please cite the original KAN paper:

```bibtex
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Soljačić, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```