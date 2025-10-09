"""
KAN (Kolmogorov-Arnold Networks) Module

This module provides implementations of Kolmogorov-Arnold Networks (KAN),
including both classic and optimized versions.

Classes:
    KAN: Classic KAN implementation with B-spline basis functions
    KANLinear: Linear layer with KAN activation
    FasterKAN: Optimized KAN implementation for better performance
    FasterKANLayer: Single layer of the optimized KAN
    VisionKAN: KAN integrated with Vision Transformer architecture
"""

from .ekan import KAN, KANLinear
from .fasterkan import FasterKAN, FasterKANLayer, SplineLinear, ReflectionalSwitchFunction
from .vision_kan import VisionKAN, MoE_KAN_MLP, kanBlock, create_kan

__version__ = "1.0.0"
__author__ = "KAN Module Team"

__all__ = [
    # Classic KAN
    'KAN', 'KANLinear',
    # Faster KAN
    'FasterKAN', 'FasterKANLayer', 'SplineLinear', 'ReflectionalSwitchFunction',
    # Vision KAN
    'VisionKAN', 'MoE_KAN_MLP', 'kanBlock', 'create_kan'
]