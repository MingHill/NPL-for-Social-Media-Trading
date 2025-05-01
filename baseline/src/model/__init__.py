"""
Model interfaces and implementations.
"""

from .model import Model
from .hf_transformers_model import HuggingFaceTransformersModel

__all__ = ["Model", "HuggingFaceTransformersModel"] 