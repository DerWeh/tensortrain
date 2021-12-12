"""Fixes for third-party functionality that is broken or lacking.

Contains own verision, till up-streams are fixed.
"""
from tensortrain.fixes._expm_multiply import expm_multiply

__all__ = (
    "expm_multiply",
)
