# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Device utility module for ChronoEdit.

Provides centralized device detection and management with automatic fallback
from CUDA to CPU when CUDA is not available.
"""

import warnings
from typing import Optional, Union

import torch


def is_cuda_available() -> bool:
    """
    Check if CUDA is available.
    
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    return torch.cuda.is_available()


def get_device_type(device: Optional[Union[str, torch.device]] = None) -> str:
    """
    Get the device type string ('cuda' or 'cpu').
    
    Args:
        device: Device specification. Can be:
            - None: Auto-detect (CUDA if available, else CPU)
            - str: "cuda", "cpu", "cuda:0", etc.
            - torch.device: Device object
    
    Returns:
        str: Device type ('cuda' or 'cpu')
    """
    if device is None:
        # Auto-detect
        if is_cuda_available():
            return "cuda"
        else:
            warnings.warn(
                "CUDA is not available. Falling back to CPU. "
                "Note: CPU inference will be significantly slower.",
                UserWarning
            )
            return "cpu"
    
    if isinstance(device, torch.device):
        return device.type
    
    if isinstance(device, str):
        # Handle "cuda:0" -> "cuda", "cpu" -> "cpu"
        device_lower = device.lower()
        if device_lower.startswith("cuda"):
            if not is_cuda_available():
                warnings.warn(
                    f"CUDA device '{device}' requested but CUDA is not available. "
                    "Falling back to CPU.",
                    UserWarning
                )
                return "cpu"
            return "cuda"
        elif device_lower == "cpu":
            return "cpu"
        else:
            raise ValueError(f"Unknown device type: {device}")
    
    raise TypeError(f"Device must be None, str, or torch.device, got {type(device)}")


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Get a torch.device object with automatic CUDA to CPU fallback.
    
    Args:
        device: Device specification. Can be:
            - None: Auto-detect (CUDA if available, else CPU)
            - str: "cuda", "cpu", "cuda:0", etc.
            - torch.device: Device object (returned as-is after validation)
    
    Returns:
        torch.device: Device object ready for use
        
    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda")  # Use CUDA (falls back to CPU if unavailable)
        >>> device = get_device("cpu")  # Force CPU
        >>> device = get_device("cuda:0")  # Specific CUDA device
    """
    if device is None:
        # Auto-detect
        device_type = get_device_type(None)
        return torch.device(device_type)
    
    if isinstance(device, torch.device):
        # Validate and potentially fall back
        if device.type == "cuda" and not is_cuda_available():
            warnings.warn(
                f"CUDA device '{device}' requested but CUDA is not available. "
                "Falling back to CPU.",
                UserWarning
            )
            return torch.device("cpu")
        return device
    
    if isinstance(device, str):
        device_lower = device.lower()
        
        if device_lower.startswith("cuda"):
            if not is_cuda_available():
                warnings.warn(
                    f"CUDA device '{device}' requested but CUDA is not available. "
                    "Falling back to CPU.",
                    UserWarning
                )
                return torch.device("cpu")
            # Return the specific CUDA device (e.g., cuda:0, cuda:1)
            return torch.device(device_lower)
        
        elif device_lower == "cpu":
            return torch.device("cpu")
        
        else:
            raise ValueError(f"Unknown device type: {device}")
    
    raise TypeError(f"Device must be None, str, or torch.device, got {type(device)}")


def get_device_map(device: Optional[Union[str, torch.device]] = None) -> str:
    """
    Get device_map string for HuggingFace models with automatic fallback.
    
    Args:
        device: Device specification (None for auto-detect)
    
    Returns:
        str: Device map string compatible with HuggingFace from_pretrained
        
    Examples:
        >>> device_map = get_device_map()  # "cuda:0" or "cpu"
        >>> device_map = get_device_map("cuda")  # "cuda:0" or "cpu" (with fallback)
    """
    device_obj = get_device(device)
    
    if device_obj.type == "cuda":
        # Use specific device index if available, otherwise default to 0
        if device_obj.index is not None:
            return f"cuda:{device_obj.index}"
        return "cuda:0"
    else:
        return "cpu"

