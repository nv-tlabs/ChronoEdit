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

import gc
import math
import os

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    
from loguru import logger as logging
import torch


def get_gpu_architecture():
    """
    Retrieves the GPU architecture of the available GPUs.

    Returns:
        str: The GPU architecture, which can be "H100", "A100", "L40S", "B200", "Other", or None (CPU mode).
    """
    if not PYNVML_AVAILABLE or not torch.cuda.is_available():
        return None
        
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            model_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(model_name, bytes):
                model_name = model_name.decode("utf-8")
            print(f"GPU {i}: Model: {model_name}")

            # Check for specific models like H100 or A100
            if "H100" in model_name or "H200" in model_name:
                return "H100"
            elif "A100" in model_name:
                return "A100"
            elif "L40S" in model_name:
                return "L40S"
            elif "B200" in model_name:
                return "B200"
    except Exception as error:
        print(f"Failed to get GPU info: {error}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    # return "Other" incase of non hopper/ampere or error
    return "Other"


class GPUArchitectureNotSupported(Exception):
    """
    Custom exception raised when the expected GPU architecture is not supported.
    """

    pass


def print_gpu_mem(str=None):
    if not PYNVML_AVAILABLE or not torch.cuda.is_available():
        if str:
            logging.info(f"{str}: Running on CPU (no GPU memory info)")
        return
        
    try:
        pynvml.nvmlInit()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        logging.info(
            f"{str}: {meminfo.used / 1024 / 1024}/{meminfo.total / 1024 / 1024}MiB used ({meminfo.free / 1024 / 1024}MiB free)"
        )
    except Exception as error:
        print(f"Failed to get GPU memory info: {error}")


def force_gc():
    print_gpu_mem()
    print("gc()")
    gc.collect()
    print_gpu_mem()
    print("empty cuda cache")
    # print(torch.cuda.memory_summary())
    print_gpu_mem()


def gpu0_has_80gb_or_less():
    if not PYNVML_AVAILABLE or not torch.cuda.is_available():
        return True  # Conservative default for CPU mode
        
    try:
        pynvml.nvmlInit()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0))
        return meminfo.total / 1024 / 1024 / 1024 <= 80
    except Exception as error:
        print(f"Failed to get GPU memory info: {error}")
        return True  # Conservative default on error


class Device:
    _nvml_affinity_elements = math.ceil(os.cpu_count() / 64) if os.cpu_count() else 1  # type: ignore

    def __init__(self, device_idx: int):
        super().__init__()
        if not PYNVML_AVAILABLE or not torch.cuda.is_available():
            raise RuntimeError("Device class requires CUDA and pynvml to be available")
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

    def get_name(self) -> str:
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_cpu_affinity(self) -> list[int]:
        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinity(self.handle, Device._nvml_affinity_elements):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string
        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list
        return [i for i, e in enumerate(affinity_list) if e != 0]
