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
Usage:
    pytest -v -s chronoedit/_ext/imaginaire/utils/misc_test.py
"""

import pytest
import torch

from chronoedit._ext.imaginaire.utils.misc import get_data_batch_size


@pytest.mark.L0
def test_get_data_batch_size():
    """
    Test get_data_batch_size function.

    This test verifies that the function returns the correct data batch size for various inputs.
    """
    data_batch = {"images": torch.zeros(1, 3, 16, 16), "tokens": torch.zeros(1, 16)}
    assert get_data_batch_size(data_batch) == 1

    data_batch = {"images": torch.zeros(2, 3, 16, 16), "tokens": torch.zeros(2, 16)}
    assert get_data_batch_size(data_batch) == 2

    # Nested dictionary "__url__" without torch tensors - should be skipped
    data_batch = {
        "__key__": "value",
        "__url__": {"k1": "v1", "k2": "v2"},
        "images": torch.zeros(2, 3, 16, 16),
        "tokens": torch.zeros(2, 16),
    }

    assert get_data_batch_size(data_batch) == 2

    # Nested dictionary "image_dict" with torch tensors - should be counted
    data_batch = {
        "__key__": "value",
        "__url__": {"k1": "v1", "k2": "v2"},
        "image_dict": {"images": torch.zeros(2, 3, 16, 16)},
    }
    assert get_data_batch_size(data_batch) == 2

    # Invalid data_batch that should raise ValueError
    invalid_data_batch = {"__key__": "value", "__url__": {"k1": "v1", "k2": "v2"}, "tokens": [0, 1, 2]}
    with pytest.raises(ValueError):
        get_data_batch_size(invalid_data_batch)
