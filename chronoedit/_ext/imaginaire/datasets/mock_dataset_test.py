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
    pytest -s chronoedit/_ext/imaginaire/datasets/mock_dataset_test.py
"""

import pytest
import torch

from chronoedit._ext.imaginaire.datasets.mock_dataset import CombinedDictDataset, LambdaDataset, RepeatDataset
from chronoedit._ext.imaginaire.lazy_config import LazyCall as L
from chronoedit._ext.imaginaire.lazy_config import instantiate


@pytest.fixture
def cfg():
    return L(CombinedDictDataset)(
        key1=L(LambdaDataset)(
            length=64,
            fn=lambda: torch.randn(3, 32, 32),
        ),
        key2=L(RepeatDataset)(
            dataset=L(LambdaDataset)(
                fn=lambda: torch.randn(3, 32, 32),
            ),
        ),
    )


@pytest.mark.L0
def test_mock_dataset(cfg):
    batch_size = 4
    dataset_obj = instantiate(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset_obj,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=1,
    )
    assert len(dataset_obj) == 64
    for ith, batch in enumerate(dataloader):
        assert batch["key1"].shape == (batch_size, 3, 32, 32)
        assert batch["key2"].shape == (batch_size, 3, 32, 32)
        if ith > 2:
            break
