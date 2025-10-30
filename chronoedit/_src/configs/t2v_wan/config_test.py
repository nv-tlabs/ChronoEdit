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


import pytest

from  chronoedit._ext.imaginaire.utils.config_helper import override
from chronoedit._src.configs.t2v_wan.config import make_config


@pytest.mark.L1
def test_make_config():
    config = make_config()
    config = override(config, ["--", "experiment=error-free_fsdp_mock-data_base-cb", "trainer.max_iter=1"])

    assert config.trainer.max_iter == 1
