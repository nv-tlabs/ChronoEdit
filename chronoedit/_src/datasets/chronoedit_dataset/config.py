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
from copy import deepcopy

DATASET_CONFIG = dict()
DATASET_CONFIG["chronoedit_data_image"] = {
    "dataset_cfg": {
        "target": "chronoedit._src.datasets.chronoedit_dataset.unified_dataset.UnifiedDataset",
        "params": {
            "base_path": "data/difix_dataset",
            "metadata_path": "data/difix_dataset/metadata.csv",
            "repeat": 1,
            "data_file_keys": ["video", "umt5"],
            "main_data_operator": {
                "target": "default_video_operator",
                "params": {
                    "base_path": "data/difix_dataset",
                    "max_pixels": 1024*1024,
                    "height_division_factor": 16,
                    "width_division_factor": 16,
                    "num_frames": 2,
                    "time_division_factor": 4,
                    "time_division_remainder": 1,
                }
            }
        },
    },
}
DATASET_CONFIG["chronoedit_data_video"] = deepcopy(DATASET_CONFIG["chronoedit_data_image"])
DATASET_CONFIG["chronoedit_data_video"]["dataset_cfg"]["params"]["main_data_operator"]["params"]["num_frames"] = 26
