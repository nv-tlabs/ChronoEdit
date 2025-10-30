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
torchrun --nproc_per_node=2 -m pytest chronoedit/_ext/imaginaire/checkpointer/ddp_test.py
"""

import os
import shutil

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from megatron.core import parallel_state

from chronoedit._ext.imaginaire.checkpointer.ddp import Checkpointer
from chronoedit._ext.imaginaire.config import CheckpointConfig, Config, JobConfig, ObjectStoreConfig
from chronoedit._ext.imaginaire.lazy_config import LazyCall as L
from chronoedit._ext.imaginaire.lazy_config import instantiate
from chronoedit._ext.imaginaire.model import ImaginaireModel
from chronoedit._ext.imaginaire.trainer import ImaginaireTrainer
from chronoedit._ext.imaginaire.utils import distributed, log


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class SimpleImaginaireModel(ImaginaireModel):
    def __init__(self):
        super().__init__()
        self.model = SimpleModel()

    def forward(self, x):
        return self.model(x)


@pytest.fixture(scope="module")
def setup_trainer_config():
    config = Config(
        model=L(SimpleImaginaireModel)(),
        optimizer=L(torch.optim.Adam)(params=None, lr=0.001),
        scheduler=L(torch.optim.lr_scheduler.StepLR)(optimizer=None, step_size=1, gamma=0.1),
        dataloader_train=None,
        dataloader_val=None,
    )
    # construct the trainer, which will also construct distributed init process group
    trainer = ImaginaireTrainer(config)
    model = instantiate(config.model).cuda()
    optimizer, scheduler = model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    grad_scaler = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    yield trainer, model, optimizer, scheduler, grad_scaler, config
    # Cleanup
    dist.destroy_process_group()
    parallel_state.destroy_model_parallel()


def setup_checkpointer(trainer, object_store_enabled, job_name):
    _object_store = ObjectStoreConfig(
        enabled=object_store_enabled,
        bucket="checkpoints" if object_store_enabled else "test-bucket",
        credentials="credentials/pbss_dir.secret" if object_store_enabled else "test-credentials",
    )
    config_checkpoint = CheckpointConfig(
        save_to_object_store=_object_store,
        load_from_object_store=_object_store,
        strict_resume=True,
        load_path=None,
        load_training_state=True,
        only_load_scheduler_state=False,
    )
    config_job = JobConfig(
        project="imaginaire4",
        group="test_checkpointer",
        name=job_name,
    )
    ckpt = Checkpointer(config_checkpoint, config_job, trainer.callbacks)
    return ckpt


def train_and_save(model, optimizer, scheduler, grad_scaler, checkpointer):
    x = torch.randn(32, 10).cuda()
    y = torch.randn(32, 5).cuda()
    criterion = nn.MSELoss()

    for _ in range(2):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()
        scheduler.step()
    checkpointer.save(model, optimizer, scheduler, grad_scaler, iteration=10)


def load_and_compare(
    model,
    optimizer,
    scheduler,
    grad_scaler,
    checkpointer: Checkpointer,
    original_model,
    original_optimizer,
    original_scheduler,
):
    iteration = checkpointer.load(model, optimizer, scheduler, grad_scaler)

    dist.barrier()
    if iteration != 10:
        log.critical(f"Iteration number does not match after loading checkpoint: {iteration}", rank0_only=False)
    assert iteration == 10, "Iteration number does not match after loading checkpoint"

    # compare model parameters
    for param1, param2 in zip(original_model.parameters(), model.parameters()):
        assert torch.allclose(param1, param2), f"Model parameters differ after loading checkpoint {param1} {param2}"
    distributed.barrier()
    log.success("Model parameters match after loading checkpoint", rank0_only=False)

    # compare optimizer states
    for param1, param2 in zip(
        original_optimizer.state_dict()["state"].values(), optimizer.state_dict()["state"].values()
    ):
        for k in param1:
            assert torch.allclose(param1[k], param2[k]), f"Optimizer state {k} differs after loading checkpoint"

    log.success("Optimizer states match after loading checkpoint", rank0_only=False)

    # compare scheduler states
    orginal_lr = original_scheduler.get_last_lr()
    loaded_lr = scheduler.get_last_lr()
    if orginal_lr != loaded_lr:
        log.critical(f"Learning rate differs after loading checkpoint: {orginal_lr} {loaded_lr}", rank0_only=False)
    assert orginal_lr == loaded_lr, "Learning rate differs after loading checkpoint"
    log.success("Learning rate matches after loading checkpoint", rank0_only=False)


@pytest.mark.skip(reason="Tests are available in the test environment, run manually")
def test_checkpointer_local(setup_trainer_config):
    trainer, model, optimizer, scheduler, grad_scaler, config = setup_trainer_config
    checkpointer = setup_checkpointer(trainer, object_store_enabled=False, job_name="local_ddp")
    if distributed.get_rank() == 0:
        if os.path.exists(checkpointer.checkpoint_dir_local):
            shutil.rmtree(checkpointer.checkpoint_dir_local)

    ddp_model = distributed.DistributedDataParallel(model)
    train_and_save(ddp_model, optimizer, scheduler, grad_scaler, checkpointer)
    checkpointer.finalize()
    distributed.barrier()
    checkpointer = setup_checkpointer(trainer, object_store_enabled=False, job_name="local_ddp")

    model2 = SimpleImaginaireModel().cuda()
    optimizer2 = optim.Adam(model2.parameters())
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.1)
    grad_scaler2 = torch.amp.GradScaler("cuda", **config.trainer.grad_scaler_args)
    ddp_model2 = distributed.DistributedDataParallel(model2)

    load_and_compare(ddp_model2, optimizer2, scheduler2, grad_scaler2, checkpointer, ddp_model, optimizer, scheduler)


@pytest.mark.skip(reason="Tests are available in the test environment, run manually")
def test_checkpointer_remote(setup_trainer_config):
    trainer, model, optimizer, scheduler, grad_scaler, config = setup_trainer_config
    checkpointer = setup_checkpointer(trainer, object_store_enabled=True, job_name="remote_ddp")

    ddp_model = distributed.DistributedDataParallel(model)
    train_and_save(ddp_model, optimizer, scheduler, grad_scaler, checkpointer)
    checkpointer.finalize()
    distributed.barrier()
    checkpointer = setup_checkpointer(trainer, object_store_enabled=True, job_name="remote_ddp")

    model2 = SimpleImaginaireModel().cuda()
    optimizer2 = optim.Adam(model2.parameters())
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.1)
    grad_scaler2 = torch.amp.GradScaler(**config.trainer.grad_scaler_args)
    ddp_model2 = distributed.DistributedDataParallel(model2)

    load_and_compare(ddp_model2, optimizer2, scheduler2, grad_scaler2, checkpointer, ddp_model, optimizer, scheduler)
