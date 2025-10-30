# Alpha Version

The project draws significant inspiration from [torchtitan](https://github.com/pytorch/torchtitan). Familiarizing yourself with torchtitan and its documentation will greatly streamline the onboarding process and enhance your ability to contribute effectively. If you are not yet acquainted with newer PyTorch features, such as FSDP2 and DTensor, torchtitan serves as an excellent resource to build your understanding.

Test docker environment: `nvcr.io/nvidian/imaginaire4:mcore_v0.0.7`

# text2video related code tests

## function,module-level test

```shell
pytest -s projects/cosmos/diffusion/v2/networks/minimal_v4_dit_test.py --all

pytest -s projects/cosmos/diffusion/v2/models/t2v_model_test.py --all
torchrun --nproc_per_node=2 -m projects.cosmos.diffusion.v2.models.model_fsdp2_test
torchrun --nproc_per_node=2 -m pytest -v --L1 projects/cosmos/diffusion/v2/context_parallel_test.py
```

## end2end test

**No skipped tests**

```shell
# forward and backward tests
pytest -s projects/cosmos/diffusion/v2/tests/t2v_end2end_test.py --all 2>&1 | tee /tmp/err.log

# train with callbacks
pytest -s projects/cosmos/diffusion/v2/tests/t2v_callback_test.py --all 2>&1 | tee /tmp/err.log

# test checkpointer
pytest -s projects/cosmos/diffusion/v2/tests/fsdp_ckpt_end2end_test.py --all 2>&1 | tee /tmp/err.log

# test against v1 network
pytest -s projects/cosmos/diffusion/v2/networks/minimal_v4_dit_test.py --all
```

## overfit test

some tests can takes several hours to finish

* check [experiment file](projects/cosmos/diffusion/v2/configs/t2v/experiment/bug_free/overfit.py)

here is small launch script with launcher

```python
import os
import pwd
import time

from  chronoedit._ext.imaginaire.utils import log


def get_executor(
    nnode: int,
    job_group: str,
    job_name: str,
    partition: str,
    stage_code: bool = True,
):
    import launcher

    if "WANDB_API_KEY" not in os.environ:
        log.critical("Please set WANDB_API_KEY in the environment variables.")
        exit(1)
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

    TIME_TAG = time.strftime("%Y%m%d-%H%M%S")

    user = os.environ.get("USER")
    if user is None:
        user = pwd.getpwuid(os.getuid()).pw_name
    assert user is not None, "Cannot get user name."

    user_fp = f"/project/cosmos/{user}"

    executor = launcher.SlurmExecutor(
        env_vars=dict(
            WANDB_API_KEY=WANDB_API_KEY,
            WANDB_ENTITY="nvidia-dir",
            TORCH_NCCL_ENABLE_MONITORING="0",
            TORCH_NCCL_AVOID_RECORD_STREAMS="1",
            TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="1800",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            IMAGINAIRE_OUTPUT_ROOT=os.path.join(user_fp, "imaginaire4-output"),
        ),
        local_root=os.getcwd(),
        docker_image="/project/cosmos/snah/dpdata/sqsh/imaginaire4_mcore_v0.0.7_efa.sqsh",
        cluster="aws-iad-cs-002",
        partition=partition,
        account="dir_cosmos_base",
        num_gpus=8,
        num_nodes=nnode,
        exclude_nodes=["pool0-0023", "pool0-0006", "pool0-0028", "pool0-0002"],
        slurm_workdir=os.path.join(user_fp, "projects/cosmos/diffusion/v2", job_group, job_name, TIME_TAG),
        slurm_logdir=os.path.join(user_fp, "logs", "cosmos_diffusion_v2", job_group, job_name),
        slurm_cachedir=user_fp,
        enable_aps=False,
    )
    if stage_code:
        executor.stage_code()

    return executor


def debug_job(
    nnode = 8,
    job_group = "debug",
    job_name = None,
    exp_name = "BASE001_001_LR-14_VideoImage_1-1",
    run_tag = "P1",
    command_args = [],
    is_halla = True,
    proj: str = "causal",
):
    partition = "pool0_datahall_a" if is_halla else "pool0_datahall_b"
    slurm_executor = get_executor(
        nnode=nnode,
        job_group=job_group,
        job_name=exp_name,
        partition=partition,
        stage_code=True,
    )

    base_command = (
        f"python -m scripts.train "
        f"--config=projects/cosmos/diffusion/v2/configs/{proj}/config.py "
        "--"
    )
    params = [
        f"experiment={exp_name}",
        f"job.group={job_group}",
    ]
    if job_name:
        params.append(f"job.name={job_name}")
    params.extend(command_args)

    command = f"{base_command} {' '.join(params)}"

    slurm_executor.submit_job(
        command=command,
        job_name=f"{nnode}N@{job_group}@{job_name or exp_name}@{run_tag}",
    )

if __name__ == "__main__":
#-----------------------------  t2v overfit -----------------------------
    debug_job(
        nnode = 4,
        job_group = "t2v_overfit",
        exp_name = "t2v_overfit_fsdp_video-only",
        is_halla = True,
        command_args=[
        ],
        proj = "t2v",
    )

    debug_job(
        nnode = 4,
        job_group = "t2v_overfit",
        exp_name = "t2v_overfit_ddp_video-only",
        is_halla = True,
        command_args=[
        ],
        proj = "t2v",
    )
```

# For researchers to modify networks arch

Please check impl in the [demo code](projects/cosmos/diffusion/v2/networks/minimal_v4_dit.py)! Since we support torch meta device and follow torchtitan, new module or new parameters must support [reset_parameters or init_weights](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py#L386).



# Important Implementation Details

## Data Batch Handling

1. In general, `data_batch` should be treated as immutable and should not be modified during processing. This helps maintain code clarity and prevents unexpected side effects.

2. The primary exception to this rule is in `model.get_data_and_condition()`, where we:
   - Normalize and augment the data
   - Add additional latent information to the batch
   - Prepare the data for model processing

3. For vid2vid models, when explicit control is needed during inference, you may need to add `"num_conditional_frames"` to the data_batch dictionary (see vid2vid README for details).
