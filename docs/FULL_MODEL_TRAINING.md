# 📑 Full Model Training Framework
We release **ChronoEdit’s** full training infrastructure and codebase, enabling **distributed inference** and **large-scale fine-tuning** of pretrained video diffusion models, including [**WAN 2.1**](https://github.com/Wan-Video/Wan2.1) and [**Cosmos 2.5**](https://github.com/nvidia-cosmos/cosmos-predict2.5/).  
The training framework and system design are directly inherited from the [**Cosmos 2.5**](https://github.com/nvidia-cosmos/cosmos-predict2.5/) release.  

> [!NOTE]
> The codebase has only been tested on **NVIDIA A100** and **NVIDIA H100** GPUs with CUDA 12.9 environments.

---

### Full Installation

Download dcp checkpoint from HuggingFace:
```bash
hf download nvidia/ChronoEdit-14B --local-dir checkpoints/ChronoEdit-14B
```

To enable full model training, ensure transformer-engine (with patch fixes) is properly installed:

```bash
conda env create -f environment.yml -n chronoedit_full
conda activate chronoedit_full
pip install torch==2.7.1 torchvision==0.22.1 
pip install -r requirements.txt
# Set a limit to the number of threads to prevent OOM for flash attention compilation
export MAX_JOBS=16
pip install flash-attn==2.6.3
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
pip install transformer-engine[pytorch]
conda install -y -c conda-forge libstdcxx-ng>=13.2.0 libgcc-ng>=13.2.0
```
Verify the installation:

```bash
python -c "import transformer_engine as te; print('TE import OK')"
```

> [!NOTE]
> If you encounter the following error:
>  ```bash
> Command 'ldconfig -p | grep 'libnvrtc'' returned non-zero exit status 1.
>  ```
> It indicates that CUDA is installed inside the Conda environment rather than in the default system path.
> To resolve this, manually link the CUDA environment:
> ```bash
> source scripts/TE_patch.sh
>  ```





### Inference with Single-GPU
The following script support inference script with only one GPU.
```bash
PYTHONPATH=$(pwd) python -m torch.distributed.run --nproc_per_node=1 --master_port=12340 -m scripts.run_inference \
    --experiment edit_14B_skip_pe8 \
    --checkpoint_path ./checkpoints/ChronoEdit-14B/nvidia/chronoedit_14B/model.pth \
    --save_root outputs/0903_wan_edit/image_edit_video_prior \
    --num_frames 2 \
    --resolution "720p" \
    --guidance 5.0 \
    --prompt "Add a person wearing red jacket" \
    --input_image_fp assets/images/input.jpg
```

> [!NOTE]
> If you encounter the following error:
>  ```bash
> [rank1]: ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by ~/.triton/cache/{...}/cuda_utils.so
>  ```
> This indicates a higher libc version was previously used to compile a library on the same device.
> To resolve this, simply delete the triton cache:
> ```bash
> rm -rf ~/.triton/cache
>  ```


### Inference with Multi-GPUs 
The following script support distributed inference with two GPUs with Context Parallel Ring Attention.
```bash
PYTHONPATH=$(pwd) python -m torch.distributed.run --nproc_per_node=2 --master_port=12340 -m scripts.run_inference \
    --experiment edit_14B_skip_pe8 \
    --checkpoint_path ./checkpoints/ChronoEdit-14B/nvidia/chronoedit_14B/model.pth \
    --save_root outputs/0903_wan_edit/image_edit_video_prior \
    --num_frames 2 \
    --resolution "720p" \
    --guidance 5.0 \
    --prompt "Add a person wearing red jacket" \
    --input_image_fp assets/images/input.jpg
```


### Full Model Training 


Test training with context parallel. Trainined was tested on  **NVIDIA A100** and **NVIDIA H100** GPUs with 80G memory.
```bash
torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train --config=chronoedit/_src/configs/chronoedit/config.py -- experiment="edit_14B_skip_pe8_mock"
```

###  Checkpoint Conversions


ChronoEdit supports **three** checkpoint formats, each optimized for a different use case. See checkpoint guidance [doc](docs/CHECKPOINT.md).
