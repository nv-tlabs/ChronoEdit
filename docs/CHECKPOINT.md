# Checkpoints

ChronoEdit supports **three** checkpoint formats, each optimized for a different use case.

### 1) DCP checkpoints
**DCP (Diffusion Checkpoint Package)** is a unified, sharded model bundle designed for **fast, distributed** saving/loading on multi-GPU/multi-node setups.  
We use **DCP** as the **default training checkpoint** format due to speed and robustness in distributed environments.

### 2) Consolidated `.pth` checkpoints
We provide a script to convert a DCP checkpoint into a single, regular PyTorch **`.pth`** checkpoint (convenient for single-node inference or simple export).

**Example:**
```bash
python scripts/convert_distcp_to_pt.py \
  "./checkpoints/ChronoEdit-14B/iter_000010000/model" \
  "./checkpoints/ChronoEdit-14B/nvidia/chronoedit_14B"
```
scripts/run_inference.py accepts both DCP checkpoints and consolidated .pth checkpoints.

### 3) Diffusers checkpoints
For Hugging Face Diffusers workflows, convert from DCP to the Diffusers format (directory with model, scheduler, and config files).

**Example:**
```bash
python scripts/convert_distcp_to_diffusers.py \
  --src "./checkpoints/ChronoEdit-14B/iter_000010000/model" \
  --dst "./checkpoints/ChronoEdit-14B-Diffusers"
```
the Diffusers pipeline only accepts Diffusers-format checkpoints.
