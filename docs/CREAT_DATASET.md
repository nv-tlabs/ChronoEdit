## Dataset Preparation

### Overview

ChronoEdit provides an automated editing labeling script to generate high-quality editing instructions from pairs of images (before and after editing). The script uses state-of-the-art vision-language models to analyze image pairs and generate precise editing prompts with Chain-of-Thought (CoT) reasoning.

**The CoT reasoning output is the actual prompt used as input to the ChronoEdit model.** It contains detailed information about what should be edited and what should be preserved (appearance, pose, style, composition, etc.).

### Quick Start

Generate editing labels for an image pair:

```bash
python scripts/data_captioning.py \
  --input-image ./assets/images/input.jpg \
  --output-image ./assets/images/output.jpg \
  --output-file ./assets/captions/caption.txt \
  --generate-cot
```

### Script Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input-image` | str | `./assets/images/input.jpg` | Path to the input (original) image |
| `--output-image` | str | `./assets/images/output.jpg` | Path to the output (edited) image |
| `--output-file` | str | `./assets/captions/caption.txt` | Path to save the generated caption |
| `--model` | str | `Qwen/Qwen3-VL-30B-A3B-Instruct` | Vision-language model to use for generation |
| `--max-resolution` | int | `1080` | Maximum resolution for the shortest edge (pixels) |
| `--generate-cot` | flag | `False` | Generate Chain-of-Thought reasoning (the actual model input prompt) |

### Supported Models

The script supports three vision-language models:

1. **`Qwen/Qwen2.5-VL-7B-Instruct`** - Smaller, faster model suitable for quick iterations
2. **`Qwen/Qwen3-VL-30B-A3B-Instruct`** - Default model, balanced quality and speed

Example with different model:
```bash
python scripts/data_captioning.py \
  --input-image ./assets/images/input.jpg \
  --output-image ./assets/images/output.jpg \
  --model Qwen/Qwen2.5-VL-7B-Instruct
```

### How It Works

The script uses a two-stage process:

1. **Stage 1 - Caption Generation**: Analyzes the image pair and generates a concise editing instruction describing the main changes.
   - Example: *"Move the knight's shield to his right hand"*

2. **Stage 2 - CoT Reasoning** (with `--generate-cot`): Expands the caption into a detailed prompt that specifies:
   - What should be edited (the changes)
   - What must be preserved (appearance, pose, style, composition)
   - Motion and spatial details
   - Visual style consistency
   - Example: *"The user wants to move the shield from the left hand to the right hand. The knight should maintain the same defensive posture and stance, with the shield now gripped firmly in the right hand. The armor reflections, proportions, and medieval style should remain consistent..."*

**The CoT reasoning (Stage 2) is the actual prompt fed into ChronoEdit for training and inference.**

### Output Format

When using `--generate-cot`, the script saves a JSON file containing both outputs:

```json
{
  "caption": "Move the knight's shield to his right hand",
  "caption_cot": "The user wants to move the shield from the left hand to the right hand. The knight should maintain the same defensive posture and stance, with the shield now gripped firmly in the right hand. The armor reflections, proportions, and medieval style should remain consistent, emphasizing a powerful defensive stance."
}
```

- **`caption`**: Concise editing instruction (intermediate output)
- **`caption_cot`**: Detailed reasoning prompt (actual model input, 80-100 words)

Without `--generate-cot`, only a plain text file with the caption is saved.

### CoT Prompt Characteristics

The CoT reasoning prompt includes:

- **Change description**: What specific edits should be made
- **Preservation details**: What must stay unchanged (pose, appearance, clothing, expression, skin tone, age)
- **Spatial information**: Location and positioning details
- **Visual style**: Genre consistency (anime, CG, cinematic, etc.)
- **Motion context**: Natural human motion and interactions when relevant
- **Composition**: Shot type and camera angle preservation

This comprehensive information helps the editing model understand both what to change and what constraints to respect.

### Example Dataset

We provide an example dataset to demonstrate the expected format for training ChronoEdit. The dataset is available at:

**[https://huggingface.co/datasets/nvidia/ChronoEdit-Example-Dataset/tree/main/difix_dataset](https://huggingface.co/datasets/nvidia/ChronoEdit-Example-Dataset/tree/main/difix_dataset)**

#### Dataset Structure

The dataset consists of three main components:

```
difix_dataset/
├── metadata.csv          # Main metadata file with prompts and paths
├── videos/              # Video clips directory
│   └── *.mp4            # Individual video clips
└── umt5/                # Pre-computed UMT5 embeddings (optional)
    └── *.pkl            # Embedding pickle files
```

#### Metadata CSV Format

The `metadata.csv` file contains the following columns:

| Column | Description |
|--------|-------------|
| `key` | Unique identifier for each sample |
| `video` | Relative path to the video file (e.g., `videos/sample_001.mp4`) |
| `prompt` | The CoT reasoning prompt (actual model input) |
| `umt5` | Relative path to pre-computed UMT5 embedding file (optional, used for full model finetuning) |

Example entry from [metadata.csv](https://huggingface.co/datasets/nvidia/ChronoEdit-Example-Dataset/blob/main/difix_dataset/metadata.csv):

```csv
key,video,prompt,umt5
sample_001,videos/sample_001.mp4,The user wants to remove visible neural rendering artifacts while keeping the core visual content entirely unchanged.,umt5/sample_001.pkl
```

**Note**: The `prompt` column contains the CoT reasoning text generated by the captioning script (Stage 2 output).

### UMT5 Embedding Extraction (For Full Model Finetuning)

For full model finetuning, ChronoEdit pre-computes UMT5 embeddings from prompts to improve training efficiency. This step is optional but recommended for large-scale training.

#### Extract UMT5 Embeddings

Run the extraction script on your prepared metadata CSV:

```bash
python scripts/extract_umt5.py --csv_path YOUR_CSV_PATH
```

Example:
```bash
python scripts/extract_umt5.py --csv_path ./data/my_dataset/metadata.csv
```

#### What the Script Does

1. **Reads the metadata CSV** - Loads all entries with video paths and prompts
2. **Computes UMT5 embeddings** - Processes each prompt through the UMT5 model
3. **Saves embeddings** - Stores each embedding as a `.pkl` file in `umt5/` directory
4. **Updates CSV** - Adds/updates the `umt5` column with paths to embedding files

#### Output

After running the script:
- A `umt5/` directory is created in the same location as your CSV
- Each prompt gets a corresponding `.pkl` file with the UMT5 embedding
- The `metadata.csv` is updated with the `umt5` column populated

Example output structure:
```
data/my_dataset/
├── metadata.csv           # Updated with umt5 column
├── videos/
│   └── sample_001.mp4
└── umt5/                  # Newly created
    └── sample_001.pkl     # Pre-computed embedding
```

#### When to Use UMT5 Pre-computation

- ✅ **Use for**: Full model finetuning with large datasets
- ✅ **Benefit**: Significantly speeds up training by avoiding repeated text encoding
- ❌ **Skip for**: LoRA training or small-scale experiments (embeddings computed on-the-fly)

### Preparing Your Own Dataset

To prepare a custom dataset for ChronoEdit training:

1. **Collect image pairs** - Organize your input (before) and output (after) images
2. **Convert image pairs to videos** - Create 2-frame videos from each pair
3. **Generate prompts** - Use `data_captioning.py` with `--generate-cot` to create CoT prompts
4. **Create metadata.csv** - Organize data in the format shown above
5. **(Optional) Extract UMT5 embeddings** - Run `extract_umt5.py` for full model finetuning

#### Creating Videos from Image Pairs

ChronoEdit expects video files as input. For image editing tasks, create a 2-frame video where:
- **Frame 1**: Input/source image (before editing)
- **Frame 2**: Output/target image (after editing)

Here's how to convert image pairs to videos:

```python
import os
import imageio

# Paths to your images
source_image_path = "path/to/input.jpg"
target_image_path = "path/to/output.jpg"
key = "sample_001"  # Unique identifier

# Create video from image pair
output_dir = "./data/my_dataset"
video_path = os.path.join(output_dir, f"videos/{key}.mp4")
os.makedirs(os.path.dirname(video_path), exist_ok=True)

with imageio.get_writer(video_path, fps=1) as writer:
    for image_path in [source_image_path, target_image_path]:
        image = imageio.imread(image_path)
        writer.append_data(image)
```

**Key points:**
- Use `fps=1` (1 frame per second) for 2-frame videos
- First frame is the source (input) image
- Second frame is the target (output) image
- Each video should have a unique identifier (`key`)

For batch processing multiple image pairs:

```python
import os
import imageio

# List of image pairs: (source_path, target_path, unique_key)
image_pairs = [
    ("inputs/image_001.jpg", "outputs/image_001.jpg", "sample_001"),
    ("inputs/image_002.jpg", "outputs/image_002.jpg", "sample_002"),
    # ... more pairs
]

output_dir = "./data/my_dataset"

for source_path, target_path, key in image_pairs:
    video_path = os.path.join(output_dir, f"videos/{key}.mp4")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    with imageio.get_writer(video_path, fps=1) as writer:
        for image_path in [source_path, target_path]:
            image = imageio.imread(image_path)
            writer.append_data(image)
    
    print(f"Created video: {video_path}")
```

### Notes

- **For training data preparation, always use `--generate-cot`** to generate the actual model input prompts
- Images are automatically resized if their shortest edge exceeds `--max-resolution`
- If no changes are detected between images, the output will be "no change"
- CoT is only generated when changes are detected (not for "no change" cases)
