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

import csv
import pickle
from pathlib import Path
import argparse
import torch
from scripts.umt5 import get_umt5_embedding

def extract_and_save_umt5_embeddings(csv_path):
    """
    Extract UMT5 embeddings from captions in the metadata CSV file.
    
    Args:
        csv_path: Path to the metadata CSV file
    """
    csv_path = Path(csv_path)
    csv_dir = csv_path.parent
    
    # Create umt5 directory if it doesn't exist
    umt5_dir = csv_dir / "umt5"
    umt5_dir.mkdir(exist_ok=True)
    
    print(f"Reading metadata from: {csv_path}")
    print(f"UMT5 embeddings will be saved to: {umt5_dir}")
    
    # Read the CSV file
    rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    print(f"Found {len(rows)} entries in metadata.csv")
    
    # Add 'umt5' column if it doesn't exist
    if 'umt5' not in fieldnames:
        fieldnames = list(fieldnames) + ['umt5']
    
    # Process each row
    for idx, row in enumerate(rows):
        video_name = row['video']
        caption = row['prompt']
        
        # Create a unique filename based on the video name
        video_basename = Path(video_name).stem
        umt5_filename = f"{video_basename}.pkl"
        umt5_path = umt5_dir / umt5_filename
        
        print(f"[{idx+1}/{len(rows)}] Processing: {video_name}")
        print(f"  Caption: {caption[:100]}..." if len(caption) > 100 else f"  Caption: {caption}")
        
        # Extract UMT5 embedding
        try:
            embeddings = get_umt5_embedding(caption)[0].to(dtype=torch.bfloat16).cpu()
            
            # Save embedding to pickle file
            with open(umt5_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Store relative path in the row
            row['umt5'] = f"umt5/{umt5_filename}"
            print(f"  Saved to: {row['umt5']}")
            
        except Exception as e:
            print(f"  Error processing caption: {e}")
            row['umt5'] = ""
    
    # Write updated CSV file
    print(f"\nWriting updated metadata to: {csv_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\nDone! Processed {len(rows)} captions.")
    print(f"UMT5 embeddings saved in: {umt5_dir}")
    print(f"Updated metadata file: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract UMT5 embeddings from captions in metadata CSV')
    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to the metadata CSV file',
        required=True,
    )
    
    args = parser.parse_args()
    extract_and_save_umt5_embeddings(args.csv_path)


if __name__ == '__main__':
    main()

