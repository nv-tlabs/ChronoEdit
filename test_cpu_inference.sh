#!/bin/bash
# Test script for CPU inference support
# Based on demo1.sh but simplified for testing

set -e  # Exit on error

echo "=========================================="
echo "ChronoEdit CPU Inference Test"
echo "=========================================="

# Check if device argument is provided
DEVICE=${1:-cpu}

echo "Testing with device: $DEVICE"
echo ""

# Set environment variables
export PYTHONPATH=$(pwd)

# Run inference without prompt enhancer for faster testing
echo "Running inference..."
python scripts/run_inference_diffusers.py \
    --input assets/images/input_2.png \
    --device "$DEVICE" \
    --prompt "Add sunglasses to the cat's face" \
    --output "output_${DEVICE}_test.mp4" \
    --num-inference-steps 5 \
    --model-path ./checkpoints/ChronoEdit-14B-Diffusers \
    --verbose

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "Output saved to: output_${DEVICE}_test.mp4"
echo "=========================================="
