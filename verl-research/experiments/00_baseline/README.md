# Baseline: Vanilla GRPO on GSM8K

## Purpose

This is the **baseline** experiment that all variants are compared against.

- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Dataset**: GSM8K (grade school math)
- **Model**: Qwen3-8B
- **Backend**: vLLM

## Configuration

See `config.yaml` for full settings. Key parameters:
- Learning rate: 1e-6
- Batch size: 1024
- Epochs: 15
- No custom extensions (pure verl)

## Expected Results

Based on verl benchmarks:
- **Initial accuracy**: ~75% (pre-trained model)
- **Final accuracy**: ~77-78% (after GRPO training)
- **Training time**: ~3 hours on 8x A100-80GB

## How to Run

```bash
cd experiments/00_baseline

# Quick test (5 min)
python ../../tools/quick_test.py .

# Full training (3 hours)
python ../../tools/train.py . --mode full --gpus 8
```

## Results

### Metrics

- Accuracy: TBD
- Train time: TBD
- GPU hours: TBD

### Notes

This baseline establishes the performance floor. Any variant should ideally beat this.

## Metadata

- **Created**: Baseline experiment
- **Status**: 📍 Reference point
- **Purpose**: Comparison target for all variants
