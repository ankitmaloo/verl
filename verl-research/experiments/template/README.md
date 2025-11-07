# [Experiment Name]

## Hypothesis

[Describe what you're testing and why]

Example: "Paper X claims that using reward - reference_reward as advantage improves sample efficiency."

## Implementation

[Describe what you changed]

Example: "Modified advantage computation in variant.py to use DAPO-style advantage."

## Results

[Document your findings]

### Quick Test (5 min)
- ✅/❌ No crashes
- ✅/❌ Loss decreasing
- Notes: [Any observations]

### Small Train (30 min)
- Baseline at epoch 3: X%
- Variant at epoch 3: Y%
- Δ: +/- Z%
- Notes: [Observations]

### Full Train (3 hours)
- Baseline: X% ± σ
- Variant: Y% ± σ
- Δ: +/- Z% (p-value: X)
- Statistical significance: ✅/❌
- Notes: [Final observations]

## Analysis

[Detailed analysis]

- What worked well?
- What didn't work?
- Why do you think that is?
- Unexpected findings?

## Next Steps

[What to do next]

Example:
- ✅ Try on MATH dataset
- ✅ Tune hyperparameters
- ❌ Not worth pursuing further

## Files

- `config.yaml`: Configuration
- `variant.py`: Implementation
- `results.json`: Metrics
- `checkpoints/`: Model checkpoints
- `plots/`: Visualizations

## References

[Link to papers, discussions, etc.]
