# verl-research

**Standalone research infrastructure for rapid RL algorithm prototyping on top of verl.**

This is a **self-contained** experiment framework that uses verl as an installed package. You can detach this folder from the verl repo and work on it independently.

## 🎯 Purpose

Test new RL algorithm ideas quickly:
- New paper drops → Implement variant → Know if it works (in < 1 day)
- Organized experiments (baseline vs variants)
- Track what works vs what sucks
- No risk to your base verl installation

## 📦 Installation

### 1. Install verl (if not already installed)

```bash
pip install verl[vllm,gpu,math]
# Or from source:
# cd /path/to/verl && pip install -e .
```

### 2. Install research dependencies

```bash
cd verl-research
pip install -r requirements.txt
```

That's it! Everything uses verl as an installed package.

## 🚀 Quick Start (Test a New Idea in 3 Steps)

### Step 1: Create a new experiment

```bash
python tools/new_experiment.py "paper_dapo_advantage"
```

This creates: `experiments/01_paper_dapo_advantage/` with template files.

### Step 2: Implement your change

```bash
cd experiments/01_paper_dapo_advantage/
```

Edit `variant.py`:

```python
from extensions.custom_advantages import BaseAdvantageCompute

class DAPOAdvantage(BaseAdvantageCompute):
    def compute(self, rewards, ref_rewards, **kwargs):
        """DAPO: Use reward difference as advantage"""
        return rewards - ref_rewards  # Your idea here
```

Edit `config.yaml`:

```yaml
variant:
  name: "dapo_advantage"
  advantage_class: "variant.DAPOAdvantage"

model:
  path: "Qwen/Qwen3-8B"

data:
  train_files: "/data/gsm8k/train.parquet"
  val_files: "/data/gsm8k/test.parquet"
```

### Step 3: Test it (3-stage pipeline)

```bash
# Stage 1: Quick smoke test (5 min)
python ../../tools/quick_test.py .

# Stage 2: Small train (30 min)
python ../../tools/train.py . --mode small

# Stage 3: Full train (3 hours)
python ../../tools/train.py . --mode full --gpus 8
```

### Step 4: Compare with baseline

```bash
python ../../tools/compare.py ../00_baseline .
```

Output:
```
╔══════════════════════════════════════╗
║     Baseline vs Variant Comparison  ║
╠══════════════════════════════════════╣
║ Baseline:  75.3% ± 0.4              ║
║ Variant:   78.1% ± 0.3              ║
║ Δ:         +2.8pp                    ║
║ p-value:   < 0.01 ✅ SIGNIFICANT    ║
╚══════════════════════════════════════╝
```

## 📊 View Leaderboard

```bash
python tools/leaderboard.py
```

Output:
```
┌──────────────────────────────────────────────┐
│         verl-research Leaderboard            │
├──────────────────────────────────────────────┤
│ Rank  Experiment           Accuracy   Δ      │
├──────────────────────────────────────────────┤
│  🥇   dapo_advantage        78.1%    +2.8pp  │
│  🥈   reward_smoothing      75.8%    +0.5pp  │
│  🥉   baseline              75.3%     0.0pp  │
│  4    curiosity_bonus       74.2%    -1.1pp  │
└──────────────────────────────────────────────┘
```

## 🔧 What Can You Change?

See `docs/EXTENSION_POINTS.md` for the complete guide.

Common changes:

| What You Want | Where to Change | Example |
|--------------|----------------|---------|
| New advantage computation | `extensions/custom_advantages.py` | GRPO → DAPO |
| New loss term | `extensions/custom_losses.py` | Add curiosity bonus |
| Reward shaping | `extensions/custom_rewards.py` | Temporal smoothing |
| Sampling strategy | `extensions/custom_samplers.py` | Temperature scheduling |

## 📁 Directory Structure

```
verl-research/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── experiments/
│   ├── 00_baseline/             # Your baseline (vanilla verl)
│   │   ├── config.yaml
│   │   ├── train.py
│   │   ├── results.json
│   │   └── README.md
│   ├── 01_your_variant/         # Your experiments
│   │   ├── config.yaml
│   │   ├── variant.py           # ONLY your changes
│   │   ├── train.py             # Generated
│   │   ├── results.json         # Auto-generated
│   │   └── README.md            # Document your idea
│   └── template/                # Template for new experiments
├── tools/
│   ├── new_experiment.py        # Create new experiment
│   ├── quick_test.py            # 5-min smoke test
│   ├── train.py                 # Small/full training
│   ├── compare.py               # Compare experiments
│   └── leaderboard.py           # Show all results
├── extensions/
│   ├── __init__.py
│   ├── custom_advantages.py     # Extend advantage computation
│   ├── custom_losses.py         # Extend loss functions
│   ├── custom_rewards.py        # Extend reward shaping
│   ├── custom_samplers.py       # Extend sampling
│   └── base.py                  # Base classes
├── docs/
│   ├── EXTENSION_POINTS.md      # Where to change what
│   ├── QUICKSTART.md            # Detailed tutorial
│   └── EXAMPLES.md              # 5 example variants
└── results/
    ├── comparison.csv           # All experiments
    └── best_configs/            # Hall of fame
```

## 🎓 Examples Included

We include 3 ready-to-run examples:

1. **00_baseline**: Vanilla GRPO on GSM8K (your reference)
2. **01_dapo_advantage**: DAPO-style advantage (from paper)
3. **02_entropy_bonus**: Add entropy bonus to loss

See `docs/EXAMPLES.md` for details.

## 🔬 How It Works (Under the Hood)

verl-research uses **monkey patching** to inject your changes:

```python
# Your variant.py
class MyCustomAdvantage:
    def compute(self, rewards, values):
        # Your algorithm here
        return advantages

# Tool applies it:
import verl.trainer.ppo.core_algos as algos
algos.compute_advantage = MyCustomAdvantage().compute
```

**No modifications to verl itself** - everything is an override layer.

## 📈 Tracking & Logging

All experiments auto-log to:
- **Local CSV**: `results/comparison.csv`
- **Weights & Biases** (optional): Set `WANDB_PROJECT` env var
- **JSON files**: Each experiment's `results.json`

## 🔥 Advanced Usage

### Sweep over hyperparameters

```bash
python tools/sweep.py experiments/01_variant/ \
  --param "learning_rate" \
  --values "1e-7,1e-6,1e-5"
```

### Multi-dataset testing

```bash
python tools/multi_dataset.py experiments/01_variant/ \
  --datasets "gsm8k,math,aime"
```

### Statistical significance testing

```bash
python tools/compare.py exp1/ exp2/ --runs 5 --test ttest
```

## 🆘 Troubleshooting

**Q: Import error: "No module named verl"**
```bash
# Make sure verl is installed:
pip list | grep verl
# If not: pip install verl[vllm,gpu,math]
```

**Q: Experiment crashes immediately**
```bash
# Use quick_test first to debug:
python tools/quick_test.py experiments/01_variant/
```

**Q: Results look weird**
```bash
# Check if your variant is actually applied:
python tools/verify_variant.py experiments/01_variant/
```

## 📚 Learn More

- **Extension Points Guide**: `docs/EXTENSION_POINTS.md`
- **Quickstart Tutorial**: `docs/QUICKSTART.md`
- **Example Variants**: `docs/EXAMPLES.md`
- **verl Documentation**: https://verl.readthedocs.io/

## 🎉 Contributing

Since this is your personal research repo, organize it however you want! Some ideas:

- Tag experiments: `experiments/[YYYYMMDD]_idea_name/`
- Archive dead ends: `experiments/archive/failed_idea/`
- Share winners: Push to your own git repo

## 📄 License

Same as verl (Apache 2.0)

---

**Happy researching! 🚀**

Questions? Check `docs/QUICKSTART.md` or the example experiments.
