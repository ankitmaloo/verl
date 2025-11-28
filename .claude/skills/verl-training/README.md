# VERL Training Skill

Write 4 simple Python files that call VERL components for RL training. For fast iteration.

## Structure

```
verl-training/
├── SKILL.md                    # Main skill documentation (entry point)
├── references/
│   ├── standalone.md          # **START HERE**: 4 simple Python files pattern
│   ├── inference.md           # Inference engine details (SGLang/vLLM/HF)
│   ├── training.md            # Policy loss & optimization details
│   ├── environment.md         # Dataset & curriculum learning details
│   └── algorithms.md          # Reward & advantage estimation details
└── README.md                  # This file
```

## Quick Start

1. **Read SKILL.md** - Overview and use cases
2. **Read references/standalone.md** - Copy-paste the 4 Python files
3. **Modify env.py** - Adapt for your task (dataset, RL game, multi-turn)
4. **Run**: `python algorithm.py`

## What You Get

**4 Python files** you can copy and modify:
- `algorithm.py` - Main training loop
- `inference.py` - Calls VERL rollout workers (SGLang/vLLM/HF)
- `training.py` - Calls VERL actor/critic workers
- `env.py` - Your data source (dataset, RL game, multi-turn)

**Detailed references** for when you need to understand VERL internals:
- `inference.md` - Multi-turn, tool calling, sampling params
- `training.md` - Policy losses, optimizers, FSDP
- `environment.md` - Dataset classes, curriculum learning
- `algorithms.md` - Reward functions, advantage estimators

## Common Use Cases

- Dataset training (GSM8K, custom data)
- RL game environments (LLM playing games)
- Multi-turn conversations
- Tool/function calling
- Custom rewards (math, code, QA)
- Curriculum learning
- GRPO, GAE, RLOO advantage estimation

All examples in `references/standalone.md`.
