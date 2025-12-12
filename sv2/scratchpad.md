## sv2 scratchpad (handoff notes)

### Goal (what user asked)
- Keep new work inside `sv2/`.
- Implement a new entrypoint "off `main_ppo.py`" to make *multi-turn + tool-calling* data→rollout flow explicit.
- Focus is the step: `dataset -> RLHFDataset -> send to rollouts -> (tools/user turns happen) -> get RLHF trajectories`.
- Multi-turn concept: prompt → assistant → tool call(s) → assistant → (interaction/user adds message) → repeat.
- **NEW (v2)**: Add user feedback injection ("are you sure, can you write python code to verify")

### What exists in repo (important)
- PPO entry: `verl/trainer/main_ppo.py` builds tokenizer, RLHFDataset, RayPPOTrainer.
- Multi-turn + tool calling is implemented in the **AgentLoop** stack:
  - `verl/experimental/agent_loop/tool_agent_loop.py` (`agent_name="tool_agent"`)
  - Tools registry + schemas: `verl/tools/utils/tool_registry.py`
  - Tool parsing formats: `verl/experimental/agent_loop/tool_parser.py` (default `hermes`)
  - Optional "user/environment feedback" is an **Interaction**:
    - Interface: `verl/interactions/base.py`
    - Config loader: `verl/interactions/utils/interaction_registry.py`
- Key gotcha: **AgentLoop selection is per-sample** via dataset column `agent_name`.
  - Default is `single_turn_agent` unless `agent_name` exists.
  - Example dataset builder that sets `agent_name="tool_agent"`:
    - `examples/data_preprocess/gsm8k_tool_agent_loop.py`

### sv2 file structure (standalone)
```
sv2/
├── __init__.py
├── install.sh                      # uv-based setup (verl 0.6.1)
├── main_ppo_multiturn_toolcall.py  # Main driver (hydra entrypoint, orchestrates everything)
├── data.py                         # Data loading utilities (tokenizer, datasets, dataloaders)
├── reward.py                       # GSM8K reward function + Sv2RewardManager
├── eval.py                         # Evaluation: rollouts + reward computation
├── train.py                        # Training loop PLACEHOLDER (PPO not implemented)
├── scratchpad.md                   # This file
├── config/
│   ├── sv2_multiturn.yaml          # Standalone hydra config (Qwen3-0.6B, H100)
│   └── interaction_config.yaml     # Interaction registry (code_verify)
├── interactions/
│   ├── __init__.py
│   └── code_verify_interaction.py  # Custom interaction class
└── data/
    ├── __init__.py
    └── preprocess_gsm8k.py         # Dataset prep with interaction_kwargs
```

### New code added (v2)
- `sv2/install.sh`: uv-based installation (creates .venv, installs verl 0.6.1 + sglang)
- `sv2/config/sv2_multiturn.yaml`: Standalone hydra config (sglang, multi-turn enabled)
- `sv2/config/interaction_config.yaml`: Registers `code_verify` interaction
- `sv2/interactions/code_verify_interaction.py`:
  - **Implements the "are you sure" follow-up feature**
  - After model's first answer, injects user message asking for Python code verification
  - Configurable via `feedback_message`, `max_turns`, `terminate_message`
- `sv2/data/preprocess_gsm8k.py`: Prepares GSM8K with `extra_info.interaction_kwargs`
- Updated `sv2/main_ppo_multiturn_toolcall.py`:
  - Added `sv2.interaction_name` config option
  - Auto-injects `interaction_kwargs` into batch if not in dataset

### How Interaction system works (CRITICAL)
1. **ToolAgentLoop** handles both tools AND interactions
2. Flow: `GENERATING → INTERACTING → GENERATING → ...`
3. In `INTERACTING` state, calls `interaction.generate_response(instance_id, messages)`
4. Interaction returns: `(should_terminate, feedback_text, reward, metadata)`
5. `feedback_text` is tokenized and appended as user message
6. Loop continues until `should_terminate=True` or max turns

### Interaction class interface (`verl/interactions/base.py`)
```python
class BaseInteraction:
    async def start_interaction(instance_id, **kwargs) -> str
    async def generate_response(instance_id, messages, **kwargs)
        -> (should_terminate: bool, response: str, reward: float, metadata: dict)
    async def calculate_score(instance_id, **kwargs) -> float
    async def finalize_interaction(instance_id, **kwargs) -> None
```

### How to run (v2 - with user feedback)

**Step 1: Setup**
```bash
cd /path/to/verl
./sv2/install.sh
source sv2/.venv/bin/activate
```

**Step 2: Prepare data**
```bash
python -m sv2.data.preprocess_gsm8k --output_dir ~/data/gsm8k_sv2 --interaction_name code_verify
```

**Step 3: Run rollouts with code verification feedback**
```bash
python -m sv2.main_ppo_multiturn_toolcall \
  --config-path sv2/config --config-name sv2_multiturn \
  data.train_files=~/data/gsm8k_sv2/train.parquet \
  data.val_files=~/data/gsm8k_sv2/test.parquet \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path=sv2/config/interaction_config.yaml \
  sv2.interaction_name=code_verify \
  sv2.batch_size=4 sv2.max_batches=1 \
  sv2.dump_jsonl=/tmp/sv2_code_verify.jsonl
```

**Alternative: Dataset already has interaction_kwargs**
```bash
# If dataset has extra_info.interaction_kwargs.name, don't need sv2.interaction_name
python -m sv2.main_ppo_multiturn_toolcall \
  --config-path sv2/config --config-name sv2_multiturn \
  data.train_files=~/data/gsm8k_sv2/train.parquet \
  data.val_files=~/data/gsm8k_sv2/test.parquet \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path=sv2/config/interaction_config.yaml
```

### Padding gotcha (AgentLoopManager)
- `AgentLoopManager.generate_sequences()` chunks the batch into `actor_rollout_ref.rollout.agent.num_workers`.
- `DataProto.chunk()` asserts `len(batch) % num_workers == 0` unless padded.
- Fix used in `sv2/main_ppo_multiturn_toolcall.py`: call
  - `pad_dataproto_to_divisor(gen_batch, num_workers)` before `generate_sequences`
  - `unpad_dataproto(...)` after.

### Known issues / failure modes to remember
- If dataset doesn't include `agent_name`, tool-calling/interaction won't happen unless you set it (driver warns + defaults to `tool_agent` if interaction or tools enabled).
- If `data.return_raw_chat` is false, AgentLoop can't run (driver errors).
- If `sv2.batch_size` isn't divisible by `actor_rollout_ref.rollout.agent.num_workers` and padding isn't applied, it crashes (now handled).
- If interaction is enabled but `extra_info.interaction_kwargs.name` is missing AND `sv2.interaction_name` not set, ToolAgentLoop raises.
- If `hydra-core` isn't installed, `sv2/main_ppo_multiturn_toolcall.py` exits early with an install hint.

### Hypothesis / Learning log

**Hypothesis 1**: Can inject user feedback via Interaction system without modifying verl core
- **Status**: SUCCESS
- **Learning**: ToolAgentLoop already supports this via `interaction_config_path`. The key is:
  1. Set `interaction_config_path` in config
  2. Ensure `extra_info.interaction_kwargs.name` matches registered interaction
  3. Interaction's `generate_response()` returns the feedback text

**Hypothesis 2**: Can make sv2 standalone with minimal deps
- **Status**: SUCCESS
- **Learning**: Created `install.sh` with uv, standalone config YAML. Key deps: verl, hydra-core, sglang
- **CRITICAL**: Version pinning is essential to avoid CUDA/PyTorch hell

### Pinned versions (from verl's install_vllm_sglang_mcore.sh)
```
Python:          3.12
SGLang:          0.5.2  (install FIRST - brings correct PyTorch)
vLLM:            0.11.0
FlashInfer:      0.3.1
CUDA:            >= 12.1 (12.4+ recommended)
PyTorch:         2.8.x (comes with sglang)
transformers:    >= 4.51.0
tensordict:      >= 0.8.0, <= 0.10.0, != 0.9.0
numpy:           < 2.0.0
```

### Install order matters!
1. Install SGLang first → brings PyTorch with correct CUDA
2. Install vLLM → must be compatible with PyTorch from step 1
3. Install verl → uses PyTorch from above
4. Install FlashInfer → must match CUDA/PyTorch versions

Sources:
- [verl installation docs](https://verl.readthedocs.io/en/latest/start/install.html)
- [verl install script](scripts/install_vllm_sglang_mcore.sh)

**Hypothesis 3**: Dataset doesn't need pre-baked interaction_kwargs
- **Status**: SUCCESS
- **Learning**: Added `_ensure_interaction_kwargs()` in driver to inject at runtime via `sv2.interaction_name`

### Key files in verl to reference
- `verl/interactions/base.py` - BaseInteraction interface
- `verl/interactions/gsm8k_interaction.py` - Simple example
- `verl/experimental/agent_loop/tool_agent_loop.py:385-434` - Interaction state handling
- `verl/interactions/utils/interaction_registry.py` - YAML config loader
- `examples/data_preprocess/gsm8k_multiturn_w_interaction.py` - Dataset with interaction_kwargs

---

## v3: Modular Architecture with Train/Eval Split

### New code added (v3)

**Modular file structure:**
- `sv2/data.py` - Data loading utilities
  - `build_tokenizer_processor()` - Creates tokenizer/processor from model config
  - `create_dataset()` - Wraps verl's `create_rl_dataset`
  - `create_dataloader()` - Creates DataLoader with RLHF collation
  - `select_data_paths()` - Selects train/val paths from config

- `sv2/reward.py` - Reward computation
  - `extract_gsm8k_answer()` - Extracts #### answer from response
  - `compute_gsm8k_score()` - Scores GSM8K responses (0 or 1)
  - `compute_score()` - Generic dispatcher by data_source
  - `Sv2RewardManager` - Reward manager class compatible with DataProto

- `sv2/eval.py` - Evaluation pipeline
  - `run_eval()` - Runs rollouts on val data, computes rewards
  - `EvalResult` - Dataclass with metrics (mean_reward, accuracy, samples)
  - Handles padding, agent_name injection, interaction_kwargs

- `sv2/train.py` - Training loop **PLACEHOLDER**
  - `run_training_loop()` - Training loop with periodic eval
  - `TrainConfig` - Training configuration dataclass
  - `_ppo_update_placeholder()` - **NOT IMPLEMENTED** - just logs warning
  - When `train=True`: runs loop but NO weight updates happen
  - When `train=False`: just runs eval

- `sv2/main_ppo_multiturn_toolcall.py` - Main orchestrator
  - Reads `train` flag from config
  - If `train=True`: calls `run_training_loop()`
  - If `train=False`: calls `run_eval()`

**Config updates (sv2/config/sv2_multiturn.yaml):**
```yaml
# Top-level train flag
train: false  # Set to true for training mode

# Training configuration
training:
  total_steps: 100
  eval_every_n_steps: 10
  save_every_n_steps: 50
  ppo_epochs: 1
  learning_rate: 1e-6
  clip_ratio: 0.2
  gamma: 0.99
  gae_lambda: 0.95
```

### How reward system works

**verl's reward architecture (from `verl/trainer/ppo/reward.py`):**
1. `load_reward_manager()` creates a reward manager instance
2. Reward managers inherit from `AbstractRewardManager`
3. Default is `NaiveRewardManager` which:
   - Decodes prompt/response from DataProto
   - Calls `compute_score(data_source, solution_str, ground_truth, extra_info)`
   - Places reward at last valid token position
4. `default_compute_score()` in `verl/utils/reward_score/__init__.py` dispatches by `data_source`:
   - `"openai/gsm8k"` → `gsm8k.compute_score()`
   - `"lighteval/MATH"` → `math_reward.compute_score()`
   - etc.

**sv2's reward (from `sv2/reward.py`):**
- Simplified version of NaiveRewardManager
- Only implements GSM8K scorer (extracts `#### answer`)
- Can extend `compute_score()` for other data sources

### How to run (v3)

**Eval-only mode (default):**
```bash
python -m sv2.main_ppo_multiturn_toolcall \
  --config-path sv2/config --config-name sv2_multiturn \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet \
  sv2.dump_jsonl=/tmp/sv2_eval.jsonl
```

**Training mode (placeholder - no actual weight updates):**
```bash
python -m sv2.main_ppo_multiturn_toolcall \
  --config-path sv2/config --config-name sv2_multiturn \
  train=true \
  training.total_steps=50 \
  training.eval_every_n_steps=10 \
  data.train_files=$DATA_DIR/train.parquet \
  data.val_files=$DATA_DIR/test.parquet
```

### What's NOT implemented (TODOs for training)

1. **PPO update** - `sv2/train.py:_ppo_update_placeholder()` needs:
   - Advantage computation (GAE)
   - Old log prob computation
   - Policy loss with clipping
   - Value loss
   - Actor/critic weight updates

2. **Checkpointing** - Save/load model weights

3. **Reference policy** - For KL penalty

4. **Critic** - Value function for advantage estimation

To implement actual training, look at:
- `verl/trainer/ppo/ray_trainer.py` - Full PPO implementation
- `verl/workers/fsdp_workers.py` - Actor/Critic workers
- `verl/trainer/ppo/core_algos.py` - PPO loss functions

### Known issues (v3)

- Training mode runs but **does NOT update weights** - it's a placeholder
- Reward computation happens in driver process (not distributed)
- No KL penalty support
- No value function / critic
- Config inherits from `ppo_trainer` which may have extra fields we don't use

---

## v3.1: Bug Fixes

### Error: `IndexError: list index out of range` in AgentLoopManager

**Symptom:**
```
AgentLoopManager: []
IndexError: list index out of range
  File "verl/experimental/agent_loop/agent_loop.py", line 84, in _choose_server
    server = self.weighted_serveres[0][1][1]
```

**Root cause:**
`AgentLoopManager._initialize_llm_servers()` calculates `num_replicas = world_size // rollout_world_size`.
When running standalone (no `worker_group`), it uses `config.trainer.n_gpus_per_node * config.trainer.nnodes`.

The issue was missing config values for `data_parallel_size` and `pipeline_model_parallel_size`
which caused `rollout_world_size` to be calculated incorrectly.

**Fix:** Added explicit values in `sv2/config/sv2_multiturn.yaml`:
```yaml
actor_rollout_ref:
  rollout:
    tensor_parallel_size: 1
    data_parallel_size: 1  # Added
    pipeline_model_parallel_size: 1  # Added
    free_cache_engine: false  # Added
```

### Error: `data.py` conflicts with `data/` folder

**Symptom:** Import errors when running from sv2/ folder because Python can't distinguish
between `sv2/data.py` (module) and `sv2/data/` (package).

**Fix:** Renamed `sv2/data.py` → `sv2/dataflow.py`

### Error: Preprocessing writes to `~/data/` instead of local folder

**Symptom:** Data preprocessing creates files in `/home/ubuntu/data/gsm8k_sv2/` instead of
within the sv2 project folder.

**Fix:** Changed default `--output_dir` from `~/data/gsm8k_sv2` to `data/gsm8k` (relative path).

### File structure after fixes
```
sv2/
├── __init__.py
├── install.sh
├── main_ppo_multiturn_toolcall.py  # Main driver
├── dataflow.py                     # Data loading (renamed from data.py)
├── reward.py                       # GSM8K reward
├── eval.py                         # Evaluation
├── train.py                        # Training placeholder
├── scratchpad.md
├── config/
│   ├── sv2_multiturn.yaml          # Main config
│   └── interaction_config.yaml
├── interactions/
│   └── code_verify_interaction.py
└── data/
    ├── __init__.py
    ├── preprocess_gsm8k.py
    └── gsm8k/                      # Created by preprocessing
        ├── train.parquet
        └── test.parquet
```

### How to run (from sv2/ folder)

**Step 1: Preprocess data**
```bash
cd /path/to/verl/sv2
python -m data.preprocess_gsm8k
# Creates data/gsm8k/train.parquet and data/gsm8k/test.parquet
```

**Step 2: Run eval**
```bash
python -m main_ppo_multiturn_toolcall \
  data.train_files=data/gsm8k/train.parquet \
  data.val_files=data/gsm8k/test.parquet \
  actor_rollout_ref.rollout.multi_turn.interaction_config_path=config/interaction_config.yaml
```



---

## v3.2: Bug Fixes (verl 0.6.1 compatibility)

### Error: `TypeError: AgentLoopManager.__init__() got an unexpected keyword argument 'rm_resource_pool'`

**Symptom:**
```
TypeError: AgentLoopManager.__init__() got an unexpected keyword argument 'rm_resource_pool'
```

**Root cause:**
verl 0.6.1 uses `rm_wg` parameter name, NOT `rm_resource_pool` (which is used in main branch).

**Fix:** Line 219 in `main_ppo_multiturn_toolcall.py`:
```python
# WRONG (main branch):
agent_loop_manager = AgentLoopManager(config=config, worker_group=None, rm_resource_pool=None)

# CORRECT (verl 0.6.1):
agent_loop_manager = AgentLoopManager(config=config, worker_group=None, rm_wg=None)
```

### Error: Config key mismatch `tensor_parallel_size` vs `tensor_model_parallel_size`

**Symptom:**
Would cause AttributeError when AgentLoopManager._initialize_llm_servers() runs.

**Root cause:**
Config had `tensor_parallel_size: 1` but verl expects `tensor_model_parallel_size`.

**Fix:** In `sv2/config/sv2_multiturn.yaml`:
```yaml
# WRONG:
tensor_parallel_size: 1

# CORRECT:
tensor_model_parallel_size: 1
```

### Key lesson: verl 0.6.1 vs main branch differences

When targeting verl 0.6.1, always check the released version's API, not main branch:
- `AgentLoopManager.__init__` parameter: `rm_wg` (0.6.1) vs `rm_resource_pool` (main)
- Config keys: Always use `tensor_model_parallel_size` (consistent across versions)

---

## v3.3: More Bug Fixes (standalone execution from sv2/)

### Error: Hydra searchpath wrong - can't find ppo_trainer defaults

**Symptom:**
```
Could not find 'ppo_trainer' in search path
```

**Root cause:**
Hydra `file://` paths are relative to the config file location (`sv2/config/`).
- WRONG: `file://../verl/trainer/config` → resolves to `sv2/config/../verl/trainer/config` = `sv2/verl/trainer/config` (doesn't exist)
- CORRECT: `file://../../verl/trainer/config` → resolves to `sv2/config/../../verl/trainer/config` = `verl/trainer/config`

**Fix:** In `sv2/config/sv2_multiturn.yaml`:
```yaml
hydra:
  searchpath:
    # Path relative to this config file (sv2/config/)
    - file://../../verl/trainer/config
```

### Error: interaction_config.yaml class_name not found

**Symptom:**
```
ModuleNotFoundError: No module named 'sv2'
```

**Root cause:**
When running from inside sv2/, Python's sys.path doesn't include the parent directory, so `sv2.interactions.code_verify_interaction` can't be resolved.

**Fix:** In `sv2/config/interaction_config.yaml`, use path relative to cwd:
```yaml
# WRONG (when running from sv2/):
class_name: "sv2.interactions.code_verify_interaction.CodeVerifyInteraction"

# CORRECT (when running from sv2/):
class_name: "interactions.code_verify_interaction.CodeVerifyInteraction"
```

### Verified working imports (verl 0.6.1)

All these imports exist in verl 0.6.1:
- ✅ `from verl import DataProto`
- ✅ `from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto`
- ✅ `from verl.experimental.agent_loop import AgentLoopManager`
- ✅ `from verl.trainer.constants_ppo import get_ppo_ray_runtime_env`
- ✅ `from verl.trainer.main_ppo import create_rl_dataset`
- ✅ `from verl.utils import hf_processor, hf_tokenizer`
- ✅ `from verl.utils.dataset.rl_dataset import collate_fn`
- ✅ `from verl.utils.fs import copy_to_local`
- ✅ `from verl.interactions.base import BaseInteraction`

### Running from sv2/ folder - checklist

1. **Activate venv**: `source .venv/bin/activate` (or create via `./install.sh`)
2. **Preprocess data**: `python -m data.preprocess_gsm8k`
3. **Run**: `python -m main_ppo_multiturn_toolcall data.train_files=data/gsm8k/train.parquet data.val_files=data/gsm8k/test.parquet`

### Potential remaining issues

1. **agent_loops/tool_agent_subagent_checker.py** - Uses `@register("sv2_tool_agent_checker")` decorator. This registration happens at import time. If this module isn't imported, the agent won't be registered. May need to add explicit import somewhere.

2. **interactions/__init__.py** - Imports `CodeVerifyInteraction` but verl's interaction registry loads via class_name string, not via this __init__.py. The __init__.py import is likely unused. 