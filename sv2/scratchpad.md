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
├── main_ppo_multiturn_toolcall.py  # Main driver (hydra entrypoint)
├── scratchpad.md                   # This file
├── config/
│   ├── sv2_multiturn.yaml          # Standalone hydra config
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
