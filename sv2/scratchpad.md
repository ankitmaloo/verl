## sv2 scratchpad (handoff notes)

### Goal (what user asked)
- Keep new work inside `sv2/`.
- Implement a new entrypoint “off `main_ppo.py`” to make *multi-turn + tool-calling* data→rollout flow explicit.
- Focus is the step: `dataset -> RLHFDataset -> send to rollouts -> (tools/user turns happen) -> get RLHF trajectories`.
- Multi-turn concept: prompt → assistant → tool call(s) → assistant → (interaction/user adds message) → repeat.

### What exists in repo (important)
- PPO entry: `verl/trainer/main_ppo.py` builds tokenizer, RLHFDataset, RayPPOTrainer.
- Multi-turn + tool calling is implemented in the **AgentLoop** stack:
  - `verl/experimental/agent_loop/tool_agent_loop.py` (`agent_name="tool_agent"`)
  - Tools registry + schemas: `verl/tools/utils/tool_registry.py`
  - Tool parsing formats: `verl/experimental/agent_loop/tool_parser.py` (default `hermes`)
  - Optional “user/environment feedback” is an **Interaction**:
    - Interface: `verl/interactions/base.py`
    - Config loader: `verl/interactions/utils/interaction_registry.py`
- Key gotcha: **AgentLoop selection is per-sample** via dataset column `agent_name`.
  - Default is `single_turn_agent` unless `agent_name` exists.
  - Example dataset builder that sets `agent_name="tool_agent"`:
    - `examples/data_preprocess/gsm8k_tool_agent_loop.py`

### New code added
- `sv2/main_ppo_multiturn_toolcall.py`
  - Hydra entrypoint (config defaults to `ppo_trainer`).
  - Loads RLHF dataset via `create_rl_dataset` (same as `main_ppo.py`).
  - Builds a generation batch and sends it to `AgentLoopManager.generate_sequences`.
  - Decodes outputs and can dump JSONL.
- `sv2/__init__.py` (so it can be run as a module).

### How to run (typical)
- Requires `python3` (in this environment `python` was not found).
- Example (edit paths/model as needed):
  - `python3 -m sv2.main_ppo_multiturn_toolcall --config-path examples/sglang_multiturn/config --config-name gsm8k_multiturn_grpo \`
    `data.train_files=$HOME/data/gsm8k_tool_agent_loop/train.parquet data.val_files=$HOME/data/gsm8k_tool_agent_loop/test.parquet \`
    `data.return_raw_chat=true actor_rollout_ref.rollout.mode=async actor_rollout_ref.rollout.name=vllm \`
    `actor_rollout_ref.rollout.multi_turn.enable=true actor_rollout_ref.rollout.multi_turn.tool_config_path=examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml \`
    `sv2.batch_size=4 sv2.max_batches=1 sv2.split=val sv2.dump_jsonl=/tmp/sv2_rollouts.jsonl`

### Multi-turn “user adds string” (Interaction)
- To get “assistant → (tools) → assistant → user feedback → …”, set:
  - `actor_rollout_ref.rollout.multi_turn.interaction_config_path=/path/to/interaction_config.yaml`
- AND ensure each sample’s parquet has `extra_info.interaction_kwargs.name` equal to one of the registered interaction names.
  - ToolAgentLoop requires `interaction_kwargs["name"]` if interaction_config_path is set.

### Padding gotcha (AgentLoopManager)
- `AgentLoopManager.generate_sequences()` chunks the batch into `actor_rollout_ref.rollout.agent.num_workers`.
- `DataProto.chunk()` asserts `len(batch) % num_workers == 0` unless padded.
- Fix used in `sv2/main_ppo_multiturn_toolcall.py`: call
  - `pad_dataproto_to_divisor(gen_batch, num_workers)` before `generate_sequences`
  - `unpad_dataproto(...)` after.

### Known issues / failure modes to remember
- If dataset doesn’t include `agent_name`, tool-calling won’t happen unless you set it (driver warns + defaults).
- If `data.return_raw_chat` is false, AgentLoop can’t run (driver errors).
- If `sv2.batch_size` isn’t divisible by `actor_rollout_ref.rollout.agent.num_workers` and padding isn’t applied, it crashes (now handled).
- If interaction is enabled but `extra_info.interaction_kwargs.name` is missing, ToolAgentLoop raises.
- If `hydra-core` isn’t installed, `sv2/main_ppo_multiturn_toolcall.py` exits early with an install hint (same as `verl/trainer/main_ppo.py` would).
