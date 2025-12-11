# Adding a Custom Rollout Extension (multi-turn, tool-using agent)

Goal: describe how to add a new rollout style that keeps the full multi-turn trace (e.g., 15-step agent → tool → agent → user → …) while integrating with verl’s existing hooks.

## What exists today
- Research-side extension points cover: `extensions/custom_advantages.py`, `extensions/custom_losses.py`, `extensions/custom_rewards.py`, `extensions/custom_samplers.py`.
- Rollout plumbing lives in verl proper:
  - Multi-turn agent loop engine: `verl/experimental/agent_loop` (see `tool_agent_loop.py` for tool + interaction flow, `single_turn_agent_loop.py` for the simplest case).
  - Config knobs: `verl/trainer/config/rollout/rollout.yaml` (`multi_turn` block and `agent` block).
  - Tool registry & schemas: `verl/tools` (`base_tool.py`, `utils/tool_registry.py`, sample schema in `examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml`).
  - Interaction (user/env feedback) registry: `verl/interactions` (`base.py`, `utils/interaction_registry.py`, sample config in `examples/sglang_multiturn/config/interaction_config/gsm8k_interaction_config.yaml`).
  - Dataset expectations: each sample exposes `raw_prompt` (list of chat messages) plus optional `extra_info` / `tools_kwargs` used by the agent loop (`verl/utils/dataset/rl_dataset.py`).
  - Walkthrough docs: `verl-research/docs/CUSTOM_MULTITURN_ENV.md` and the `examples/sglang_multiturn/*` configs.

## How to add a custom rollout (keeping long workflows + “thinking” tokens)
1) **Implement your agent loop**
   - Create a module in research, e.g., `extensions/custom_rollouts.py`, and subclass `verl.experimental.agent_loop.tool_agent_loop.ToolAgentLoop` (best starting point for tool + interaction heavy flows).
   - Override pieces you need:
     - Termination: relax limits in `_handle_generating_state`/`_handle_processing_tools_state` to allow ~15 steps (and increase `response_length` accordingly).
     - Thinking visibility: keep `use_inference_chat_template=False` in config so token IDs include the model’s reasoning, or add your own flag to avoid stripping rationale before training.
     - Custom user turns: extend `_handle_interacting_state` to inject scripted/user turns or extra rewards per turn.
   - Minimal skeleton:
     ```python
     from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

     class LongWorkflowAgent(ToolAgentLoop):
         async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
             # call super() then change stop rules / add logging
             state = await super()._handle_generating_state(
                 agent_data, sampling_params, ignore_termination=True
             )
             # e.g., cap at 15 assistant turns
             if agent_data.assistant_turns >= 15:
                 return agent_state.AgentState.TERMINATED
             return state
     ```

2) **Register the agent loop so Hydra can find it**
   - Add a YAML next to your experiment, e.g., `extensions/custom_rollouts.yaml`:
     ```yaml
     - name: long_workflow
       _target_: extensions.custom_rollouts.LongWorkflowAgent
       # optional custom args here
     ```
   - Point the trainer to it via config: set `actor_rollout_ref.rollout.agent.agent_loop_config_path` to that YAML, and choose it per-sample via `agent_name` in the dataset or as `default_agent_loop`.

3) **Enable multi-turn rollout + configure depth**
   - In your experiment’s `config.yaml`, set:
     ```yaml
     actor_rollout_ref:
       rollout:
         multi_turn:
           enable: true
           max_assistant_turns: 15   # or higher
           max_user_turns: 15        # if you alternate user feedback
           tool_config_path: "<path to your tool config>"
           interaction_config_path: "<path to your interaction config>"
           use_inference_chat_template: false  # keep reasoning tokens
         prompt_length: 2048        # make room for many turns
         response_length: 2048      # must fit tool responses + thinking
         agent:
           default_agent_loop: long_workflow
           agent_loop_config_path: "<path to custom_rollouts.yaml>"
     ```
   - Tool config must describe the callable APIs (see GSM8K sample), and interaction config can provide per-turn feedback/reward logic.

4) **Prep data so the loop has what it needs**
   - Each row should include `raw_prompt` as a list of `{role, content}` messages. Include any tool/user control knobs in `extra_info` (e.g., `interaction_kwargs`) and per-tool args in `tools_kwargs`.
   - If you need 15 explicit alternations, encode the starting context accordingly (e.g., seed with a system + user turn and let the agent loop handle the rest).

5) **Test quickly before long runs**
   - Use `examples/sglang_multiturn/run_*` scripts as a template to sanity-check config wiring.
   - Run the research `tools/quick_test.py` to catch import/config issues.
   - Validate tokenization sanity: set `multi_turn.tokenization_sanity_check_mode=ignore_strippable` if your model’s chat template differs turn-by-turn.

## Open questions for you
- How do you want the “thinking” tokens treated in rewards? Keep them in the loss, strip from reward shaping, or log-only?
- What tool stack will the 15-step workflow use (existing gsm8k/search, MCP tools, or new ones)? Need schemas to size `response_length`.
- Should user turns be scripted (interaction) or come from a live source? The implementation differs (`Interaction` vs external feed).
- Do you want per-turn rewards (dense) or only at the end? That changes how we extend `calculate_score` in your Interaction class.
