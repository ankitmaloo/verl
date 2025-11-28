"""Inference using VERL rollout workers."""
from verl import DataProto

# SGLang: best for multi-turn + tool calling
from verl.workers.rollout.sglang_rollout.sglang_rollout import SglangRollout

# Alternatives:
# from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import VllmRollout
# from verl.workers.rollout.hf_rollout import HFRollout

_rollout = None


def init_rollout(config):
    """Initialize rollout worker (call once)."""
    global _rollout

    rollout_config = {
        'model_path': config['model_path'],
        'temperature': config['inference']['temperature'],
        'top_p': config['inference']['top_p'],
        'top_k': config['inference'].get('top_k', 50),
        'max_response_length': config['inference']['max_response_length'],
        'n': config['inference']['n'],
        'tensor_parallel_size': config.get('num_gpus', 1),
    }

    # Add tool config if specified
    if config['inference'].get('tools'):
        rollout_config['tools'] = config['inference']['tools']

    _rollout = SglangRollout(rollout_config, role='rollout')
    return _rollout


def generate(prompts, ground_truths=None, n=1):
    """
    Generate n responses per prompt using VERL rollout.

    Args:
        prompts: List[str] or List[List[dict]] for multi-turn
        ground_truths: Optional list of answers
        n: Generations per prompt

    Returns:
        DataProto with responses, log_probs
    """
    if _rollout is None:
        raise RuntimeError("Call init_rollout(config) first")

    # Build batch
    batch_dict = {
        'prompts': prompts,
    }
    if ground_truths:
        batch_dict['ground_truths'] = ground_truths

    batch = DataProto.from_single_dict(batch_dict)

    # Repeat for n generations per prompt
    if n > 1:
        batch = batch.repeat(n, interleave=True)

    # Generate via VERL rollout
    output = _rollout.generate_sequences(batch)

    return output


def get_rollout():
    """Get the rollout worker for direct access."""
    return _rollout
