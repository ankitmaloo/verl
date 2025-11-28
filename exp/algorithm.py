#!/usr/bin/env python3
"""Main training loop using VERL components."""
import yaml
import torch
from verl import DataProto

from env import get_batch
from inference import init_rollout, generate
from training import init_training, compute_advantages, update_policy


def compute_reward(data: DataProto) -> torch.Tensor:
    """
    Score responses against ground truths.
    Modify this for your task.
    """
    responses = data.batch.get('responses', [])
    ground_truths = data.batch.get('ground_truths', [])

    rewards = torch.zeros(len(responses))
    for i, (resp, gt) in enumerate(zip(responses, ground_truths)):
        if gt and str(gt).lower() in str(resp).lower():
            rewards[i] = 1.0
        else:
            rewards[i] = 0.0

    return rewards


def main():
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Initialize VERL rollout (SGLang)
    print("Initializing rollout...")
    rollout = init_rollout(config)

    # Initialize VERL training
    print("Initializing training...")
    model = rollout.model  # Share model with rollout
    init_training(config, model)

    n = config['n_generations']

    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")
        epoch_rewards = []

        for step in range(config['steps_per_epoch']):
            # 1. Get batch from env
            batch = get_batch(size=config['batch_size'])
            prompts = batch['prompts']
            ground_truths = batch['ground_truths']

            # 2. Generate via VERL rollout
            output = generate(prompts, ground_truths, n=n)

            # 3. Compute rewards
            rewards = compute_reward(output)
            epoch_rewards.extend(rewards.tolist())

            # 4. Compute advantages (GRPO)
            advantages = compute_advantages(rewards)

            # 5. Expand advantages to sequence length
            log_probs = output.batch['log_probs']
            seq_len = log_probs.shape[1]
            advantages_expanded = advantages.unsqueeze(1).expand(-1, seq_len)

            # 6. Create response mask
            response_mask = torch.ones_like(log_probs)

            # 7. Update policy
            old_log_probs = log_probs.detach()
            metrics = update_policy(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages_expanded,
                response_mask=response_mask,
            )

            if (step + 1) % 10 == 0:
                avg_reward = sum(rewards.tolist()) / len(rewards)
                print(f"  Step {step + 1}: loss={metrics['policy_loss']:.4f}, reward={avg_reward:.2f}")

        # Epoch summary
        avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
        print(f"Epoch {epoch + 1} avg reward: {avg_epoch_reward:.3f}")

        # Save checkpoint
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
            }, f'checkpoint_epoch_{epoch + 1}.pt')
            print(f"Saved checkpoint_epoch_{epoch + 1}.pt")


if __name__ == '__main__':
    main()
