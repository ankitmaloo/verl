#!/usr/bin/env python3
"""Main training script with proper dataset iteration."""

import yaml
import torch
import wandb
from huggingface_hub import HfApi

from env import Env
from datasets import get_dataset
from inference import init_rollout, generate
from training import init_training, compute_advantages, update_policy


def compute_reward(games):
    """Compute rewards from game instances."""
    rewards = []
    for g in games:
        if g.attempts:
            r = g.calculate_reward(g.attempts[-1])
        else:
            r = 0.0
        rewards.append(r)
    return torch.tensor(rewards)


def save_to_hf(model, tokenizer, repo_id, config=None):
    """Upload model to HuggingFace Hub."""
    import os
    save_dir = f"./hf_upload_{repo_id.split('/')[-1]}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    if config:
        with open(f"{save_dir}/training_config.yaml", "w") as f:
            yaml.dump(config, f)
    api = HfApi()
    api.upload_folder(folder_path=save_dir, repo_id=repo_id, repo_type="model")


def run_games(games, max_turns=50):
    """Run a batch of games to completion."""
    env = Env(games)
    size = len(games)
    all_log_probs = [[] for _ in range(size)]

    while not env.done():
        prompts = env.get_prompts()
        active_idx = [i for i, p in enumerate(prompts) if p is not None]
        active_prompts = [prompts[i] for i in active_idx]

        if not active_prompts:
            break

        output = generate(active_prompts, n=1)
        responses = output.batch.get('responses', [])
        log_probs = output.batch.get('log_probs', None)

        if hasattr(responses, 'tolist'):
            responses = responses.tolist()
        if isinstance(responses[0], list):
            responses = [''.join(map(str, r)) for r in responses]

        full_responses = [None] * size
        for j, idx in enumerate(active_idx):
            full_responses[idx] = responses[j] if j < len(responses) else ""
            if log_probs is not None and j < len(log_probs):
                all_log_probs[idx].append(log_probs[j])

        env.step(full_responses)

        if len(env.convs[0]) > max_turns * 2:
            break

    batch = env.get_batch()
    return batch, all_log_probs


def main():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    wandb.init(
        project=config.get('wandb', {}).get('project', 'verl-games'),
        name=config.get('wandb', {}).get('name', None),
        config=config,
    )

    print("Initializing rollout...")
    rollout = init_rollout(config)

    print("Initializing training...")
    model = rollout.model
    tokenizer = rollout.tokenizer
    init_training(config, model)

    # Get dataset
    game_name = config.get('game', {}).get('name', 'password')
    max_turns = config.get('game', {}).get('max_turns', 50)
    batch_size = config['batch_size']

    print(f"Loading dataset: {game_name}")
    dataset = get_dataset(game_name)

    # Determine batches per epoch
    if hasattr(dataset, '__len__') and dataset.__len__() != float('inf'):
        batches_per_epoch = len(dataset) // batch_size
        print(f"Dataset size: {len(dataset)}, batches per epoch: {batches_per_epoch}")
    else:
        batches_per_epoch = config.get('steps_per_epoch', 100)
        print(f"Infinite dataset, using {batches_per_epoch} steps per epoch")

    for epoch in range(config['num_epochs']):
        print(f"\n=== Epoch {epoch + 1}/{config['num_epochs']} ===")
        dataset.reset()
        epoch_rewards = []

        for step in range(batches_per_epoch):
            games = dataset.get_batch(batch_size)
            batch, all_log_probs = run_games(games, max_turns)

            rewards = compute_reward(batch['games'])
            epoch_rewards.extend(rewards.tolist())

            advantages = compute_advantages(rewards)

            # Flatten log probs per trajectory
            flat_log_probs = []
            for lps in all_log_probs:
                if lps:
                    flat_log_probs.append(torch.cat([lp.flatten() for lp in lps]))
                else:
                    flat_log_probs.append(torch.zeros(1))

            max_len = max(lp.shape[0] for lp in flat_log_probs)
            padded = []
            for lp in flat_log_probs:
                if lp.shape[0] < max_len:
                    lp = torch.cat([lp, torch.zeros(max_len - lp.shape[0])])
                padded.append(lp)
            log_probs = torch.stack(padded)

            seq_len = log_probs.shape[1]
            advantages_expanded = advantages.unsqueeze(1).expand(-1, seq_len)
            response_mask = torch.ones_like(log_probs)

            old_log_probs = log_probs.detach()
            metrics = update_policy(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages_expanded,
                response_mask=response_mask,
            )

            avg_reward = rewards.mean().item()
            wandb.log({
                'step': epoch * batches_per_epoch + step,
                'loss': metrics['policy_loss'],
                'reward': avg_reward,
            })

            if (step + 1) % 10 == 0:
                print(f"  Step {step + 1}/{batches_per_epoch}: loss={metrics['policy_loss']:.4f}, reward={avg_reward:.2f}")

        avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        wandb.log({'epoch': epoch + 1, 'epoch_reward': avg_epoch_reward})
        print(f"Epoch {epoch + 1} avg reward: {avg_epoch_reward:.3f}")

        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
            }, f'checkpoint_epoch_{epoch + 1}.pt')
            print(f"Saved checkpoint_epoch_{epoch + 1}.pt")

    wandb.finish()

    hf_repo = config.get('hf_repo')
    if hf_repo:
        print(f"Uploading to HuggingFace: {hf_repo}")
        save_to_hf(model, tokenizer, hf_repo, config)


if __name__ == '__main__':
    main()
