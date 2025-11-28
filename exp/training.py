"""Training using VERL actor/critic workers."""
import torch
from verl.trainer.ppo.core_algos import (
    compute_policy_loss,
    compute_grpo_outcome_advantage,
    compute_gae_advantage,
)

# VERL workers for distributed training
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.critic.dp_critic import DataParallelPPOCritic

_actor = None
_critic = None
_config = None


def init_training(config, model):
    """Initialize actor/critic workers."""
    global _actor, _critic, _config
    _config = config

    actor_config = {
        'model': model,
        'strategy': config['training']['actor']['strategy'],
        'optim': {
            'lr': config['training']['actor']['lr'],
        },
        'max_grad_norm': config['training']['actor'].get('max_grad_norm', 1.0),
        'use_kl_loss': config['training']['actor'].get('use_kl_loss', False),
        'kl_loss_coef': config['training']['actor'].get('kl_loss_coef', 0.001),
    }

    _actor = DataParallelPPOActor(actor_config, role='actor')

    # Critic is optional (not needed for GRPO)
    if config['algorithm']['adv_estimator'] == 'gae':
        critic_config = {
            'model': model,
            'strategy': config['training']['critic']['strategy'],
            'optim': {
                'lr': config['training']['critic']['lr'],
            },
        }
        _critic = DataParallelPPOCritic(critic_config, role='critic')


def compute_advantages(rewards, values=None, response_mask=None):
    """Compute advantages using configured estimator."""
    adv_type = _config['algorithm']['adv_estimator']

    if adv_type == 'grpo':
        # GRPO: group-relative advantages
        n_per_prompt = _config['n_generations']
        advantages = compute_grpo_outcome_advantage(
            rewards=rewards,
            n=n_per_prompt,
        )
    elif adv_type == 'gae':
        # GAE: requires value function
        if values is None:
            raise ValueError("GAE requires value estimates")
        advantages, returns = compute_gae_advantage(
            rewards=rewards,
            values=values,
            gamma=_config['algorithm']['gamma'],
            gae_lambda=_config['algorithm']['gae_lambda'],
        )
    else:
        # Simple: just normalize rewards
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    return advantages


def update_policy(log_probs, old_log_probs, advantages, response_mask):
    """Update policy using VERL's loss function."""
    loss_fn = _config['algorithm']['policy_loss_fn']

    # Compute loss using VERL
    policy_loss, loss_info = compute_policy_loss(
        old_log_probs=old_log_probs,
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode='token_level',
        algorithm_config=_config['algorithm'],
    )

    # Backward pass via actor worker
    _actor.backward(policy_loss)

    return {
        'policy_loss': policy_loss.item(),
        **loss_info,
    }


def update_critic(values, returns, response_mask):
    """Update value function (for GAE)."""
    if _critic is None:
        return {}

    value_loss = ((values - returns) ** 2 * response_mask).sum() / response_mask.sum()
    _critic.backward(value_loss)

    return {'value_loss': value_loss.item()}


def get_actor():
    return _actor


def get_critic():
    return _critic
