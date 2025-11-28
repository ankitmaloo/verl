"""
VERL Custom Training Setup Package.

Provides core components for integrating custom environments with VERL's PPO training pipeline.

Modules:
    - env.py: Environment base classes and configuration
    - algo.py: Core PPO algorithm functions
    - inference.py: VLLM and SGLang inference engines
    - trainer.py: Ray-based PPO trainer

Usage:
    from setup import ConfigStore, BaseEnvironment, RayPPOTrainer
    
    # Define your environment
    class MyEnv(BaseEnvironment):
        def reset(self): ...
        def step(self, action): ...
        def get_state_description(self): ...
    
    # Create config
    config = ConfigStore()
    config.model.model_name = "meta-llama/Llama-2-7b"
    config.training.num_epochs = 10
    
    # Initialize trainer
    env = MyEnv(config.environment)
    trainer = RayPPOTrainer(config, env, reward_fn=lambda x: 1.0)
    
    # Train
    trainer.fit()
"""

from .env import (
    ConfigStore,
    ModelConfig,
    InferenceConfig,
    RewardConfig,
    EnvironmentConfig,
    TrainingConfig,
    RayConfig,
    CheckpointConfig,
    BaseEnvironment,
    SimpleGameEnvironment,
    Trajectory,
    RolloutData,
    create_default_config,
    merge_configs,
)

from .algo import (
    AdvantageEstimator,
    compute_gae_advantage_return,
    compute_grpo_advantage,
    compute_advantage,
    compute_policy_loss,
    compute_value_loss,
    kl_divergence,
    apply_kl_penalty,
    KLController,
)

from .inference import (
    BaseInferenceEngine,
    VLLMInferenceEngine,
    SGLangInferenceEngine,
    MultiTurnInferenceEngine,
    create_inference_engine,
    BatchInferenceManager,
    GenerationOutput,
)

from .trainer import (
    RayPPOTrainer,
    PolicyHead,
    ValueHead,
    PolicyValueModel,
    PPOBatch,
)

from .tools import (
    ToolDefinition,
    ToolParameter,
    Tool,
    ToolRegistry,
    ParsedAction,
    ActionParser,
    JSONActionParser,
    ReActActionParser,
    FunctionCallingParser,
    MultiTurnManager,
    ToolCall,
    Turn,
    EnvironmentStepTool,
    QueryTool,
    CalculateTool,
)

__all__ = [
    # Config
    "ConfigStore",
    "ModelConfig",
    "InferenceConfig",
    "RewardConfig",
    "EnvironmentConfig",
    "TrainingConfig",
    "RayConfig",
    "CheckpointConfig",
    # Environment
    "BaseEnvironment",
    "SimpleGameEnvironment",
    "Trajectory",
    "RolloutData",
    "create_default_config",
    "merge_configs",
    # Algorithm
    "AdvantageEstimator",
    "compute_gae_advantage_return",
    "compute_grpo_advantage",
    "compute_advantage",
    "compute_policy_loss",
    "compute_value_loss",
    "kl_divergence",
    "apply_kl_penalty",
    "KLController",
    # Inference
    "BaseInferenceEngine",
    "VLLMInferenceEngine",
    "SGLangInferenceEngine",
    "MultiTurnInferenceEngine",
    "create_inference_engine",
    "BatchInferenceManager",
    "GenerationOutput",
    # Trainer
    "RayPPOTrainer",
    "PolicyHead",
    "ValueHead",
    "PolicyValueModel",
    "PPOBatch",
    # Tools
    "ToolDefinition",
    "ToolParameter",
    "Tool",
    "ToolRegistry",
    "ParsedAction",
    "ActionParser",
    "JSONActionParser",
    "ReActActionParser",
    "FunctionCallingParser",
    "MultiTurnManager",
    "ToolCall",
    "Turn",
    "EnvironmentStepTool",
    "QueryTool",
    "CalculateTool",
]

__version__ = "0.1.0"
