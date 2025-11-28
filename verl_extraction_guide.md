# VERL Training/Inference Code Extraction Guide

This guide shows exactly where to find the core training/inference code in VERL and what to copy for your custom implementation.

## **Core Training/Inference Code Locations**

### **1. Main Training Loop**
**File**: `/Users/ankit/Documents/dev/RL/verl/verl/trainer/ppo/ray_trainer.py`
- **Function**: `fit()` method in `RayPPOTrainer` class (around line 600-800)
- **What to copy**: The entire training orchestration logic including:
  - Data loading and batching
  - Worker coordination (actor, critic, reward model)
  - PPO step execution
  - Checkpointing logic

### **2. PPO Algorithm Core**
**File**: `/Users/ankit/Documents/dev/RL/verl/verl/trainer/ppo/core_algos.py`
- **Functions**: 
  - `compute_ppo_loss()` - Main PPO loss computation
  - `advantage_estimation()` - Advantage computation
  - `kl_penalty()` - KL divergence computation
- **What to copy**: All PPO algorithm functions (lines 50-500)

### **3. Model Training (Actor)**
**File**: `/Users/ankit/Documents/dev/RL/verl/verl/workers/actor/dp_actor.py`
- **Class**: `DataParallelPPOActor`
- **Key Methods**:
  - `forward()` - Forward pass for training
  - `compute_log_prob()` - Log probability computation
  - `update_policy()` - Policy update logic
- **What to copy**: The entire actor implementation (lines 49-528)

### **4. Inference/Generation**
**File**: `/Users/ankit/Documents/dev/RL/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- **Class**: `VLLMRolloutSPMD`
- **Key Methods**:
  - `generate_sequences()` - Main generation function
  - `inference()` - Model inference
  - `batch_generation()` - Batched inference
- **What to copy**: Generation and inference logic (lines 50-688)

### **5. Alternative Inference (SGLang)**
**File**: `/Users/ankit/Documents/dev/RL/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py`
- **What to copy**: If you prefer SGLang over VLLM for inference

### **6. Reward Computation**
**File**: `/Users/ankit/Documents/dev/RL/verl/verl/trainer/ppo/reward.py`
- **Functions**:
  - `compute_reward()` - Batch reward computation
  - `compute_reward_async()` - Async reward computation
- **What to copy**: Reward computation framework

### **7. Data Pipeline**
**File**: `/Users/ankit/Documents/dev/RL/verl/verl/utils/dataset/rl_dataset.py`
- **Class**: `RLHFDataset`
- **Functions**: `collate_fn()` - Data batching
- **What to copy**: Dataset loading and batching logic

## **What to Copy - Minimal Setup**

### **Essential Files to Copy:**
1. **`core_algos.py`** - PPO algorithm implementation
2. **`dp_actor.py`** - Model training logic
3. **`vllm_rollout_spmd.py`** - Inference/generation
4. **`reward.py`** - Reward computation framework
5. **`rl_dataset.py`** - Data handling

### **Key Functions to Extract:**

#### **Training:**
```python
# From core_algos.py
def compute_ppo_loss(old_log_probs, new_log_probs, advantages, response_mask, config)
def estimate_advantages(rewards, values, dones, gamma, lam)
def compute_kl_penalty(log_probs, ref_log_probs)

# From dp_actor.py  
class DataParallelPPOActor:
    def forward(self, data)
    def compute_log_prob(self, data)
    def update_policy(self, data)
```

#### **Inference:**
```python
# From vllm_rollout_spmd.py
class VLLMRolloutSPMD:
    def generate_sequences(self, prompts, **kwargs)
    def inference(self, inputs)
    def batch_generation(self, batch)
```

#### **Data:**
```python
# From rl_dataset.py
class RLHFDataset(Dataset)
def collate_fn(data_list)
```

## **Integration Points**

### **Where to Insert Your Custom Logic:**
1. **Reward Function**: Replace `compute_reward()` in `reward.py`
2. **Environment Interface**: Modify `RLHFDataset` to load your environment data
3. **Action Extraction**: Modify generation output parsing in rollout workers
4. **State Representation**: Update data preprocessing in dataset

### **Minimal Working Setup:**
Copy these 5 files and modify:
- **Dataset class** → Load your environment data
- **Reward function** → Compute your custom rewards  
- **Generation parsing** → Extract actions from model outputs
- **Training loop** → Keep as-is (PPO is standard)

The training orchestration in `ray_trainer.py` can remain largely unchanged - just point it to your custom dataset and reward functions.

## **File Structure for Custom Implementation**

```
your_custom_project/
├── training/
│   ├── core_algos.py          # Copy from verl/trainer/ppo/
│   ├── ray_trainer.py         # Copy from verl/trainer/ppo/
│   └── reward.py              # Copy from verl/trainer/ppo/
├── workers/
│   ├── dp_actor.py            # Copy from verl/workers/actor/
│   └── vllm_rollout_spmd.py   # Copy from verl/workers/rollout/
├── data/
│   └── rl_dataset.py          # Copy from verl/utils/dataset/
├── custom/
│   ├── env_dataset.py         # Your custom environment dataset
│   ├── reward_manager.py      # Your custom reward manager
│   └── env_wrapper.py         # Your environment wrapper
└── main.py                    # Your main training script
```

## **Step-by-Step Implementation**

### **Step 1: Copy Core Files**
```bash
# Copy the essential files
cp /Users/ankit/Documents/dev/RL/verl/verl/trainer/ppo/core_algos.py your_project/training/
cp /Users/ankit/Documents/dev/RL/verl/verl/trainer/ppo/ray_trainer.py your_project/training/
cp /Users/ankit/Documents/dev/RL/verl/verl/trainer/ppo/reward.py your_project/training/
cp /Users/ankit/Documents/dev/RL/verl/verl/workers/actor/dp_actor.py your_project/workers/
cp /Users/ankit/Documents/dev/RL/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py your_project/workers/
cp /Users/ankit/Documents/dev/RL/verl/verl/utils/dataset/rl_dataset.py your_project/data/
```

### **Step 2: Create Custom Environment Dataset**
```python
# custom/env_dataset.py
from torch.utils.data import Dataset
from your_project.data.rl_dataset import RLHFDataset

class CustomEnvDataset(RLHFDataset):
    def __init__(self, your_env, **kwargs):
        self.env = your_env
        # Load your environment data
        super().__init__(**kwargs)
    
    def __getitem__(self, idx):
        # Get environment state/action data
        state = self.env.get_state(idx)
        action = self.env.get_action(idx)
        
        # Format for language model
        prompt = self.state_to_text(state)
        
        # Tokenize
        tokens = self.tokenizer(prompt, **self.tokenizer_kwargs)
        
        return {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask'],
            'prompt': prompt,
            'state': state,
            'expected_action': action
        }
    
    def state_to_text(self, state):
        """Convert environment state to text prompt"""
        # Implement your state-to-text conversion
        pass
```

### **Step 3: Create Custom Reward Manager**
```python
# custom/reward_manager.py
from your_project.training.reward import compute_reward
from verl.protocol import DataProto
import torch

class CustomRewardManager:
    def __init__(self, environment, **kwargs):
        self.env = environment
    
    def __call__(self, data: DataProto):
        """Compute custom rewards"""
        rewards = []
        
        for i in range(len(data.batch)):
            # Get model response
            response = data.batch['responses'][i]
            
            # Extract action from response
            action = self.extract_action(response)
            
            # Get environment state
            state = data.batch['state'][i]
            
            # Compute reward in environment
            reward = self.env.compute_reward(state, action)
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def extract_action(self, response):
        """Extract action from model text response"""
        # Implement your action extraction logic
        pass
```

### **Step 4: Modify Main Training Script**
```python
# main.py
from your_project.training.ray_trainer import RayPPOTrainer
from your_project.custom.env_dataset import CustomEnvDataset
from your_project.custom.reward_manager import CustomRewardManager

def main():
    # Setup your environment
    env = YourEnvironment()
    
    # Create custom dataset
    dataset = CustomEnvDataset(env, tokenizer=tokenizer)
    
    # Create custom reward manager
    reward_fn = CustomRewardManager(env)
    
    # Initialize trainer with custom components
    trainer = RayPPOTrainer(
        config=config,
        train_dataset=dataset,
        reward_fn=reward_fn,
        # ... other parameters
    )
    
    # Start training
    trainer.fit()

if __name__ == "__main__":
    main()
```

## **Key Modifications Needed**

### **In `core_algos.py`:**
- No changes needed (PPO algorithm is standard)

### **In `dp_actor.py`:**
- No changes needed (actor training is standard)

### **In `vllm_rollout_spmd.py`:**
- Modify `generate_sequences()` to handle your specific prompt format
- Update response parsing to extract actions correctly

### **In `rl_dataset.py`:**
- Inherit and extend for your environment data format
- Modify `collate_fn()` if needed for your data structure

### **In `reward.py`:**
- Replace `compute_reward()` with your custom reward logic
- Keep the async framework if needed

## **Dependencies to Install**

```bash
# Core VERL dependencies
pip install torch transformers
pip install ray
pip install vllm  # or sglang
pip install omegaconf hydra-core
pip install tensordict

# Additional for custom environment
pip install gymnasium  # if using gym environments
pip install numpy pandas
```

## **Testing Your Setup**

### **Test Dataset Loading:**
```python
python -c "
from custom.env_dataset import CustomEnvDataset
from your_environment import YourEnvironment

env = YourEnvironment()
dataset = CustomEnvDataset(env)
print(f'Dataset size: {len(dataset)}')
print(f'Sample: {dataset[0]}')
"
```

### **Test Reward Computation:**
```python
python -c "
from custom.reward_manager import CustomRewardManager
from verl.protocol import DataProto

reward_fn = CustomRewardManager(env)
# Test with sample data
print('Reward manager working')
"
```

### **Test Training Loop:**
```python
python main.py --config-path=config --config-name=test_config
```

## **Common Issues and Solutions**

### **1. Import Errors:**
- Add all copied directories to PYTHONPATH
- Ensure relative imports are updated

### **2. Data Format Mismatches:**
- Check tensor shapes in collate_fn
- Verify tokenizer outputs match expected format

### **3. Reward Computation Errors:**
- Ensure reward function returns torch.Tensor
- Check device placement (CPU vs GPU)

### **4. Memory Issues:**
- Reduce batch sizes in config
- Enable gradient checkpointing

This setup gives you the core VERL training/inference infrastructure with hooks for your custom environment and rewards.

---

*Document created by Cascade, AI Assistant*

*Powered by Penguin Alpha model, created by Cognition*
