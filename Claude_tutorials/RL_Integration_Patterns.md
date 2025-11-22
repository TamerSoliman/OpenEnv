# RL Integration Patterns for OpenEnv

## Table of Contents
- [Introduction](#introduction)
- [Core Integration Pattern](#core-integration-pattern)
- [Stable-Baselines3 Integration](#stable-baselines3-integration)
- [TorchRL Integration](#torchrl-integration)
- [Ray RLlib Integration](#ray-rllib-integration)
- [Forge Integration (LLM-Based RL)](#forge-integration-llm-based-rl)
- [Custom Training Loops](#custom-training-loops)
- [Logging and Monitoring](#logging-and-monitoring)
- [Best Practices](#best-practices)

---

## Introduction

This guide provides **architectural patterns** for integrating OpenEnv with popular RL frameworks. Each pattern includes:
- High-level architecture diagram
- Code structure template
- Key integration points
- Common pitfalls and solutions

**Philosophy:** These are patterns, not full implementations. Adapt them to your specific needs.

---

## Core Integration Pattern

All RL integrations follow this base pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYERS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] Training Infrastructure (RLlib/SB3/Custom)                 │
│      ↕ (config, checkpoints, distributed coordination)          │
│                                                                  │
│  [2] Environment Adapter (Gymnasium wrapper or custom)          │
│      ↕ (action/obs conversion, episode management)              │
│                                                                  │
│  [3] OpenEnv HTTP Client (Echo/Connect4/Git/etc.)              │
│      ↕ (HTTP requests, JSON serialization)                      │
│                                                                  │
│  [4] OpenEnv Environment Server (Docker container)              │
│      ↕ (environment logic, state management)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Universal Integration Template

```python
# Step 1: Create OpenEnv environment
from envs.echo_env import EchoEnv
client = EchoEnv.from_docker_image("echo-env:latest")

# Step 2: Create adapter (Gymnasium wrapper or custom)
from Claude_tutorials.OpenEnvGymnasiumWrapper import EchoEnvGymnasiumWrapper
env = EchoEnvGymnasiumWrapper(client, max_episode_steps=100)

# Step 3: Initialize RL algorithm with adapted environment
from rl_library import Algorithm
agent = Algorithm(env, config={...})

# Step 4: Train
agent.train(num_steps=10000)

# Step 5: Evaluate
rewards = agent.evaluate(num_episodes=100)

# Step 6: Cleanup
env.close()
```

---

## Stable-Baselines3 Integration

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   STABLE-BASELINES3                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  PPO/SAC/DQN │  │   Policies   │  │  Buffers     │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │ (algo logic)    │ (neural nets)   │ (replay)        │
│         └─────────┬───────┴─────────┬───────┘                 │
│                   ▼                 ▼                          │
│         ┌─────────────────────────────────────┐                │
│         │   VecEnv (vectorized envs)          │                │
│         └─────────┬───────────────────────────┘                │
│                   │                                            │
└───────────────────┼────────────────────────────────────────────┘
                    │ Gymnasium API
┌───────────────────▼────────────────────────────────────────────┐
│         OpenEnvGymnasiumWrapper                                 │
│         (converts Gym ↔ OpenEnv)                               │
└───────────────────┬────────────────────────────────────────────┘
                    │ OpenEnv API
┌───────────────────▼────────────────────────────────────────────┐
│         OpenEnv Environment (Echo, Connect4, Git, etc.)         │
└────────────────────────────────────────────────────────────────┘
```

### Pattern 1: Single Environment Training

**Use case:** Basic training with one environment instance

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.echo_env import EchoEnv
from Claude_tutorials.OpenEnvGymnasiumWrapper import EchoEnvGymnasiumWrapper

def train_single_env():
    """Train PPO on single OpenEnv environment."""

    # Create and wrap environment
    client = EchoEnv.from_docker_image("echo-env:latest")
    env = EchoEnvGymnasiumWrapper(client, max_episode_steps=50)

    # Verify environment
    check_env(env)

    # Create model
    model = PPO(
        "MultiInputPolicy",  # For dict observations
        env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )

    # Train
    model.learn(
        total_timesteps=100000,
        progress_bar=True,
    )

    # Save
    model.save("ppo_echo_env")

    # Cleanup
    env.close()

    return model
```

### Pattern 2: Vectorized Environments

**Use case:** Parallel data collection for faster training

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from envs.echo_env import EchoEnv
from Claude_tutorials.OpenEnvGymnasiumWrapper import EchoEnvGymnasiumWrapper

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    Args:
        rank: Index of subprocess
        seed: Random seed

    Returns:
        Function that creates environment
    """
    def _init():
        # Create OpenEnv client (each process gets its own)
        client = EchoEnv.from_docker_image("echo-env:latest")
        env = EchoEnvGymnasiumWrapper(client, max_episode_steps=50)
        env.reset(seed=seed + rank)
        return env
    return _init

def train_vectorized():
    """Train with vectorized environments for parallel data collection."""

    # Create 4 parallel environments
    num_envs = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Create model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=512,  # Steps per env, total = 512 * 4 = 2048
        batch_size=64,
    )

    # Train (4x faster data collection!)
    model.learn(total_timesteps=100000)

    # Save
    model.save("ppo_echo_vectorized")

    # Cleanup
    env.close()

    return model
```

**Note:** Each subprocess creates its own Docker container, so ensure you have sufficient resources.

### Pattern 3: Custom Policies for Text Environments

**Use case:** Custom neural network architecture for text observations

```python
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from gymnasium import spaces

class TextEncodingPolicy(ActorCriticPolicy):
    """
    Custom policy that encodes text observations.

    Architecture:
    - Text → Embedding layer → LSTM → MLP → Actions
    """

    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        # Add custom network architecture
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

        # Custom feature extractor for text
        # (This is simplified - real implementation would handle text properly)

    def _build_mlp_extractor(self):
        """Override to add text processing layers."""
        # Custom feature extraction for text observations
        pass

# Usage
model = PPO(
    TextEncodingPolicy,  # Custom policy
    env,
    verbose=1,
)
```

---

## TorchRL Integration

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        TORCHRL                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Loss Module │  │   Policy Net │  │  Replay Buf  │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └─────────┬───────┴─────────┬───────┘                 │
│                   ▼                 ▼                          │
│         ┌─────────────────────────────────────┐                │
│         │   EnvBase (TorchRL env wrapper)     │                │
│         └─────────┬───────────────────────────┘                │
│                   │                                            │
└───────────────────┼────────────────────────────────────────────┘
                    │ TorchRL TensorDict API
┌───────────────────▼────────────────────────────────────────────┐
│         OpenEnv Custom Wrapper                                  │
│         (converts TensorDict ↔ OpenEnv)                        │
└───────────────────┬────────────────────────────────────────────┘
                    │ OpenEnv API
┌───────────────────▼────────────────────────────────────────────┐
│         OpenEnv Environment                                     │
└────────────────────────────────────────────────────────────────┘
```

### Pattern: TorchRL Custom Environment

**Use case:** PyTorch-native RL with TorchRL

```python
from torchrl.envs import EnvBase
from torchrl.data import TensorDict
from tensordict import TensorDict
import torch

class OpenEnvTorchRLWrapper(EnvBase):
    """
    Wrapper that converts OpenEnv to TorchRL environment.

    TorchRL uses TensorDict for all data, which is PyTorch-native.
    """

    def __init__(self, openenv_client, device="cpu"):
        super().__init__(device=device)
        self.openenv_client = openenv_client
        self._current_step = 0

    def _reset(self, tensordict=None):
        """
        Reset environment and return TensorDict observation.

        TorchRL expects TensorDict with:
        - observation: tensor of observations
        - done: tensor of done flags
        """
        result = self.openenv_client.reset()
        self._current_step = 0

        # Convert observation to tensor
        # (Simplified - real implementation would handle different obs types)
        obs_tensor = torch.tensor([0.0], device=self.device)  # Placeholder

        return TensorDict({
            "observation": obs_tensor,
            "done": torch.tensor([False], device=self.device),
        }, batch_size=[])

    def _step(self, tensordict):
        """
        Execute action and return TensorDict with results.

        TorchRL passes actions as TensorDict and expects
        TensorDict with observation, reward, done.
        """
        # Extract action from tensordict
        action_tensor = tensordict["action"]
        action_str = self._tensor_to_action(action_tensor)

        # Execute in OpenEnv
        result = self.openenv_client.step(action_str)
        self._current_step += 1

        # Convert to TensorDict
        obs_tensor = torch.tensor([0.0], device=self.device)  # Placeholder
        reward_tensor = torch.tensor([result.reward], device=self.device)
        done_tensor = torch.tensor([result.done], device=self.device)

        return TensorDict({
            "observation": obs_tensor,
            "reward": reward_tensor,
            "done": done_tensor,
        }, batch_size=[])

    def _tensor_to_action(self, tensor):
        """Convert tensor to OpenEnv action."""
        # Implement conversion logic
        pass

# Usage
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import DQNLoss

# Create environment
client = EchoEnv.from_docker_image("echo-env:latest")
env = OpenEnvTorchRLWrapper(client)

# Create collector
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=1000,
)

# Collect data
for data in collector:
    # data is a TensorDict
    loss = loss_module(data)
    # ... training step ...
```

---

## Ray RLlib Integration

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                          RAY RLLIB                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Trainer    │  │  Workers     │  │  Replay Pool │         │
│  │   (Driver)   │  │ (Rollouts)   │  │  (Optional)  │         │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘         │
│         │                 │                                     │
│         └─────────┬───────┘                                     │
│                   ▼                                             │
│         ┌─────────────────────────────────────┐                 │
│         │   Env Registry + Worker Envs       │                 │
│         └─────────┬───────────────────────────┘                 │
│                   │                                             │
└───────────────────┼─────────────────────────────────────────────┘
                    │ Gymnasium API
┌───────────────────▼─────────────────────────────────────────────┐
│         OpenEnvGymnasiumWrapper (on each worker)                 │
└───────────────────┬─────────────────────────────────────────────┘
                    │ OpenEnv API
┌───────────────────▼─────────────────────────────────────────────┐
│         OpenEnv Environment (separate instance per worker)       │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern: Distributed Training with RLlib

**Use case:** Large-scale distributed training

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from envs.echo_env import EchoEnv
from Claude_tutorials.OpenEnvGymnasiumWrapper import EchoEnvGymnasiumWrapper

def env_creator(env_config):
    """
    Environment factory function.

    RLlib calls this on each worker to create environment instances.

    Args:
        env_config: Dict with environment configuration

    Returns:
        Gymnasium-wrapped OpenEnv environment
    """
    # Each worker creates its own OpenEnv client
    client = EchoEnv.from_docker_image("echo-env:latest")
    env = EchoEnvGymnasiumWrapper(
        client,
        max_episode_steps=env_config.get("max_steps", 100)
    )
    return env

def train_with_rllib():
    """Train with Ray RLlib for distributed RL."""

    # Initialize Ray
    ray.init(num_cpus=8, num_gpus=1)

    # Register environment
    register_env("openenv-echo", env_creator)

    # Configure algorithm
    config = (
        PPOConfig()
        .environment(
            "openenv-echo",
            env_config={"max_steps": 50},
        )
        .framework("torch")
        .rollouts(
            num_rollout_workers=4,      # 4 parallel workers
            num_envs_per_worker=2,       # 2 envs per worker = 8 total
        )
        .training(
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
            lr=5e-5,
        )
        .resources(
            num_gpus=1,
            num_cpus_per_worker=1,
        )
    )

    # Train
    tuner = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=tune.RunConfig(
            stop={"training_iteration": 100},
            checkpoint_config=tune.CheckpointConfig(
                checkpoint_frequency=10,
            ),
        ),
    )

    results = tuner.fit()

    # Get best checkpoint
    best_result = results.get_best_result()
    print(f"Best checkpoint: {best_result.checkpoint}")

    # Cleanup
    ray.shutdown()
```

---

## Forge Integration (LLM-Based RL)

### Architecture

Based on the existing `examples/grpo_blackjack/` example in the repository.

```
┌────────────────────────────────────────────────────────────────┐
│                          FORGE                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Generator   │  │  RLTrainer   │  │ Replay Buf   │         │
│  │   (vLLM)     │  │   (FSDP)     │  │ (TorchStore) │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │ (LLM inference) │ (RL updates)    │ (off-policy)    │
│         └─────────┬───────┴─────────┬───────┘                 │
│                   ▼                 ▼                          │
│         ┌─────────────────────────────────────┐                │
│         │   GRPO Training Loop                │                │
│         └─────────┬───────────────────────────┘                │
│                   │                                            │
└───────────────────┼────────────────────────────────────────────┘
                    │ OpenEnv API (text-based)
┌───────────────────▼────────────────────────────────────────────┐
│         OpenEnv Environment (OpenSpiel, TextArena, etc.)        │
└────────────────────────────────────────────────────────────────┘
```

### Pattern: GRPO Training (from grpo_blackjack example)

**Use case:** Training LLMs with reinforcement learning

**Key files:** `/home/user/OpenEnv/examples/grpo_blackjack/`

```python
# Based on examples/grpo_blackjack/grpo_utils.py
# This is a conceptual pattern, not full implementation

import asyncio
from forge import Generator, RLTrainer, ReplayBuffer

async def train_llm_with_grpo(env_name: str, config_path: str):
    """
    Train LLM with GRPO on OpenEnv environment.

    GRPO (Group Relative Policy Optimization):
    - Sample multiple responses per prompt
    - Compute advantages relative to group
    - Update policy with PPO-style loss
    - Only needs 2 models (policy + reference) instead of 3

    Args:
        env_name: OpenEnv environment name
        config_path: Path to YAML config file
    """

    # 1. Load configuration
    config = load_config(config_path)

    # 2. Initialize Forge components
    generator = await Generator.create(config["generator"])
    trainer = await RLTrainer.create(config["trainer"])
    buffer = ReplayBuffer(config["buffer"])

    # 3. Connect to OpenEnv environment
    env_client = create_openenv_client(env_name)

    # 4. Training loop
    for iteration in range(config["num_iterations"]):

        # Collect rollouts (LLM generates multiple responses)
        rollouts = await collect_rollouts(
            generator=generator,
            env=env_client,
            num_episodes=config["episodes_per_iteration"],
            group_size=config["group_size"],  # Multiple samples per prompt
        )

        # Compute advantages (group-relative)
        advantages = compute_group_advantages(rollouts)

        # Add to buffer
        buffer.add(rollouts, advantages)

        # Update policy
        if buffer.size >= config["min_buffer_size"]:
            batch = buffer.sample(config["batch_size"])
            loss = trainer.update(batch)

            print(f"Iteration {iteration}: Loss = {loss}")

        # Save checkpoint
        if iteration % config["save_frequency"] == 0:
            await trainer.save_checkpoint(f"ckpt_{iteration}")

    # Cleanup
    env_client.close()

def collect_rollouts(generator, env, num_episodes, group_size):
    """
    Collect rollouts from environment.

    For each episode:
    1. Reset environment, get task/prompt
    2. Generate multiple responses (group_size)
    3. Execute each response in environment
    4. Collect rewards

    Returns:
        List of rollouts with (prompt, responses, rewards)
    """
    rollouts = []

    for episode in range(num_episodes):
        # Get task
        result = env.reset()
        prompt = format_prompt(result.observation)

        # Generate multiple responses
        responses = []
        rewards = []

        for _ in range(group_size):
            # LLM generates response
            response = generator.generate(prompt)
            responses.append(response)

            # Execute in environment
            action = parse_action(response)
            result = env.step(action)
            rewards.append(result.reward)

        rollouts.append({
            "prompt": prompt,
            "responses": responses,
            "rewards": rewards,
        })

    return rollouts

def compute_group_advantages(rollouts):
    """
    Compute advantages relative to group.

    For each response in a group:
        advantage = reward - mean(group_rewards)

    This is the key GRPO innovation: advantages are relative
    to the group, not an absolute baseline.
    """
    advantages = []

    for rollout in rollouts:
        rewards = rollout["rewards"]
        mean_reward = sum(rewards) / len(rewards)

        # Advantage = reward - group mean
        adv = [r - mean_reward for r in rewards]
        advantages.append(adv)

    return advantages
```

**Key configuration (blackjack.yaml):**
```yaml
generator:
  model: "meta-llama/Llama-2-7b-hf"
  vllm_config:
    tensor_parallel_size: 1

trainer:
  learning_rate: 5e-6
  beta: 0.01  # KL penalty coefficient

buffer:
  capacity: 10000

# Training settings
num_iterations: 100
episodes_per_iteration: 32
group_size: 4  # Sample 4 responses per prompt
batch_size: 128
min_buffer_size: 512
save_frequency: 10
```

---

## Custom Training Loops

### Pattern: Simple PyTorch Training Loop

**Use case:** Full control over training, no frameworks

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class SimplePolicy(nn.Module):
    """Simple MLP policy for discrete actions."""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs):
        return self.network(obs)

def train_custom_loop(env, num_episodes=1000):
    """
    Custom training loop with simple policy gradient.

    Algorithm: REINFORCE
    - Collect full episodes
    - Compute returns
    - Update policy to increase probability of high-return actions
    """

    # Initialize policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy = SimplePolicy(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    # Training loop
    for episode in range(num_episodes):
        # Collect episode
        observations = []
        actions = []
        rewards = []

        obs, info = env.reset()
        done = False

        while not done:
            # Select action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            logits = policy(obs_tensor)
            action_probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            observations.append(obs_tensor)
            actions.append(action)
            rewards.append(reward)

            obs = next_obs

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G  # Gamma = 0.99
            returns.insert(0, G)

        # Convert to tensors
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize

        # Compute loss
        loss = 0
        for obs_tensor, action, G in zip(observations, actions, returns):
            logits = policy(obs_tensor)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_prob = log_probs[0, action]
            loss -= log_prob * G  # Negative because we want to maximize

        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress
        if episode % 100 == 0:
            total_reward = sum(rewards)
            print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

    return policy
```

---

## Logging and Monitoring

### Pattern: TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class TrainingLogger:
    """Logging wrapper for training metrics."""

    def __init__(self, log_dir="./logs"):
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = []
        self.episode = 0

    def log_episode(self, total_reward, episode_length, additional_metrics=None):
        """Log metrics for completed episode."""
        self.episode += 1
        self.episode_rewards.append(total_reward)

        # Log to TensorBoard
        self.writer.add_scalar("Episode/Reward", total_reward, self.episode)
        self.writer.add_scalar("Episode/Length", episode_length, self.episode)

        # Running average
        if len(self.episode_rewards) >= 100:
            avg_reward = np.mean(self.episode_rewards[-100:])
            self.writer.add_scalar("Episode/Reward_100ep_avg", avg_reward, self.episode)

        # Additional metrics
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.writer.add_scalar(f"Episode/{key}", value, self.episode)

    def log_training_step(self, step, loss, learning_rate):
        """Log training step metrics."""
        self.writer.add_scalar("Training/Loss", loss, step)
        self.writer.add_scalar("Training/LearningRate", learning_rate, step)

    def close(self):
        """Close writer."""
        self.writer.close()

# Usage
logger = TrainingLogger(log_dir="./runs/exp1")

for episode in range(1000):
    # ... training loop ...
    logger.log_episode(
        total_reward=sum(rewards),
        episode_length=len(rewards),
        additional_metrics={"success_rate": success, "steps": steps}
    )

logger.close()
```

---

## Best Practices

### 1. Environment Lifecycle Management

```python
from contextlib import contextmanager

@contextmanager
def openenv_session(env_class, docker_image, **kwargs):
    """
    Context manager for OpenEnv environments.

    Ensures proper cleanup even if errors occur.
    """
    client = None
    env = None

    try:
        # Create client
        client = env_class.from_docker_image(docker_image)

        # Wrap if needed
        if "wrapper" in kwargs:
            wrapper_class = kwargs.pop("wrapper")
            env = wrapper_class(client, **kwargs)
        else:
            env = client

        yield env

    finally:
        # Always cleanup
        if env is not None:
            env.close()
        elif client is not None:
            client.close()

# Usage
with openenv_session(EchoEnv, "echo-env:latest", wrapper=EchoEnvGymnasiumWrapper) as env:
    # Training code here
    model.learn(env, total_timesteps=10000)
# Environment automatically cleaned up
```

### 2. Checkpointing and Recovery

```python
def train_with_checkpointing(env, num_steps, checkpoint_freq=1000):
    """Train with periodic checkpointing."""

    # Try to load existing checkpoint
    start_step = 0
    if os.path.exists("checkpoint.pt"):
        checkpoint = torch.load("checkpoint.pt")
        policy.load_state_dict(checkpoint["policy"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        print(f"Resuming from step {start_step}")

    # Training loop
    for step in range(start_step, num_steps):
        # ... training code ...

        # Save checkpoint
        if step % checkpoint_freq == 0:
            checkpoint = {
                "step": step,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, "checkpoint.pt")
            print(f"Checkpoint saved at step {step}")
```

### 3. Hyperparameter Configuration

```python
# config.yaml
training:
  algorithm: "PPO"
  total_timesteps: 100000
  learning_rate: 0.0003
  batch_size: 64

environment:
  name: "echo-env"
  docker_image: "echo-env:latest"
  max_episode_steps: 100

logging:
  tensorboard_log: "./logs"
  save_freq: 10000
```

```python
import yaml

def train_from_config(config_path):
    """Train from YAML configuration."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create environment
    env = create_env_from_config(config["environment"])

    # Create algorithm
    algorithm = create_algorithm_from_config(
        config["training"],
        env,
        config["logging"]
    )

    # Train
    algorithm.learn(total_timesteps=config["training"]["total_timesteps"])
```

---

## Summary

**Key Takeaways:**
1. All RL frameworks follow similar integration patterns
2. Gymnasium wrapper enables broad compatibility
3. Vectorization and distribution scale training
4. Custom loops provide maximum flexibility
5. Proper logging and checkpointing are essential

**Integration Compatibility Matrix:**

| Framework | Complexity | Scaling | LLM Support | Best For |
|-----------|-----------|---------|-------------|----------|
| **Stable-Baselines3** | Low | Moderate | No | Quick experiments, standard algorithms |
| **Ray RLlib** | High | Excellent | Partial | Large-scale distributed training |
| **TorchRL** | Moderate | Good | No | PyTorch-native, custom algorithms |
| **Forge** | Moderate | Excellent | Yes | LLM-based RL (GRPO, RLHF) |
| **Custom** | Low | Manual | Yes | Full control, research |

**Recommended Path:**
1. Start with Stable-Baselines3 for standard RL algorithms
2. Use Forge for LLM-based RL (following grpo_blackjack example)
3. Scale with Ray RLlib for distributed training
4. Build custom loops for novel algorithms

**Next Steps:**
- Choose framework based on your use case
- Adapt patterns to your specific environment
- Start with single environment, then scale
- Monitor training with TensorBoard
- Iterate on hyperparameters
