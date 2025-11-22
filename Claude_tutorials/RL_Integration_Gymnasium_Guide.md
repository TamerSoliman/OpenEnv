# OpenEnv to Gymnasium Integration Guide

## Table of Contents
- [Introduction](#introduction)
- [Why Gymnasium?](#why-gymnasium)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Integration with RL Libraries](#integration-with-rl-libraries)
- [Custom Wrappers](#custom-wrappers)
- [Troubleshooting](#troubleshooting)

---

## Introduction

This guide explains how to use OpenEnv environments with the broader Gymnasium/RL ecosystem through the `OpenEnvGymnasiumWrapper`.

**What you'll learn:**
- Convert any OpenEnv environment to Gymnasium format
- Integrate with Stable-Baselines3, CleanRL, RLlib
- Handle text-based vs numeric observations/actions
- Create environment-specific wrappers

**Prerequisites:**
- Understanding of Gymnasium API
- Basic familiarity with OpenEnv
- Python 3.10+

---

## Why Gymnasium?

### The Gymnasium Ecosystem

[Gymnasium](https://gymnasium.farama.org/) (formerly OpenAI Gym) is the de facto standard API for RL environments. By wrapping OpenEnv environments in Gymnasium format, you gain access to:

**RL Libraries:**
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/): PPO, SAC, DQN, A2C, etc.
- [CleanRL](https://github.com/vwxyzjn/cleanrl): Single-file RL implementations
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/): Distributed RL at scale
- [TorchRL](https://pytorch.org/rl/): PyTorch-native RL library

**Tools:**
- [Stable-Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo): Trained models and hyperparameters
- [RL Baselines3 Zoo Contrib](https://github.com/Stable-Baselines-Team/rl-baselines3-zoo): Additional algorithms
- Logging, visualization, evaluation tools

### The Challenge: Text vs Numeric

Traditional RL environments (Atari, MuJoCo) use numeric observations and actions:
```python
# Traditional RL
observation = np.array([0.1, 0.5, -0.2])  # Numeric state
action = 2  # Discrete action or continuous vector
```

OpenEnv environments are text-based (designed for LLMs):
```python
# OpenEnv
observation = EchoObservation(echoed_message="Hello", message_length=5)
action = EchoAction(message="Test message")
```

**The wrapper bridges this gap** by providing multiple modes:
- **Text mode**: Keep text format (for LLM-based RL)
- **Discrete mode**: Map to discrete indices (for traditional RL algorithms)
- **Array mode**: Convert to numpy arrays (for neural network policies)

---

## Architecture Overview

### Wrapper Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    RL ALGORITHM                               │
│              (Stable-Baselines3, RLlib, etc.)                │
└──────────────────┬───────────────────────▲───────────────────┘
                   │ Gymnasium API         │ Gymnasium API
                   │ action (int/array)    │ (obs, reward, done)
                   │                       │
┌──────────────────▼───────────────────────┴───────────────────┐
│           OpenEnvGymnasiumWrapper                             │
│  • Converts Gym actions → OpenEnv actions                    │
│  • Converts OpenEnv observations → Gym observations          │
│  • Maps terminated/truncated flags                           │
│  • Handles episode resets and cleanup                        │
└──────────────────┬───────────────────────▲───────────────────┘
                   │ OpenEnv API           │ OpenEnv API
                   │ EchoAction            │ StepResult[EchoObs]
                   │                       │
┌──────────────────▼───────────────────────┴───────────────────┐
│                  OpenEnv HTTP Client                          │
│                  (EchoEnv, Connect4Env, etc.)                │
└──────────────────┬───────────────────────▲───────────────────┘
                   │ HTTP POST/GET         │ JSON Response
                   │                       │
┌──────────────────▼───────────────────────┴───────────────────┐
│              OpenEnv Environment Server                       │
│              (Running in Docker container)                    │
└───────────────────────────────────────────────────────────────┘
```

### Key Translations

| Aspect | Gymnasium | OpenEnv | Wrapper Handles |
|--------|-----------|---------|-----------------|
| **Reset** | `(obs, info) = reset()` | `StepResult = reset()` | Extracts obs, builds info |
| **Step** | `(obs, r, term, trunc, info) = step(action)` | `StepResult = step(Action)` | Maps done→term/trunc |
| **Action** | int or array | Action object | Creates Action from int |
| **Observation** | array or dict | Observation object | Extracts fields |

---

## Quick Start

### Installation

```bash
# Install OpenEnv
git clone https://github.com/meta-pytorch/OpenEnv.git
cd OpenEnv
pip install -e .

# Install Gymnasium
pip install gymnasium

# Optional: Install RL library
pip install stable-baselines3
```

### Basic Usage Example

```python
from envs.echo_env import EchoEnv
from Claude_tutorials.OpenEnvGymnasiumWrapper import EchoEnvGymnasiumWrapper

# 1. Create OpenEnv client
client = EchoEnv.from_docker_image("echo-env:latest")

# 2. Wrap with Gymnasium interface
gym_env = EchoEnvGymnasiumWrapper(client, max_episode_steps=100)

# 3. Use standard Gymnasium API
observation, info = gym_env.reset()
print(f"Initial obs: {observation}")

for step in range(10):
    # Sample random action
    action = gym_env.action_space.sample()

    # Execute action
    obs, reward, terminated, truncated, info = gym_env.step(action)

    print(f"Step {step}: reward={reward}, done={terminated or truncated}")

    if terminated or truncated:
        break

# 4. Cleanup
gym_env.close()
```

### Output

```
Initial obs: {'echoed_message': 'Echo environment ready!', 'message_length': array([0])}
Step 0: reward=0.5, done=False
Step 1: reward=1.3, done=False
Step 2: reward=0.4, done=False
...
```

---

## Advanced Usage

### Observation Modes

The wrapper supports three observation modes:

#### 1. Text Mode (Default for LLM agents)

```python
gym_env = OpenEnvGymnasiumWrapper(
    client,
    observation_mode="text",
)

obs, info = gym_env.reset()
# obs is a string: "EchoObservation(echoed_message='...', message_length=0)"
```

**Best for:**
- LLM-based policies
- Text-based RL algorithms
- Debugging (human-readable)

#### 2. Dictionary Mode (Structured data)

```python
gym_env = OpenEnvGymnasiumWrapper(
    client,
    observation_mode="dict",
)

obs, info = gym_env.reset()
# obs is a dict: {'echoed_message': '...', 'message_length': 0}
```

**Best for:**
- Multi-input neural networks
- Stable-Baselines3 MultiInputPolicy
- Structured observations

#### 3. Array Mode (Numeric)

```python
gym_env = OpenEnvGymnasiumWrapper(
    client,
    observation_mode="array",
)

obs, info = gym_env.reset()
# obs is numpy array: np.array([72, 101, 108, ...])  # Character codes
```

**Best for:**
- Traditional RL algorithms (PPO, DQN)
- Convolutional/MLP policies
- Numeric optimization

### Action Modes

#### 1. Text Mode (LLM-based)

```python
gym_env = OpenEnvGymnasiumWrapper(
    client,
    action_mode="text",
)

action = "Some text action"
obs, reward, terminated, truncated, info = gym_env.step(action)
```

**Use case:** LLM generates action strings directly

#### 2. Discrete Mode (Traditional RL)

```python
discrete_actions = [
    "Hello",
    "Test message",
    "Echo this",
    "Goodbye"
]

gym_env = OpenEnvGymnasiumWrapper(
    client,
    action_mode="discrete",
    discrete_actions=discrete_actions,
)

action = 0  # Maps to "Hello"
obs, reward, terminated, truncated, info = gym_env.step(action)
```

**Use case:** Traditional RL algorithms (PPO, DQN) with discrete action space

---

## Integration with RL Libraries

### Stable-Baselines3 (PPO)

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.echo_env import EchoEnv
from Claude_tutorials.OpenEnvGymnasiumWrapper import EchoEnvGymnasiumWrapper

# Create wrapped environment
client = EchoEnv.from_docker_image("echo-env:latest")
gym_env = EchoEnvGymnasiumWrapper(client, max_episode_steps=50)

# Verify Gym compatibility
check_env(gym_env)

# Create PPO model
model = PPO(
    "MultiInputPolicy",  # For dict observations
    gym_env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=64,
    batch_size=16,
)

# Train
model.learn(total_timesteps=10000)

# Save model
model.save("ppo_echo_env")

# Load and evaluate
model = PPO.load("ppo_echo_env")
obs, info = gym_env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = gym_env.step(action)
    if terminated or truncated:
        obs, info = gym_env.reset()

gym_env.close()
```

### CleanRL (DQN)

```python
import torch
import torch.nn as nn
from envs.connect4_env import Connect4Env
from Claude_tutorials.OpenEnvGymnasiumWrapper import OpenEnvGymnasiumWrapper

# Create environment
client = Connect4Env.from_docker_image("connect4-env:latest")

# Define discrete actions (column indices)
discrete_actions = [str(i) for i in range(7)]  # Columns 0-6

gym_env = OpenEnvGymnasiumWrapper(
    client,
    observation_mode="array",
    action_mode="discrete",
    discrete_actions=discrete_actions,
    max_episode_steps=42,  # Max moves in Connect4
)

# DQN Q-Network
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.network(x)

# Initialize network
obs_dim = gym_env.observation_space.shape[0]
action_dim = gym_env.action_space.n
q_network = QNetwork(obs_dim, action_dim)

# Training loop (simplified)
optimizer = torch.optim.Adam(q_network.parameters(), lr=0.001)

for episode in range(1000):
    obs, info = gym_env.reset()
    done = False

    while not done:
        # Epsilon-greedy action selection
        if torch.rand(1) < 0.1:
            action = gym_env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = q_network(torch.FloatTensor(obs))
                action = q_values.argmax().item()

        next_obs, reward, terminated, truncated, info = gym_env.step(action)
        done = terminated or truncated

        # Update network (simplified - no replay buffer)
        # ... DQN update logic ...

        obs = next_obs

    if episode % 100 == 0:
        print(f"Episode {episode} completed")

gym_env.close()
```

### Ray RLlib

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from envs.echo_env import EchoEnv
from Claude_tutorials.OpenEnvGymnasiumWrapper import EchoEnvGymnasiumWrapper

# Initialize Ray
ray.init()

# Environment factory function (required by RLlib)
def env_creator(env_config):
    client = EchoEnv.from_docker_image("echo-env:latest")
    return EchoEnvGymnasiumWrapper(client, max_episode_steps=50)

# Register environment
from ray.tune.registry import register_env
register_env("openenv-echo", env_creator)

# Configure PPO
config = (
    PPOConfig()
    .environment("openenv-echo")
    .rollouts(num_rollout_workers=4)
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=30,
    )
)

# Train
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=train.RunConfig(
        stop={"training_iteration": 100},
    ),
)

results = tuner.fit()

# Cleanup
ray.shutdown()
```

---

## Custom Wrappers

For environment-specific optimizations, create custom wrapper subclasses:

### Example: Connect4 Wrapper

```python
from Claude_tutorials.OpenEnvGymnasiumWrapper import OpenEnvGymnasiumWrapper
from gymnasium import spaces
import numpy as np

class Connect4GymnasiumWrapper(OpenEnvGymnasiumWrapper):
    """
    Specialized wrapper for Connect4 with board representation.
    """

    def __init__(self, openenv_client):
        # Define column actions (0-6)
        discrete_actions = [str(i) for i in range(7)]

        super().__init__(
            openenv_client=openenv_client,
            observation_mode="array",
            action_mode="discrete",
            discrete_actions=discrete_actions,
            max_episode_steps=42,  # Max possible moves
        )

        # Define observation space: 6x7 board + current player
        self.observation_space = spaces.Box(
            low=0, high=2,
            shape=(6, 7),
            dtype=np.int8
        )

    def _convert_action(self, gym_action):
        """Convert discrete action to Connect4Action."""
        from envs.connect4_env import Connect4Action
        return Connect4Action(column=gym_action)

    def _convert_observation(self, openenv_obs):
        """Convert Connect4Observation to numpy board."""
        # Extract board from observation
        board = np.array(openenv_obs.board, dtype=np.int8)
        return board
```

### Example: Coding Environment Wrapper

```python
class CodingEnvGymnasiumWrapper(OpenEnvGymnasiumWrapper):
    """
    Wrapper for Coding environment with code execution.
    """

    def __init__(self, openenv_client, max_code_length=1000):
        super().__init__(
            openenv_client=openenv_client,
            observation_mode="dict",
            action_mode="text",
            max_episode_steps=10,  # Max code submissions
        )

        # Define observation space
        self.observation_space = spaces.Dict({
            "stdout": spaces.Text(max_length=10000),
            "stderr": spaces.Text(max_length=10000),
            "exit_code": spaces.Discrete(256),  # 0-255
        })

    def _convert_action(self, gym_action):
        """Convert text to CodeAction."""
        from envs.coding_env import CodeAction
        return CodeAction(code=gym_action, language="python")

    def _convert_observation(self, openenv_obs):
        """Extract stdout, stderr, exit_code."""
        return {
            "stdout": openenv_obs.stdout,
            "stderr": openenv_obs.stderr,
            "exit_code": openenv_obs.exit_code,
        }

    def _compute_reward(self, openenv_obs):
        """Custom reward: +1 if code runs successfully."""
        if openenv_obs.exit_code == 0:
            return 1.0
        else:
            return -0.1  # Small penalty for errors
```

---

## Troubleshooting

### Issue 1: "Space type not supported"

**Error:**
```
TypeError: The observation returned by reset() is not in observation_space
```

**Solution:**
Define proper observation space for your environment:

```python
class MyCustomWrapper(OpenEnvGymnasiumWrapper):
    def __init__(self, client):
        super().__init__(client, observation_mode="dict")

        # Define specific observation space
        self.observation_space = spaces.Dict({
            "field1": spaces.Box(low=-1, high=1, shape=(10,)),
            "field2": spaces.Discrete(5),
        })
```

### Issue 2: "Action not valid"

**Error:**
```
AssertionError: Action must be in action_space
```

**Solution:**
Ensure actions are within defined action space:

```python
# Check action space
print(f"Action space: {gym_env.action_space}")

# Sample valid action
valid_action = gym_env.action_space.sample()
print(f"Valid action: {valid_action}")

# Use valid action
obs, reward, terminated, truncated, info = gym_env.step(valid_action)
```

### Issue 3: "Episode never terminates"

**Problem:** Episode runs forever

**Solution:**
Set `max_episode_steps` to enforce truncation:

```python
gym_env = OpenEnvGymnasiumWrapper(
    client,
    max_episode_steps=100,  # Force truncation after 100 steps
)
```

### Issue 4: "Docker container not found"

**Error:**
```
ConnectionError: Could not connect to environment server
```

**Solution:**
Ensure Docker container is running:

```bash
# Check running containers
docker ps

# Check if environment is accessible
curl http://localhost:8000/health

# Restart environment
docker restart <container_id>
```

---

## Best Practices

### 1. Always Check Environment Compatibility

```python
from stable_baselines3.common.env_checker import check_env

# Verify before training
check_env(gym_env)
```

### 2. Define Appropriate Spaces

```python
# Match observation space to actual observations
self.observation_space = spaces.Dict({
    "field1": spaces.Box(low=0, high=1, shape=(10,)),
    # NOT: spaces.Box(low=-np.inf, high=np.inf, shape=(100,))
})
```

### 3. Handle Episode Termination Properly

```python
# Distinguish natural termination from truncation
obs, reward, terminated, truncated, info = gym_env.step(action)

if terminated:
    print("Episode ended naturally (goal reached)")
elif truncated:
    print("Episode truncated (max steps)")
```

### 4. Cleanup Resources

```python
try:
    # Training loop
    ...
finally:
    # Always cleanup
    gym_env.close()
```

---

## Summary

**Key Takeaways:**
1. `OpenEnvGymnasiumWrapper` enables integration with entire Gymnasium ecosystem
2. Multiple modes support both text-based and numeric RL
3. Custom wrappers optimize for specific environments
4. Compatible with Stable-Baselines3, RLlib, CleanRL, TorchRL

**Next Steps:**
- Try basic example with Echo environment
- Integrate with your favorite RL library
- Create custom wrapper for your environment
- Train agents and compare performance

**Resources:**
- Gymnasium docs: https://gymnasium.farama.org/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- OpenEnv examples: `/home/user/OpenEnv/examples/`
