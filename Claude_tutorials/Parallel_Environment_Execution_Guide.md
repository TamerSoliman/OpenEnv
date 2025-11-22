# Parallel Environment Execution Guide

## Table of Contents
1. [Overview](#overview)
2. [Why Parallel Execution?](#why-parallel-execution)
3. [Architecture Patterns](#architecture-patterns)
4. [Threading vs Async vs Multiprocessing](#threading-vs-async-vs-multiprocessing)
5. [Use Cases](#use-cases)
6. [Performance Considerations](#performance-considerations)
7. [Integration Examples](#integration-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

Parallel environment execution allows you to run multiple OpenEnv environments simultaneously, dramatically reducing wall-clock time for data collection, training, and evaluation. This guide covers three main parallel execution patterns:

1. **Thread-based Execution**: Using `ThreadPoolExecutor` for I/O-bound operations
2. **Async Execution**: Using `asyncio` for high-concurrency scenarios
3. **Process-based Execution**: Using `ProcessPoolExecutor` for CPU-bound operations (limited applicability for OpenEnv)

**Key Benefits:**
- **Faster Training**: Collect rollouts from multiple environments simultaneously
- **Curriculum Learning**: Run environments at different difficulty levels in parallel
- **Multi-Task Learning**: Train on multiple tasks concurrently
- **Efficient Evaluation**: Test agents across multiple scenarios simultaneously

---

## Why Parallel Execution?

### The Sequential Bottleneck

Traditional RL training collects data sequentially:

```python
# Sequential rollout collection (SLOW)
for i in range(num_episodes):
    obs = env.reset()
    for step in range(max_steps):
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        # ... store data
```

**Problem**: Each environment must wait for the previous one to complete.

**Time complexity**: `O(num_episodes * avg_episode_length * step_time)`

### The Parallel Solution

With parallel execution:

```python
# Parallel rollout collection (FAST)
collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)
rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)
```

**Benefit**: All environments run simultaneously.

**Time complexity**: `O(avg_episode_length * step_time)` for 8 episodes

**Speedup**: ~8x faster with 8 workers (actual speedup depends on network latency)

### Real-World Example

**Scenario**: Training an agent on Echo environment for 1000 episodes

**Sequential timing**:
- Average episode: 20 steps × 50ms/step = 1 second
- Total time: 1000 episodes × 1 second = **1000 seconds (16.7 minutes)**

**Parallel timing (8 workers)**:
- Collect 8 episodes at once
- Total time: (1000 / 8) × 1 second = **125 seconds (2.1 minutes)**
- **Speedup**: 8x faster

---

## Architecture Patterns

### Pattern 1: Thread-Based Parallel Execution

**Best for**: OpenEnv environments (I/O-bound HTTP requests)

```
┌─────────────────────────────────────────────────────────┐
│          ParallelEnvironmentExecutor                     │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         ThreadPoolExecutor (4 threads)           │  │
│  │                                                  │  │
│  │  Thread 1 ──► HttpEnvClient ──► Environment 1   │  │
│  │  Thread 2 ──► HttpEnvClient ──► Environment 2   │  │
│  │  Thread 3 ──► HttpEnvClient ──► Environment 3   │  │
│  │  Thread 4 ──► HttpEnvClient ──► Environment 4   │  │
│  │                                                  │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│                  List[Rollout]                          │
└─────────────────────────────────────────────────────────┘
```

**How it works**:
1. Create `ThreadPoolExecutor` with `max_workers` threads
2. Submit `collect_rollout()` task for each environment
3. Each thread runs episode independently
4. Results collected as threads complete

**Code reference**: See `ParallelEnvironmentExecutor` in `Parallel_Environment_Execution.py:88-191`

### Pattern 2: Async-Based Execution

**Best for**: Very high concurrency (50+ environments), fine-grained control

```
┌─────────────────────────────────────────────────────────┐
│            AsyncParallelExecutor                         │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         asyncio Event Loop                       │  │
│  │                                                  │  │
│  │  Coroutine 1 ──► to_thread ──► Environment 1    │  │
│  │  Coroutine 2 ──► to_thread ──► Environment 2    │  │
│  │  Coroutine 3 ──► to_thread ──► Environment 3    │  │
│  │       ...                                        │  │
│  │  Coroutine 50 ──► to_thread ──► Environment 50  │  │
│  │                                                  │  │
│  └──────────────────────────────────────────────────┘  │
│                          │                              │
│                          ▼                              │
│                  List[Rollout]                          │
└─────────────────────────────────────────────────────────┘
```

**How it works**:
1. Create async coroutine for each environment
2. Use `asyncio.to_thread()` to offload blocking I/O
3. `asyncio.gather()` runs all coroutines concurrently
4. Results returned in submission order

**Code reference**: See `AsyncParallelExecutor` in `Parallel_Environment_Execution.py:194-296`

### Pattern 3: High-Level Collector with Statistics

**Best for**: Production use cases, curriculum learning, monitoring

```
┌─────────────────────────────────────────────────────────┐
│         ParallelRolloutCollector                         │
│                                                          │
│  1. Execute Parallel Collection                         │
│     ┌────────────────────────────────┐                  │
│     │   Thread/Async Executor        │                  │
│     └────────────────────────────────┘                  │
│                  │                                       │
│                  ▼                                       │
│  2. Collect Rollouts                                    │
│     [Rollout₁, Rollout₂, ..., Rolloutₙ]                │
│                  │                                       │
│                  ▼                                       │
│  3. Compute Statistics                                  │
│     ┌────────────────────────────────┐                  │
│     │ • Mean/Std Return              │                  │
│     │ • Per-Environment Breakdown    │                  │
│     │ • Episode Lengths              │                  │
│     │ • Total Steps                  │                  │
│     └────────────────────────────────┘                  │
│                  │                                       │
│                  ▼                                       │
│  4. Return (Rollouts, Stats)                           │
└─────────────────────────────────────────────────────────┘
```

**How it works**:
1. Wraps thread or async executor
2. Collects rollouts using chosen execution mode
3. Automatically computes aggregated statistics
4. Returns both rollouts and stats for analysis

**Code reference**: See `ParallelRolloutCollector` in `Parallel_Environment_Execution.py:299-398`

---

## Threading vs Async vs Multiprocessing

### Comparison Table

| Criterion | Threading | Async | Multiprocessing |
|-----------|-----------|-------|-----------------|
| **Best for** | I/O-bound (HTTP) | Very high concurrency | CPU-bound |
| **OpenEnv suitability** | ✅ **Excellent** | ✅ Good | ❌ Limited |
| **Max concurrency** | ~10-20 workers | 50+ coroutines | # of CPU cores |
| **Memory overhead** | Medium | Low | High |
| **Ease of use** | ✅ Simple | Moderate | Complex |
| **GIL impact** | None (I/O releases GIL) | None | None |
| **Pickling required** | No | No | Yes (problematic) |
| **Typical speedup** | 4-8x | 4-8x | N/A for OpenEnv |

### Why Threading is Best for OpenEnv

OpenEnv environments make HTTP requests, which are **I/O-bound**:

1. **Network Latency Dominates**: Most time spent waiting for server responses
2. **GIL Not a Problem**: I/O operations release the GIL
3. **Shared Memory**: Easy to share agent policy across threads
4. **No Pickling**: Avoid serialization overhead

**Recommendation**: Use `ParallelEnvironmentExecutor` (threading) for most OpenEnv use cases.

### When to Use Async

Use `AsyncParallelExecutor` when:
- Running 50+ environments simultaneously
- Need fine-grained control over concurrency
- Want lower memory overhead than threading
- Integrating with async-native frameworks

### Why Not Multiprocessing?

Multiprocessing is **not recommended** for OpenEnv because:
1. **Pickling Overhead**: Environment clients must be serialized
2. **Memory Duplication**: Each process has separate memory
3. **No Benefit**: OpenEnv is I/O-bound, not CPU-bound
4. **Complexity**: Inter-process communication is harder

**Exception**: Use multiprocessing if you have CPU-intensive agent policies that can't be batched.

---

## Use Cases

### Use Case 1: Faster Training Data Collection

**Problem**: Need 10,000 environment steps for training, but sequential collection takes too long.

**Solution**: Parallel rollout collection

```python
from Claude_tutorials.Parallel_Environment_Execution import ParallelRolloutCollector

# Create 8 environment clients
env_clients = [HttpEnvClient("http://localhost:8000") for _ in range(8)]

# Create collector
collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)

# Collect 8 episodes in parallel (repeat until 10k steps)
total_steps = 0
all_rollouts = []

while total_steps < 10000:
    rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)
    all_rollouts.extend(rollouts)
    total_steps += stats.total_steps
    print(f"Collected {total_steps}/10000 steps")

collector.shutdown()
```

**Benefit**: ~8x faster than sequential collection

### Use Case 2: Curriculum Learning

**Problem**: Train agent progressively on easy → medium → hard tasks.

**Solution**: Parallel execution across difficulty levels with adaptive sampling

```python
from Claude_tutorials.Parallel_Environment_Execution import ParallelRolloutCollector

# Create environments at different difficulty levels
easy_clients = [HttpEnvClient("http://localhost:8000") for _ in range(4)]
medium_clients = [HttpEnvClient("http://localhost:8001") for _ in range(2)]
hard_clients = [HttpEnvClient("http://localhost:8002") for _ in range(2)]

all_clients = easy_clients + medium_clients + hard_clients
env_names = (["easy"] * 4) + (["medium"] * 2) + (["hard"] * 2)

collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)

for iteration in range(100):
    rollouts, stats = collector.collect_with_stats(all_clients, agent_fn, env_names=env_names)

    # Make curriculum decisions based on per-environment performance
    print(f"\nIteration {iteration}")
    print(f"Easy: {stats.env_breakdown['easy']:.2f}")
    print(f"Medium: {stats.env_breakdown['medium']:.2f}")
    print(f"Hard: {stats.env_breakdown['hard']:.2f}")

    # Adaptive curriculum: if easy is mastered, increase medium proportion
    if stats.env_breakdown["easy"] > 15.0:
        # Adjust environment proportions (implementation-specific)
        print("→ Agent mastered easy! Increasing medium proportion")

collector.shutdown()
```

**Benefit**: Monitor progress across all difficulty levels simultaneously

### Use Case 3: Multi-Task Learning

**Problem**: Train agent on multiple different environments (e.g., Echo + Connect4).

**Solution**: Parallel collection from different environment types

```python
from Claude_tutorials.Parallel_Environment_Execution import ParallelRolloutCollector

# Different environment types
echo_clients = [HttpEnvClient("http://localhost:8000") for _ in range(4)]
connect4_clients = [HttpEnvClient("http://localhost:8001") for _ in range(4)]

all_clients = echo_clients + connect4_clients
env_names = (["echo"] * 4) + (["connect4"] * 4)

collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)

rollouts, stats = collector.collect_with_stats(all_clients, agent_fn, env_names=env_names)

# Train on mixed data
print(f"Echo performance: {stats.env_breakdown['echo']:.2f}")
print(f"Connect4 performance: {stats.env_breakdown['connect4']:.2f}")

collector.shutdown()
```

**Benefit**: Efficient multi-task data collection

### Use Case 4: Hyperparameter Search

**Problem**: Test different agent hyperparameters across multiple runs.

**Solution**: Parallel evaluation of different agent configurations

```python
from Claude_tutorials.Parallel_Environment_Execution import ParallelEnvironmentExecutor

def create_agent_fn(temperature):
    """Create agent with specific temperature parameter."""
    def agent_fn(obs, done):
        # Use temperature in policy
        return sample_action(obs, temperature=temperature)
    return agent_fn

# Test 4 different temperatures in parallel
temperatures = [0.5, 0.7, 0.9, 1.1]
env_clients = [HttpEnvClient("http://localhost:8000") for _ in range(4)]
executor = ParallelEnvironmentExecutor(max_workers=4)

futures = []
for temp, client in zip(temperatures, env_clients):
    agent_fn = create_agent_fn(temp)
    future = executor.executor.submit(
        executor.collect_rollout, client, agent_fn, 1000, f"temp_{temp}"
    )
    futures.append(future)

# Collect results
from concurrent.futures import as_completed
for future in as_completed(futures):
    rollout = future.result()
    print(f"{rollout.env_name}: return={rollout.episode_return:.2f}")

executor.shutdown()
```

**Benefit**: Test multiple hyperparameters simultaneously

---

## Performance Considerations

### Optimal Worker Count

**Rule of thumb**: Start with `max_workers = 4-8` for OpenEnv

**Factors to consider**:
1. **Network bandwidth**: More workers = more simultaneous HTTP requests
2. **Server capacity**: Don't overwhelm the environment server
3. **Memory**: Each worker needs memory for episode data
4. **Diminishing returns**: Beyond 8-16 workers, speedup plateaus

**Empirical testing**:
```python
import time

for num_workers in [1, 2, 4, 8, 16]:
    collector = ParallelRolloutCollector(execution_mode="thread", max_workers=num_workers)

    start = time.time()
    rollouts, stats = collector.collect_with_stats(env_clients[:num_workers], agent_fn)
    elapsed = time.time() - start

    print(f"Workers: {num_workers}, Time: {elapsed:.2f}s, Speedup: {1.0/elapsed:.2f}x")
    collector.shutdown()
```

**Expected results**:
- 1 worker: baseline
- 2 workers: ~1.8x speedup
- 4 workers: ~3.5x speedup
- 8 workers: ~6.0x speedup
- 16 workers: ~7.0x speedup (diminishing returns)

### Memory Management

**Memory usage per worker**:
- Environment state: ~1-10 MB (depends on observation size)
- Episode buffer: ~1 MB per episode (depends on max_steps)
- Thread overhead: ~8 MB per thread

**Total memory estimate**: `num_workers × (env_state + episode_buffer + thread_overhead)`

**For 8 workers**: ~80-160 MB (manageable)

**Optimization tips**:
1. **Stream rollouts**: Process and discard data as it arrives
2. **Limit episode length**: Use `max_steps` to bound memory
3. **Clear buffers**: Delete processed rollouts to free memory

```python
# Memory-efficient streaming collection
collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)

for batch_idx in range(100):
    rollouts, stats = collector.collect_with_stats(env_clients, agent_fn, max_steps=100)

    # Process and train on rollouts
    train_agent(rollouts)

    # Clear rollouts to free memory
    del rollouts
```

### Network Considerations

**Bandwidth requirements**:
- Typical OpenEnv request: ~1-10 KB
- Typical OpenEnv response: ~1-10 KB
- Per step: ~2-20 KB round-trip

**For 8 workers at 20 steps/sec**: ~320-3200 KB/sec = **0.3-3 MB/sec**

**Latency sensitivity**:
- Local server: ~1-10 ms per step
- Remote server: ~50-200 ms per step

**Recommendation**: Run environment servers on same machine or local network for best performance.

---

## Integration Examples

### Integration 1: Stable-Baselines3 with Parallel Collection

Combine parallel rollout collection with SB3 off-policy training:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Claude_tutorials.OpenEnvGymnasiumWrapper import OpenEnvGymnasiumWrapper
from Claude_tutorials.Parallel_Environment_Execution import ParallelRolloutCollector

# Create parallel environments
def make_env(env_url):
    def _init():
        client = HttpEnvClient(env_url)
        return OpenEnvGymnasiumWrapper(client, observation_mode="text", action_mode="discrete")
    return _init

# SB3 expects vectorized environments
vec_env = DummyVecEnv([make_env("http://localhost:8000") for _ in range(4)])

# Train with PPO (automatically uses parallel envs)
model = PPO("MlpPolicy", vec_env, verbose=1, n_steps=128)
model.learn(total_timesteps=10000)
```

**Note**: SB3's `DummyVecEnv` runs environments sequentially. For true parallelism, use `SubprocVecEnv` or our custom parallel collector.

### Integration 2: Custom PyTorch Training Loop

Use parallel collection with custom training:

```python
import torch
from Claude_tutorials.Parallel_Environment_Execution import ParallelRolloutCollector

# Initialize policy
policy = MyPolicyNetwork()
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

# Create parallel collector
env_clients = [HttpEnvClient("http://localhost:8000") for _ in range(8)]
collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)

# Training loop
for epoch in range(100):
    # Collect rollouts in parallel
    def agent_fn(obs, done):
        with torch.no_grad():
            return policy.get_action(obs)

    rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)

    # Convert rollouts to tensors
    observations = torch.tensor([r.observations for r in rollouts])
    actions = torch.tensor([r.actions for r in rollouts])
    rewards = torch.tensor([r.rewards for r in rollouts])

    # Compute loss and update
    loss = compute_policy_loss(observations, actions, rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}: Mean return = {stats.mean_return:.2f}")

collector.shutdown()
```

### Integration 3: Ray RLlib with Parallel Envs

Use Ray RLlib's distributed training with OpenEnv:

```python
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from Claude_tutorials.OpenEnvGymnasiumWrapper import OpenEnvGymnasiumWrapper

# Ray automatically parallelizes environments
config = {
    "env": OpenEnvGymnasiumWrapper,
    "env_config": {
        "env_url": "http://localhost:8000",
        "observation_mode": "text",
        "action_mode": "discrete"
    },
    "num_workers": 4,  # 4 parallel workers
    "num_envs_per_worker": 2,  # 2 envs per worker = 8 total
    "framework": "torch",
}

# Train with Ray
ray.init()
tune.run(PPO, config=config, stop={"timesteps_total": 100000})
```

**Note**: Ray handles parallelization automatically. Our parallel collector is useful for custom workflows.

---

## Best Practices

### 1. Resource Management

**Always clean up resources**:
```python
collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)
try:
    rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)
finally:
    collector.shutdown()  # Ensures threads are cleaned up
```

**Use context managers** (if implementing custom collectors):
```python
class ParallelCollectorContext:
    def __enter__(self):
        self.collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)
        return self.collector

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.collector.shutdown()

with ParallelCollectorContext() as collector:
    rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)
```

### 2. Error Handling

**Handle individual environment failures gracefully**:
```python
def robust_collect_rollout(env_client, agent_fn, max_steps, env_name):
    try:
        return executor.collect_rollout(env_client, agent_fn, max_steps, env_name)
    except Exception as e:
        print(f"Environment {env_name} failed: {e}")
        return None  # Return None for failed rollouts

# Filter out None values
rollouts = [r for r in rollouts if r is not None]
```

**Retry failed environments**:
```python
def collect_with_retry(env_client, agent_fn, max_retries=3):
    for attempt in range(max_retries):
        try:
            return executor.collect_rollout(env_client, agent_fn, 1000, "env")
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Retry {attempt + 1}/{max_retries}")
            time.sleep(1)
```

### 3. Monitoring and Logging

**Log statistics during training**:
```python
import wandb

wandb.init(project="openenv-training")

for epoch in range(100):
    rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)

    # Log to Weights & Biases
    wandb.log({
        "mean_return": stats.mean_return,
        "std_return": stats.std_return,
        "mean_length": stats.mean_length,
        "total_steps": stats.total_steps,
        **{f"return/{env}": ret for env, ret in stats.env_breakdown.items()}
    })
```

**Track per-environment metrics**:
```python
from collections import defaultdict

env_metrics = defaultdict(list)

for epoch in range(100):
    rollouts, stats = collector.collect_with_stats(env_clients, agent_fn, env_names=env_names)

    for env_name, mean_return in stats.env_breakdown.items():
        env_metrics[env_name].append(mean_return)

    # Visualize trends
    if epoch % 10 == 0:
        for env_name, returns in env_metrics.items():
            print(f"{env_name}: {returns[-10:]}")  # Last 10 epochs
```

### 4. Deterministic Execution

**Set seeds for reproducibility**:
```python
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# Note: Thread scheduling is non-deterministic, so exact rollout order may vary
# But individual rollouts should be deterministic if agent is seeded
```

### 5. Profiling and Optimization

**Profile to find bottlenecks**:
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 time-consuming functions
```

**Identify slow components**:
- If `env_client.step()` is slow: Check network latency
- If `agent_fn()` is slow: Optimize policy inference (batching, GPU)
- If thread creation is slow: Reuse thread pool across epochs

---

## Troubleshooting

### Issue 1: Slower than Expected

**Symptoms**: Parallel execution only 2x faster with 8 workers

**Possible causes**:
1. **Network bottleneck**: Server can't handle 8 simultaneous requests
2. **CPU-bound agent**: Agent inference is the bottleneck, not environment
3. **GIL contention**: Rare for I/O-bound, but possible if agent is CPU-heavy

**Solutions**:
```python
# Profile to identify bottleneck
import time

# Test environment speed
start = time.time()
for _ in range(100):
    result = env_client.step(action)
elapsed = time.time() - start
print(f"Environment: {elapsed/100*1000:.2f} ms/step")

# Test agent speed
start = time.time()
for _ in range(100):
    action = agent_fn(obs, False)
elapsed = time.time() - start
print(f"Agent: {elapsed/100*1000:.2f} ms/step")

# If agent is slow, batch inference
def batched_agent_fn(observations):
    # Run inference on batch of observations
    return policy.predict_batch(observations)
```

### Issue 2: Memory Errors

**Symptoms**: `MemoryError` or system slowdown

**Possible causes**:
1. Too many workers
2. Very long episodes
3. Large observation spaces
4. Memory leaks

**Solutions**:
```python
# Reduce workers
collector = ParallelRolloutCollector(execution_mode="thread", max_workers=4)  # Down from 8

# Limit episode length
rollouts, stats = collector.collect_with_stats(env_clients, agent_fn, max_steps=100)

# Stream and discard data
for batch in range(100):
    rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)
    process_and_train(rollouts)
    del rollouts  # Free memory immediately

# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

### Issue 3: Inconsistent Results

**Symptoms**: Different results across runs

**Possible causes**:
1. Non-deterministic thread scheduling
2. Race conditions in agent
3. Environment randomness

**Solutions**:
```python
# Set all seeds
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Use deterministic algorithms
torch.use_deterministic_algorithms(True)

# For thread-safe agent, use locks
import threading

class ThreadSafeAgent:
    def __init__(self, policy):
        self.policy = policy
        self.lock = threading.Lock()

    def get_action(self, obs):
        with self.lock:
            return self.policy(obs)
```

### Issue 4: Environment Server Overload

**Symptoms**: Timeouts, connection errors, slow responses

**Possible causes**:
1. Too many simultaneous requests
2. Server not scaled for parallel load

**Solutions**:
```python
# Reduce concurrent requests
collector = ParallelRolloutCollector(execution_mode="thread", max_workers=2)

# Add request throttling
import time

def throttled_step(env_client, action):
    result = env_client.step(action)
    time.sleep(0.01)  # 10ms delay between requests
    return result

# Scale up environment servers (run multiple instances)
env_clients = [
    HttpEnvClient("http://localhost:8000"),  # Server 1
    HttpEnvClient("http://localhost:8001"),  # Server 2
    HttpEnvClient("http://localhost:8002"),  # Server 3
    HttpEnvClient("http://localhost:8003"),  # Server 4
]
```

### Issue 5: Thread Pool Not Shutting Down

**Symptoms**: Program hangs on exit

**Possible causes**:
1. Forgot to call `shutdown()`
2. Threads blocked on I/O
3. Daemon threads

**Solutions**:
```python
# Always use try-finally
collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)
try:
    rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)
finally:
    collector.shutdown()  # Guaranteed cleanup

# Or use timeout
collector.executor.executor.shutdown(wait=True, timeout=10)  # Wait max 10 seconds

# Force kill threads (last resort)
import sys
sys.exit(0)  # Force exit
```

---

## Summary

**Key Takeaways**:
1. ✅ Use **thread-based execution** for OpenEnv (I/O-bound)
2. ✅ Start with **4-8 workers** for optimal speedup
3. ✅ Use **`ParallelRolloutCollector`** for high-level API with stats
4. ✅ Always **call `shutdown()`** to clean up resources
5. ✅ Monitor **per-environment metrics** for curriculum learning
6. ✅ Profile to identify bottlenecks (network vs. agent vs. environment)

**Code References**:
- Thread-based executor: `Parallel_Environment_Execution.py:88-191`
- Async executor: `Parallel_Environment_Execution.py:194-296`
- High-level collector: `Parallel_Environment_Execution.py:299-398`
- Examples: `Parallel_Environment_Execution.py:409-679`

**Next Steps**:
- Try the examples in `Parallel_Environment_Execution.py`
- Integrate with your RL training loop
- Experiment with different `max_workers` values
- Monitor statistics for curriculum learning decisions

**Related Guides**:
- `RL_Integration_Gymnasium_Guide.md`: Gymnasium wrapper integration
- `RL_Integration_Patterns.md`: Framework-specific integration patterns
- (Next) `Curriculum_Learning_Framework_Guide.md`: Using parallel execution for curriculum learning
