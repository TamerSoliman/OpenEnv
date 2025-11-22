# Curriculum Learning Framework Guide

## Table of Contents
1. [Overview](#overview)
2. [What is Curriculum Learning?](#what-is-curriculum-learning)
3. [Why Use Curriculum Learning?](#why-use-curriculum-learning)
4. [Core Concepts](#core-concepts)
5. [Progression Strategies](#progression-strategies)
6. [Integration with OpenEnv](#integration-with-openenv)
7. [Practical Examples](#practical-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Research Background](#research-background)

---

## Overview

Curriculum learning is a training paradigm inspired by human education: start with simple tasks and gradually increase difficulty. This guide shows how to implement curriculum learning for OpenEnv environments, leading to faster training and better final performance.

**Key Benefits:**
- **Faster Learning**: Agents master basics before tackling complexity
- **Better Exploration**: Avoid getting stuck in difficult tasks early
- **Higher Final Performance**: Progressive skill building leads to better policies
- **Sample Efficiency**: Less total training time to reach target performance

**What You'll Learn:**
1. When and why to use curriculum learning
2. Different progression strategies (threshold, adaptive, mixed)
3. How to integrate with parallel environment execution
4. Real-world implementation patterns

---

## What is Curriculum Learning?

### The Core Idea

**Without Curriculum** (Random initialization):
```
Agent → Hard Task → Struggle → Poor exploration → Slow learning
```

**With Curriculum** (Progressive difficulty):
```
Agent → Easy Task → Quick mastery → Medium Task → Build on basics → Hard Task → Success
```

### Example: Teaching a Robot to Walk

**Bad Approach**: Start on rough terrain
- Robot falls repeatedly
- Learns to fear falling, becomes overly cautious
- Never learns to walk properly

**Good Approach**: Progressive difficulty
1. **Stage 1**: Flat surface → learns basic balance
2. **Stage 2**: Gentle slopes → learns to adjust gait
3. **Stage 3**: Rough terrain → combines all skills

### Mathematical Formulation

Traditional RL objective:
```
max E[Σ γᵗ r(s_t, a_t)]  over hard task distribution
```

Curriculum RL objective:
```
Stage 1: max E[Σ γᵗ r(s_t, a_t)]  over easy task distribution
Stage 2: max E[Σ γᵗ r(s_t, a_t)]  over medium task distribution
Stage 3: max E[Σ γᵗ r(s_t, a_t)]  over hard task distribution
```

**Key insight**: Each stage initializes from previous stage's policy, transferring learned skills.

---

## Why Use Curriculum Learning?

### Problem 1: Sparse Rewards

**Scenario**: Environment only gives reward when task is fully completed.

**Without Curriculum**:
- Agent randomly explores
- Rarely completes hard task
- Gets almost no reward signal
- Learning stagnates

**With Curriculum**:
- Easy task: Higher completion rate → more reward signal
- Builds basic skills with frequent feedback
- Transfers skills to harder tasks

### Problem 2: Exploration Challenges

**Scenario**: Hard tasks require specific action sequences.

**Without Curriculum**:
- Random actions unlikely to discover correct sequence
- Agent gives up, learns "do nothing" policy

**With Curriculum**:
- Easy tasks have more forgiving action sequences
- Agent discovers useful patterns
- Generalizes to hard tasks

### Problem 3: Catastrophic Forgetting

**Scenario**: Agent trained on hard tasks forgets how to handle easy cases.

**With Curriculum**:
- Progressive training prevents forgetting
- Smooth skill transfer across difficulty levels

### Empirical Results from Literature

**"Curriculum Learning" (Bengio et al., 2009)**:
- 30-50% faster convergence on vision tasks
- Higher final accuracy

**"Automatic Curriculum Learning for Deep RL" (Portelas et al., 2020)**:
- 2-4x sample efficiency improvement
- Better generalization to unseen tasks

**"Teacher-Student Curriculum Learning" (Matiisen et al., 2017)**:
- 60% reduction in training time for Atari games
- Higher final scores

---

## Core Concepts

### Curriculum Stage

A **stage** represents one difficulty level in the curriculum.

```python
@dataclass
class CurriculumStage:
    name: str                      # "easy", "medium", "hard"
    env_config: Dict[str, Any]     # Environment configuration
    target_performance: float      # Threshold to advance
    min_episodes: int              # Minimum training episodes
    weight: float                  # For mixed strategies
```

**Example**:
```python
easy_stage = CurriculumStage(
    name="easy",
    env_config={"difficulty": 1, "max_steps": 10},
    target_performance=15.0,  # Advance when avg return > 15
    min_episodes=20           # But train for at least 20 episodes
)
```

### Progression Strategy

How to decide when to move to the next stage.

**Four strategies**:

1. **LINEAR**: Fixed schedule (e.g., 100 episodes per stage)
   - Simple, predictable
   - Ignores agent performance
   - May move too fast or too slow

2. **THRESHOLD**: Advance when performance exceeds target
   - Adapts to agent learning speed
   - Requires careful threshold tuning
   - Most common in practice

3. **ADAPTIVE**: Detect when performance plateaus
   - Fully automatic
   - No manual threshold tuning
   - May be too aggressive or conservative

4. **MIXED**: Train on multiple stages simultaneously
   - Maximum flexibility
   - Prevents catastrophic forgetting
   - More complex to manage

### Performance Window

Number of recent episodes used to evaluate performance.

**Small window (5-10 episodes)**:
- ✅ Responsive to improvements
- ❌ Sensitive to noise
- Use when: Episodes are long, environment is deterministic

**Large window (20-50 episodes)**:
- ✅ Robust to noise
- ❌ Slow to react
- Use when: High variance, stochastic environments

---

## Progression Strategies

### Strategy 1: Threshold-Based Progression

**When to use**: Clear performance thresholds exist for each difficulty level.

**How it works**:
1. Train on current stage
2. Track moving average of recent performance
3. When avg performance > threshold, advance to next stage
4. Repeat until final stage

**Example**:
```python
stages = [
    CurriculumStage(
        name="easy",
        env_config={"difficulty": 1},
        target_performance=15.0,  # 15.0 return to advance
        min_episodes=20
    ),
    CurriculumStage(
        name="hard",
        env_config={"difficulty": 3},
        target_performance=30.0,
        min_episodes=50
    ),
]

scheduler = CurriculumScheduler(
    stages=stages,
    strategy=ProgressionStrategy.THRESHOLD,
    performance_window=10  # Use last 10 episodes
)

for episode in range(1000):
    rollouts = collect_rollouts(scheduler.get_current_stage())
    mean_return = compute_mean_return(rollouts)

    advanced = scheduler.update(mean_return)
    if advanced:
        print(f"Advanced to {scheduler.get_current_stage().name}!")
```

**Visualization**:
```
Return
  ^
40|                                    ╱╱╱╱╱  (hard stage)
  |                          ╱╱╱╱╱╱╱╱╱
30|                    ╱╱╱╱╱╱  ← threshold
  |              ╱╱╱╱╱
20|        ╱╱╱╱╱╱  (medium stage)
  |  ╱╱╱╱╱  ← threshold
10|╱╱╱╱╱  (easy stage)
  |
  +────────────────────────────────────────> Episodes
     0   50  100 150 200 250 300
```

**Code reference**: `Curriculum_Learning_Framework.py:359-420`

### Strategy 2: Adaptive Progression (Plateau Detection)

**When to use**: Don't know appropriate thresholds, want fully automatic progression.

**How it works**:
1. Track performance over sliding window
2. Compare first half vs second half of window
3. If improvement < threshold (e.g., 5%), declare plateau
4. Advance to next stage

**Example**:
```python
scheduler = CurriculumScheduler(
    stages=stages,
    strategy=ProgressionStrategy.ADAPTIVE,
    performance_window=20  # Larger window for robust plateau detection
)

# No target_performance needed - automatic detection!
```

**Plateau Detection Algorithm**:
```python
def detect_plateau(recent_performance):
    window = recent_performance[-20:]  # Last 20 episodes
    first_half = mean(window[:10])
    second_half = mean(window[10:])

    improvement = (second_half - first_half) / first_half
    return improvement < 0.05  # Less than 5% improvement
```

**Visualization**:
```
Return
  ^
  |     ╱╱╱╱╱╱╱╱╱╱╱╱╱╱──────────  ← plateau detected!
  |   ╱╱
  | ╱╱
  |╱
  +──────────────────────────────────> Episodes
  0    10   20   30   40   50
```

**Pros**:
- No manual threshold tuning
- Adapts to agent learning rate

**Cons**:
- May advance too early if plateau is temporary
- Sensitive to window size

**Code reference**: `Curriculum_Learning_Framework.py:423-482`

### Strategy 3: Mixed Multi-Stage Training

**When to use**: Want to prevent forgetting, maintain skills across all difficulty levels.

**How it works**:
1. Train on all stages simultaneously
2. Weight each stage (e.g., 50% easy, 30% medium, 20% hard)
3. Sample environments according to weights

**Example**:
```python
stages = [
    CurriculumStage(name="easy", env_config={...}, weight=0.5),
    CurriculumStage(name="medium", env_config={...}, weight=0.3),
    CurriculumStage(name="hard", env_config={...}, weight=0.2),
]

scheduler = CurriculumScheduler(
    stages=stages,
    strategy=ProgressionStrategy.MIXED
)

# Get environment distribution
distribution = scheduler.get_stage_distribution()
# {"easy": 0.5, "medium": 0.3, "hard": 0.2}

# Create 10 parallel environments
easy_envs = create_envs("easy", count=5)    # 50%
medium_envs = create_envs("medium", count=3)  # 30%
hard_envs = create_envs("hard", count=2)    # 20%
```

**Visualization**:
```
Parallel Environments (10 total):

┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  Easy (5)
│ Easy│ │ Easy│ │ Easy│ │ Easy│ │ Easy│
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘

┌──────┐ ┌──────┐ ┌──────┐  Medium (3)
│Medium│ │Medium│ │Medium│
└──────┘ └──────┘ └──────┘

┌─────┐ ┌─────┐  Hard (2)
│ Hard│ │ Hard│
└─────┘ └─────┘
```

**Pros**:
- Prevents catastrophic forgetting
- Maintains skills across all levels
- More robust training

**Cons**:
- More complex to manage
- Requires more compute (more environments)

**Code reference**: `Curriculum_Learning_Framework.py:485-533`

---

## Integration with OpenEnv

### Pattern 1: Manual Stage Switching

**Simple approach**: Manually switch environment configurations.

```python
from core.http_env_client import HttpEnvClient
from Claude_tutorials.Curriculum_Learning_Framework import CurriculumScheduler

# Define stages
stages = [
    CurriculumStage(name="easy", env_config={"port": 8000}, target_performance=15.0, min_episodes=50),
    CurriculumStage(name="hard", env_config={"port": 8001}, target_performance=30.0, min_episodes=100),
]

scheduler = CurriculumScheduler(stages, strategy=ProgressionStrategy.THRESHOLD)

# Create environment for current stage
current_stage = scheduler.get_current_stage()
env_client = HttpEnvClient(f"http://localhost:{current_stage.env_config['port']}")

# Training loop
for episode in range(500):
    # Collect rollout
    obs = env_client.reset()
    episode_return = 0

    for step in range(100):
        action = agent.get_action(obs)
        result = env_client.step(action)
        obs = result.observation
        episode_return += result.reward
        if result.done:
            break

    # Update curriculum
    advanced = scheduler.update(episode_return)

    # Switch environment if stage changed
    if advanced:
        current_stage = scheduler.get_current_stage()
        env_client = HttpEnvClient(f"http://localhost:{current_stage.env_config['port']}")
```

### Pattern 2: Integration with Parallel Execution

**Advanced approach**: Combine curriculum + parallel rollout collection.

```python
from Claude_tutorials.Parallel_Environment_Execution import ParallelRolloutCollector
from Claude_tutorials.Curriculum_Learning_Framework import CurriculumParallelTrainer

# Factory function to create environments from config
def create_env_client(config):
    return HttpEnvClient(f"http://localhost:{config['port']}")

# Create curriculum trainer
stages = [
    CurriculumStage(name="easy", env_config={"port": 8000}, target_performance=15.0, min_episodes=20),
    CurriculumStage(name="medium", env_config={"port": 8001}, target_performance=25.0, min_episodes=30),
    CurriculumStage(name="hard", env_config={"port": 8002}, target_performance=35.0, min_episodes=50),
]

scheduler = CurriculumScheduler(stages, strategy=ProgressionStrategy.THRESHOLD)

trainer = CurriculumParallelTrainer(
    curriculum_scheduler=scheduler,
    env_factory=create_env_client,
    num_parallel_envs=8
)

# Create parallel collector
collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)

# Training loop
for iteration in range(100):
    # Collect rollouts from current stage (8 parallel envs)
    rollouts, curriculum_stats, stage_changed = trainer.train_step(agent_fn, collector)

    # Train agent on collected data
    train_agent(rollouts)

    # Log progress
    print(f"Iteration {iteration}: Stage={curriculum_stats['current_stage']}, "
          f"Performance={curriculum_stats['recent_performance']:.2f}")

    if stage_changed:
        print(f"  → Advanced to {curriculum_stats['current_stage']}!")

collector.shutdown()
```

**Code reference**: `Curriculum_Learning_Framework.py:316-356`

### Pattern 3: Environment-Specific Curriculum

Different OpenEnv environments have different natural curriculum progressions:

**BrowserGym**: Task complexity
```python
stages = [
    CurriculumStage(name="click_only", env_config={"allowed_actions": ["click"]}, ...),
    CurriculumStage(name="click_type", env_config={"allowed_actions": ["click", "type"]}, ...),
    CurriculumStage(name="full", env_config={"allowed_actions": "all"}, ...),
]
```

**Coding Environment**: Problem difficulty
```python
stages = [
    CurriculumStage(name="easy", env_config={"problem_set": "leetcode_easy"}, ...),
    CurriculumStage(name="medium", env_config={"problem_set": "leetcode_medium"}, ...),
    CurriculumStage(name="hard", env_config={"problem_set": "leetcode_hard"}, ...),
]
```

**Git Environment**: Command complexity
```python
stages = [
    CurriculumStage(name="basic", env_config={"allowed_commands": ["add", "commit", "status"]}, ...),
    CurriculumStage(name="intermediate", env_config={"allowed_commands": ["add", "commit", "branch", "merge"]}, ...),
    CurriculumStage(name="advanced", env_config={"allowed_commands": "all"}, ...),
]
```

---

## Practical Examples

### Example 1: Echo Environment Curriculum

Train agent to produce longer, more complex echo responses.

```python
from envs.echo_env import EchoAction

# Define curriculum: short → medium → long echoes
stages = [
    CurriculumStage(
        name="short_echo",
        env_config={"max_message_length": 10},
        target_performance=5.0,
        min_episodes=30
    ),
    CurriculumStage(
        name="medium_echo",
        env_config={"max_message_length": 50},
        target_performance=10.0,
        min_episodes=50
    ),
    CurriculumStage(
        name="long_echo",
        env_config={"max_message_length": 200},
        target_performance=15.0,
        min_episodes=100
    ),
]

scheduler = CurriculumScheduler(stages, strategy=ProgressionStrategy.THRESHOLD)

# Training loop
for episode in range(500):
    env_client = create_echo_env(scheduler.get_current_stage().env_config)

    obs = env_client.reset()
    episode_return = 0

    for step in range(20):
        # Agent learns to echo with increasing complexity
        message = agent.generate_message(obs, max_len=scheduler.get_current_stage().env_config["max_message_length"])
        action = EchoAction(message=message)

        result = env_client.step(action)
        episode_return += result.reward
        if result.done:
            break

    advanced = scheduler.update(episode_return)
    if advanced:
        print(f"Mastered {stages[scheduler.state.current_stage_idx - 1].name}!")
```

### Example 2: Connect4 Curriculum

Train agent against opponents of increasing skill.

```python
stages = [
    CurriculumStage(
        name="random_opponent",
        env_config={"opponent": "random"},
        target_performance=0.7,  # 70% win rate
        min_episodes=100
    ),
    CurriculumStage(
        name="heuristic_opponent",
        env_config={"opponent": "heuristic"},
        target_performance=0.6,  # 60% win rate
        min_episodes=200
    ),
    CurriculumStage(
        name="minimax_opponent",
        env_config={"opponent": "minimax_depth3"},
        target_performance=0.5,  # 50% win rate (fair game)
        min_episodes=500
    ),
]

scheduler = CurriculumScheduler(stages, strategy=ProgressionStrategy.THRESHOLD, performance_window=20)

# Training loop
for episode in range(1000):
    current_stage = scheduler.get_current_stage()
    env_client = create_connect4_env(opponent=current_stage.env_config["opponent"])

    # Play episode
    win = play_connect4_episode(env_client, agent)
    episode_return = 1.0 if win else 0.0

    advanced = scheduler.update(episode_return)
    if advanced and scheduler.state.current_stage_idx < len(stages):
        print(f"Agent ready for {scheduler.get_current_stage().name}!")
```

### Example 3: BrowserGym Curriculum

Progressive web navigation skills.

```python
stages = [
    CurriculumStage(
        name="miniwob_click",
        env_config={"task_suite": "miniwob", "task": "click-button"},
        target_performance=0.8,
        min_episodes=50
    ),
    CurriculumStage(
        name="miniwob_form",
        env_config={"task_suite": "miniwob", "task": "enter-text"},
        target_performance=0.7,
        min_episodes=100
    ),
    CurriculumStage(
        name="webarena",
        env_config={"task_suite": "webarena", "task": "shopping"},
        target_performance=0.5,
        min_episodes=200
    ),
]

# Use adaptive progression (no manual thresholds needed)
scheduler = CurriculumScheduler(stages, strategy=ProgressionStrategy.ADAPTIVE)

for episode in range(500):
    current_stage = scheduler.get_current_stage()
    env_client = create_browsergym_env(current_stage.env_config)

    success = run_browser_episode(env_client, agent)
    episode_return = 1.0 if success else 0.0

    advanced = scheduler.update(episode_return)
```

---

## Best Practices

### 1. Stage Design

**Good stage progression**:
- Clear difficulty gradient
- Each stage teachable (not impossibly hard)
- Skills transfer between stages

**Bad stage progression**:
- Jumps too quickly (easy → impossible)
- Stages teach conflicting skills
- No overlap in required skills

**Rule of thumb**: Agent should achieve ~60-80% performance on stage before advancing.

### 2. Threshold Tuning

**Too low**: Agent advances before mastering stage
```python
# BAD: Threshold too easy
CurriculumStage(name="hard", target_performance=1.0, ...)  # Advances on first success
```

**Too high**: Agent wastes time on mastered stage
```python
# BAD: Threshold too hard
CurriculumStage(name="easy", target_performance=100.0, ...)  # Unreachable perfection
```

**Just right**: Advance when clear mastery demonstrated
```python
# GOOD: Reasonable threshold
CurriculumStage(name="medium", target_performance=15.0, min_episodes=30, ...)
```

**Recommendation**: Start conservative (higher thresholds), tune down if too slow.

### 3. Minimum Episodes

**Always set `min_episodes`** to prevent premature advancement:

```python
CurriculumStage(
    name="easy",
    target_performance=10.0,
    min_episodes=20  # ← Prevents lucky early success from triggering advancement
)
```

**Why**: Agent might get lucky and exceed threshold early, but not truly master the skill.

### 4. Performance Window Size

**Small window (5-10)**: Responsive but noisy
- Use for: Deterministic environments, long episodes

**Medium window (10-20)**: Balanced (recommended)
- Use for: Most scenarios

**Large window (20-50)**: Stable but slow
- Use for: High-variance, stochastic environments

### 5. Logging and Monitoring

**Track per-stage metrics**:
```python
stats = scheduler.get_statistics()

# Log to Weights & Biases
wandb.log({
    "curriculum/stage": stats["current_stage"],
    "curriculum/episodes_in_stage": stats["episodes_in_stage"],
    "curriculum/recent_performance": stats["recent_performance"],
    # Per-stage statistics
    "performance/easy_mean": stats.get("easy_mean", 0),
    "performance/medium_mean": stats.get("medium_mean", 0),
    "performance/hard_mean": stats.get("hard_mean", 0),
})
```

**Visualize progression**:
```python
import matplotlib.pyplot as plt

# Plot performance across stages
plt.plot(scheduler.state.performance_history)
plt.axvline(x=stage_transitions[0], color='r', linestyle='--', label='Easy→Medium')
plt.axvline(x=stage_transitions[1], color='r', linestyle='--', label='Medium→Hard')
plt.xlabel("Episode")
plt.ylabel("Return")
plt.legend()
plt.show()
```

### 6. Saving and Loading Curriculum State

**Save curriculum progress**:
```python
import pickle

# Save curriculum state
with open("curriculum_state.pkl", "wb") as f:
    pickle.dump(scheduler.state, f)

# Load and resume
with open("curriculum_state.pkl", "rb") as f:
    scheduler.state = pickle.load(f)
```

**Why**: Resume training from checkpoint without restarting curriculum.

---

## Troubleshooting

### Issue 1: Agent Stuck on Early Stage

**Symptoms**: Never exceeds threshold for first stage

**Possible causes**:
1. Threshold too high
2. Agent architecture insufficient
3. Learning rate too low

**Solutions**:
```python
# Lower threshold
stage.target_performance = 10.0  # Down from 20.0

# Check agent is learning
print(f"Recent performance: {scheduler._get_recent_performance()}")
print(f"Threshold: {scheduler.get_current_stage().target_performance}")

# Adjust learning rate or architecture
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)  # Increase from 1e-4
```

### Issue 2: Premature Advancement

**Symptoms**: Advances to next stage but performance drops

**Possible causes**:
1. Lucky spike in performance
2. `min_episodes` too low
3. Performance window too small

**Solutions**:
```python
# Increase min_episodes
stage.min_episodes = 50  # Up from 20

# Increase performance window
scheduler = CurriculumScheduler(stages, performance_window=20)  # Up from 10

# Add "stability check" - must sustain performance
def stable_performance(recent_perf):
    return all(p > threshold for p in recent_perf[-5:])  # Last 5 all above threshold
```

### Issue 3: Curriculum Too Slow

**Symptoms**: Spends too long on easy stages

**Possible causes**:
1. Thresholds too high
2. Adaptive strategy too conservative

**Solutions**:
```python
# Lower thresholds
for stage in stages:
    stage.target_performance *= 0.8  # Reduce by 20%

# Switch to LINEAR strategy for fixed schedule
scheduler = CurriculumScheduler(stages, strategy=ProgressionStrategy.LINEAR)

# Or use ADAPTIVE with more aggressive plateau detection
def detect_plateau_aggressive(recent_perf):
    improvement = (recent_perf[-5:].mean() - recent_perf[-10:-5].mean()) / recent_perf[-10:-5].mean()
    return improvement < 0.10  # 10% threshold instead of 5%
```

### Issue 4: Forgetting Previous Stages

**Symptoms**: After advancing, performance on easy stages drops

**Possible causes**:
1. Catastrophic forgetting
2. No mixed training

**Solutions**:
```python
# Switch to MIXED strategy
scheduler = CurriculumScheduler(stages, strategy=ProgressionStrategy.MIXED)

# Or periodically revisit earlier stages
if episode % 100 == 0:
    # Test on all previous stages
    for past_stage_idx in range(scheduler.state.current_stage_idx):
        stage = stages[past_stage_idx]
        test_performance(stage)
```

---

## Research Background

### Seminal Papers

**1. "Curriculum Learning" (Bengio et al., 2009)**
- First formalization of curriculum learning
- Shows 30-50% faster convergence on vision tasks
- Key idea: Order training examples from easy to hard

**2. "Automatic Curriculum Learning for Deep RL" (Portelas et al., 2020)**
- Automatic curriculum generation
- 2-4x sample efficiency improvement
- Adaptive difficulty based on agent progress

**3. "Teacher-Student Curriculum Learning" (Matiisen et al., 2017)**
- Teacher network selects tasks for student
- 60% reduction in training time for Atari
- Prevents forgetting through strategic task selection

**4. "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey" (Narvekar et al., 2020)**
- Comprehensive survey of curriculum RL methods
- Taxonomy of curriculum generation approaches
- Best practices and open problems

### Key Findings from Research

**1. Starting difficulty matters**:
- Too easy: Slow initial learning
- Too hard: Agent never learns
- Just right: Fast skill acquisition

**2. Progression strategy impacts final performance**:
- Fixed schedule: Simple but suboptimal
- Performance-based: Better final performance
- Adaptive: Best sample efficiency

**3. Curriculum transfer**:
- Skills learned in early stages transfer to later stages
- Most effective when stages share state/action structure

**4. When curriculum helps most**:
- Sparse rewards
- Long horizons
- Complex action spaces
- Hierarchical tasks

### Recommended Reading

1. **Bengio, Y., et al. (2009).** "Curriculum Learning." ICML.
2. **Portelas, R., et al. (2020).** "Automatic Curriculum Learning for Deep RL: A Short Survey." IJCAI.
3. **Narvekar, S., et al. (2020).** "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey." JMLR.

---

## Summary

**Key Takeaways**:
1. ✅ Use curriculum learning when tasks have clear difficulty levels
2. ✅ Start with THRESHOLD strategy, tune target_performance empirically
3. ✅ Always set min_episodes to prevent premature advancement
4. ✅ Use performance_window=10-20 for most scenarios
5. ✅ Log per-stage metrics to monitor progression
6. ✅ Combine with parallel execution for efficient data collection

**Decision Tree**:
```
Do you have clear difficulty levels?
├─ Yes: Use curriculum learning
│   ├─ Know good thresholds? → Use THRESHOLD strategy
│   ├─ Don't know thresholds? → Use ADAPTIVE strategy
│   └─ Worried about forgetting? → Use MIXED strategy
└─ No: Standard training (no curriculum)
```

**Code References**:
- Threshold curriculum: `Curriculum_Learning_Framework.py:359-420`
- Adaptive curriculum: `Curriculum_Learning_Framework.py:423-482`
- Mixed curriculum: `Curriculum_Learning_Framework.py:485-533`
- Parallel integration: `Curriculum_Learning_Framework.py:316-356`

**Next Steps**:
1. Run examples: `python Claude_tutorials/Curriculum_Learning_Framework.py`
2. Design stages for your environment
3. Start with threshold strategy, tune thresholds
4. Monitor per-stage performance
5. Iterate on progression strategy

**Related Guides**:
- `Parallel_Environment_Execution_Guide.md`: Efficient data collection
- `RL_Integration_Gymnasium_Guide.md`: Gymnasium wrapper for RL training
- (Next) `Transfer_Learning_Guide.md`: Transfer skills across environments
