# Environment State Transition and Isolation Guide

## Table of Contents
- [Overview](#overview)
- [Theoretical Foundation](#theoretical-foundation)
- [The Gymnasium-Style API for LLM Agents](#the-gymnasium-style-api-for-llm-agents)
- [State Transition Mechanics](#state-transition-mechanics)
- [State Isolation and Safety](#state-isolation-and-safety)
- [Critical File Paths and Functions](#critical-file-paths-and-functions)
- [Implementation Deep Dive](#implementation-deep-dive)
- [Best Practices](#best-practices)

---

## Overview

This guide explains how OpenEnv implements **state transitions** and **state isolation** in agent-environment interactions. The framework follows the **Gymnasium API pattern**, which is the gold standard for reinforcement learning environments.

**Key Concepts:**
- **State Transition**: How the environment changes from one state to another based on agent actions
- **State Isolation**: How each environment instance maintains independent state, preventing cross-contamination
- **Episode Management**: How episodes are tracked, initialized, and terminated

---

## Theoretical Foundation

### The Markov Decision Process (MDP)

OpenEnv environments implement a **Markov Decision Process**, defined by the tuple `(S, A, T, R, γ)`:

- **S**: State space (all possible environment configurations)
- **A**: Action space (all possible agent actions)
- **T**: Transition function `T(s' | s, a)` - probability of reaching state `s'` after taking action `a` in state `s`
- **R**: Reward function `R(s, a, s')` - scalar reward for transitioning from `s` to `s'` via action `a`
- **γ**: Discount factor (for multi-step planning)

### The Gymnasium API Pattern

The Gymnasium API (successor to OpenAI Gym) provides a standardized interface:

```python
# Initialize new episode
observation = env.reset()

# Interaction loop
done = False
while not done:
    action = agent.select_action(observation)
    observation, reward, done, info = env.step(action)
```

**Why this pattern?**
1. **Standardization**: Same interface across all environments
2. **Modularity**: Agents and environments can be developed independently
3. **Reproducibility**: Clear episode boundaries and state initialization
4. **Composability**: Easy to add wrappers, transforms, and monitors

---

## The Gymnasium-Style API for LLM Agents

### Why Gymnasium for LLM Agents?

Traditional Gymnasium environments were designed for RL agents operating in simulated physics (Atari, MuJoCo, etc.). OpenEnv adapts this pattern for **LLM-based agents**:

**Traditional RL Environment:**
```python
observation = env.reset()  # Returns: array([x, y, velocity, ...])
action = policy(observation)  # Returns: discrete action ID or continuous vector
next_obs, reward, done, info = env.step(action)
```

**LLM Agent Environment (OpenEnv):**
```python
observation = env.reset()  # Returns: Observation(message="Ready!", done=False, reward=0.0)
action = llm_agent.decide(observation)  # Returns: Action(command="...", args={...})
next_obs, reward, done, info = env.step(action)
```

**Key Differences:**
1. **Structured Data**: Actions and observations are dataclasses, not arrays
2. **Rich Semantics**: Observations can contain text, code, game boards, etc.
3. **HTTP Distribution**: Environments run in Docker containers, accessed via HTTP
4. **LLM-Native**: Designed for text-based reasoning, not numeric optimization

---

## State Transition Mechanics

### The `step()` Function: Core Transition Logic

The `step()` method implements the state transition function `T(s' | s, a)`:

```python
def step(self, action: Action) -> Observation:
    """
    Execute one timestep of the environment dynamics.

    Transition: (s_t, a_t) → (s_{t+1}, r_t, done_t)
    """
    # 1. VALIDATE ACTION (optional)
    if not self._is_valid_action(action):
        return self._create_error_observation("Invalid action")

    # 2. UPDATE STATE
    self._state.step_count += 1
    self._update_internal_state(action)

    # 3. COMPUTE REWARD
    reward = self._compute_reward(action)

    # 4. CHECK TERMINATION
    done = self._is_terminal_state()

    # 5. CREATE OBSERVATION
    observation = self._create_observation(reward, done)

    # 6. APPLY TRANSFORMS (optional)
    return self._apply_transform(observation)
```

### Step-by-Step Breakdown

#### 1. Action Validation
```python
# Example: Connect4 environment validates moves
def _is_valid_action(self, action: Connect4Action) -> bool:
    column = action.column
    # Check if column is in bounds
    if not (0 <= column < self.board_width):
        return False
    # Check if column is not full
    if self.board[0][column] != 0:
        return False
    return True
```

**Why validate?**
- Prevents invalid state transitions
- Provides clear error feedback to agents
- Maintains environment invariants

#### 2. State Update
```python
# Example: Echo environment state update
def step(self, action: EchoAction) -> EchoObservation:
    # Increment step counter (part of state)
    self._state.step_count += 1

    # Process action and update environment-specific state
    message = action.message
    length = len(message)
    # ... (Echo env has minimal state to update)
```

**State Components:**
- **Episode State** (`State` object):
  - `episode_id`: Unique identifier for this episode
  - `step_count`: Number of steps taken
- **Environment-Specific State**:
  - Game boards (Connect4)
  - Code execution context (Coding environment)
  - Conversation history (Chat environments)

#### 3. Reward Computation
```python
# Example: Echo environment reward
def step(self, action: EchoAction) -> EchoObservation:
    message = action.message
    length = len(message)
    reward = length * 0.1  # Simple: longer messages → higher reward
```

**Reward Design Principles:**
1. **Informative**: Should guide agent toward desired behavior
2. **Consistent**: Same action in same state → same reward
3. **Bounded**: Prevents numerical instability (optionally)
4. **Sparse vs Dense**: Trade-off between learning speed and credit assignment

#### 4. Termination Checking
```python
# Example: Connect4 termination conditions
def _is_terminal_state(self) -> bool:
    # Check for win
    if self._check_win(self.last_player):
        return True

    # Check for draw (board full)
    if self._is_board_full():
        return True

    # Check for max steps (optional timeout)
    if self._state.step_count >= self.max_steps:
        return True

    return False
```

**Termination Conditions:**
- **Goal Reached**: Task successfully completed
- **Failure**: Invalid move, constraint violation, etc.
- **Timeout**: Maximum steps exceeded
- **Natural End**: Game over, conversation ended, etc.

#### 5. Observation Construction
```python
# Example: Echo environment observation
def step(self, action: EchoAction) -> EchoObservation:
    # ... state update, reward computation ...

    return EchoObservation(
        echoed_message=message,
        message_length=length,
        done=False,
        reward=reward,
        metadata={"step": self._state.step_count},
    )
```

**Observation Structure:**
- **Inherited Fields** (from `Observation` base class):
  - `done: bool` - Is episode finished?
  - `reward: float | None` - Scalar reward signal
  - `metadata: dict` - Optional diagnostic info
- **Environment-Specific Fields**:
  - `echoed_message: str` (Echo env)
  - `board: List[List[int]]` (Connect4 env)
  - `stdout: str, stderr: str` (Coding env)

---

## State Isolation and Safety

### Per-Instance State Management

Each environment instance maintains **independent state**:

```python
# Creating two independent environment instances
env1 = EchoEnvironment()
env2 = EchoEnvironment()

# Each has its own state
env1.reset()  # Creates episode_id = "abc-123"
env2.reset()  # Creates episode_id = "xyz-789"

env1.step(EchoAction(message="Hello"))  # env1.step_count = 1
env2.step(EchoAction(message="Hi"))     # env2.step_count = 1

# States are independent
assert env1.state.episode_id != env2.state.episode_id
assert env1.state.step_count == env2.state.step_count == 1
```

**How is this achieved?**

```python
class EchoEnvironment(Environment):
    def __init__(self):
        super().__init__()
        # INSTANCE VARIABLE: Each environment has its own _state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> EchoObservation:
        # CREATE NEW STATE for new episode
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # ...
```

**Key Mechanisms:**
1. **Instance Variables**: State stored in `self._state`, not class variables
2. **Episode ID**: Unique UUID per episode prevents confusion
3. **No Shared State**: No global variables or class-level state

### HTTP Isolation in Distributed Setup

When environments run in Docker containers, isolation is **enforced by containerization**:

```
┌─────────────────┐       ┌─────────────────┐
│  Agent 1        │       │  Agent 2        │
│                 │       │                 │
│  client.reset() │       │  client.reset() │
│  client.step()  │       │  client.step()  │
└────────┬────────┘       └────────┬────────┘
         │ HTTP                    │ HTTP
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  Container 1    │       │  Container 2    │
│  Port: 8001     │       │  Port: 8002     │
│                 │       │                 │
│  EchoEnv()      │       │  EchoEnv()      │
│  episode_id: A  │       │  episode_id: B  │
└─────────────────┘       └─────────────────┘
```

**Container-Level Isolation:**
- Each agent gets its own container
- Containers run on different ports
- No shared memory or file system
- Network isolation (unless explicitly configured)

### Safety Guarantees

OpenEnv's state isolation provides several safety guarantees:

| Guarantee | Mechanism | Benefit |
|-----------|-----------|---------|
| **No Cross-Episode Contamination** | New `episode_id` on `reset()` | Episodes are independent |
| **No Cross-Instance Interference** | Instance variables (`self._state`) | Parallel agents don't conflict |
| **No Shared Mutable State** | Immutable dataclasses for Actions/Observations | Prevents accidental mutation |
| **Reproducibility** | Episode ID tracking | Easy to correlate logs and traces |
| **Container Isolation** | Docker containers | Strong OS-level separation |

---

## Critical File Paths and Functions

Here are the **5 critical files** that implement state transition and isolation:

### 1. Environment Base Class - State Transition Interface
**File**: `/home/user/OpenEnv/src/core/env_server/interfaces.py`

**Key Components:**
- **`Environment` class** (lines 88-118): Abstract base class defining the API
  - `reset() -> Observation` (line 99-101): Initialize episode state
  - `step(action: Action) -> Observation` (line 103-106): Execute state transition
  - `state -> State` (line 108-112): Query episode metadata

**Why Critical:**
- Enforces the Gymnasium API contract
- All environments must implement these methods
- Defines the interface for state transitions

### 2. Environment Types - State Data Structures
**File**: `/home/user/OpenEnv/src/core/env_server/types.py`

**Key Components:**
- **`State` dataclass** (lines 32-36): Episode metadata structure
  ```python
  @dataclass
  class State:
      episode_id: Optional[str] = None  # Unique episode identifier
      step_count: int = 0               # Number of steps taken
  ```
- **`Action` base class** (lines 16-19): Base for all actions
- **`Observation` base class** (lines 23-28): Base for all observations

**Why Critical:**
- Defines the state structure used by all environments
- `episode_id` enables episode tracking and isolation
- `step_count` tracks transition progress

### 3. Echo Environment - Concrete Implementation
**File**: `/home/user/OpenEnv/src/envs/echo_env/server/echo_environment.py`

**Key Components:**
- **`__init__` method** (lines 46-49): Instance-level state initialization
  ```python
  def __init__(self):
      self._state = State(episode_id=str(uuid4()), step_count=0)
      self._reset_count = 0
  ```
- **`reset` method** (lines 51-66): Episode initialization with new `episode_id`
  ```python
  def reset(self) -> EchoObservation:
      self._state = State(episode_id=str(uuid4()), step_count=0)
      # ...
  ```
- **`step` method** (lines 68-92): State transition implementation
  ```python
  def step(self, action: EchoAction) -> EchoObservation:
      self._state.step_count += 1  # Increment step counter
      # ... process action, compute reward ...
  ```
- **`state` property** (lines 94-102): State accessor
  ```python
  @property
  def state(self) -> State:
      return self._state
  ```

**Why Critical:**
- Shows how to implement state transitions correctly
- Demonstrates instance-level state management
- Example of episode_id generation and step_count tracking

### 4. HTTP Server - State Transition Routing
**File**: `/home/user/OpenEnv/src/core/env_server/http_server.py`

**Key Components:**
- **`/reset` endpoint** (lines 77-82): Routes HTTP POST to `env.reset()`
  ```python
  @app.post("/reset")
  async def reset(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
      observation = self.env.reset()
      return self._serialize_observation(observation)
  ```
- **`/step` endpoint** (lines 84-97): Routes HTTP POST to `env.step()`
  ```python
  @app.post("/step")
  async def step(request: Dict[str, Any]) -> Dict[str, Any]:
      action_data = request.get("action", {})
      action = self._deserialize_action(action_data)
      observation = self.env.step(action)
      return self._serialize_observation(observation)
  ```
- **`/state` endpoint** (lines 99-103): Routes HTTP GET to `env.state`
  ```python
  @app.get("/state")
  async def get_state() -> Dict[str, Any]:
      state = self.env.state
      return asdict(state)
  ```

**Why Critical:**
- Exposes environment state transitions via HTTP
- Each HTTP request triggers a state transition
- Serialization/deserialization preserves state integrity

### 5. HTTP Client - State Transition Invocation
**File**: `/home/user/OpenEnv/src/core/http_env_client.py`

**Key Components:**
- **`reset` method** (lines 142-154): Initiates episode via HTTP POST
  ```python
  def reset(self) -> StepResult[ObsT]:
      body: Dict[str, Any] = {}
      r = self._http.post(f"{self._base}/reset", json=body, ...)
      r.raise_for_status()
      return self._parse_result(r.json())
  ```
- **`step` method** (lines 156-171): Executes state transition via HTTP POST
  ```python
  def step(self, action: ActT) -> StepResult[ObsT]:
      body: Dict[str, Any] = {
          "action": self._step_payload(action),
          "timeout_s": int(self._timeout),
      }
      r = self._http.post(f"{self._base}/step", json=body, ...)
      r.raise_for_status()
      return self._parse_result(r.json())
  ```
- **`state` method** (lines 173-193): Queries state via HTTP GET
  ```python
  def state(self) -> Any:
      r = self._http.get(f"{self._base}/state", ...)
      r.raise_for_status()
      return self._parse_state(r.json())
  ```

**Why Critical:**
- Agent-facing API for triggering state transitions
- Abstracts HTTP complexity from agent code
- Ensures consistent state transition protocol

---

## Implementation Deep Dive

### Complete State Transition Flow

Here's what happens during a complete `reset() → step() → step()` sequence:

```
AGENT                    CLIENT                  HTTP                    SERVER                  ENVIRONMENT
  │                        │                       │                       │                         │
  │ client.reset()         │                       │                       │                         │
  ├────────────────────────►                       │                       │                         │
  │                        │ POST /reset           │                       │                         │
  │                        ├───────────────────────►                       │                         │
  │                        │ {}                    │ reset()               │                         │
  │                        │                       ├───────────────────────►                         │
  │                        │                       │                       │ env.reset()             │
  │                        │                       │                       ├─────────────────────────►
  │                        │                       │                       │                         │
  │                        │                       │                       │  1. episode_id = uuid4() │
  │                        │                       │                       │  2. step_count = 0       │
  │                        │                       │                       │  3. Create Observation   │
  │                        │                       │                       │                         │
  │                        │                       │                       │ return Observation      │
  │                        │                       │                       ◄─────────────────────────┤
  │                        │                       │ serialize to JSON     │                         │
  │                        │                       ◄───────────────────────┤                         │
  │                        │ {obs, reward, done}   │                       │                         │
  │                        ◄───────────────────────┤                       │                         │
  │ StepResult[Obs]        │                       │                       │                         │
  ◄────────────────────────┤                       │                       │                         │
  │                        │                       │                       │                         │
  │ client.step(action)    │                       │                       │                         │
  ├────────────────────────►                       │                       │                         │
  │                        │ POST /step            │                       │                         │
  │                        ├───────────────────────►                       │                         │
  │                        │ {action: {...}}       │ step(request)         │                         │
  │                        │                       ├───────────────────────►                         │
  │                        │                       │                       │ deserialize Action      │
  │                        │                       │                       │ env.step(action)        │
  │                        │                       │                       ├─────────────────────────►
  │                        │                       │                       │                         │
  │                        │                       │                       │  1. step_count += 1      │
  │                        │                       │                       │  2. Update state         │
  │                        │                       │                       │  3. Compute reward       │
  │                        │                       │                       │  4. Check done           │
  │                        │                       │                       │  5. Create Observation   │
  │                        │                       │                       │                         │
  │                        │                       │                       │ return Observation      │
  │                        │                       │                       ◄─────────────────────────┤
  │                        │                       │ serialize to JSON     │                         │
  │                        │                       ◄───────────────────────┤                         │
  │                        │ {obs, reward, done}   │                       │                         │
  │                        ◄───────────────────────┤                       │                         │
  │ StepResult[Obs]        │                       │                       │                         │
  ◄────────────────────────┤                       │                       │                         │
  │                        │                       │                       │                         │
```

### State Mutation Timeline

```
Time  │  Episode ID      │  Step Count  │  Environment State         │  Event
──────┼──────────────────┼──────────────┼────────────────────────────┼──────────────────
  0   │  None            │  None        │  Uninitialized             │  Environment created
  1   │  "abc-123-..."   │  0           │  Ready                     │  reset() called
  2   │  "abc-123-..."   │  1           │  After action 1            │  step(action1) called
  3   │  "abc-123-..."   │  2           │  After action 2            │  step(action2) called
  4   │  "abc-123-..."   │  3           │  After action 3            │  step(action3) called
  5   │  "xyz-789-..."   │  0           │  Ready (new episode)       │  reset() called
  6   │  "xyz-789-..."   │  1           │  After action 1            │  step(action1) called
```

**Key Observations:**
1. `episode_id` changes on `reset()`, creating a new episode
2. `step_count` resets to 0 on `reset()`
3. `step_count` increments on each `step()`
4. Environment-specific state evolves with each transition

---

## Best Practices

### ✅ DO: Implement Proper State Isolation

```python
class MyEnvironment(Environment):
    def __init__(self):
        super().__init__()
        # GOOD: Instance variable
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._game_board = [[0] * 8 for _ in range(8)]

    def reset(self) -> Observation:
        # GOOD: Create new episode_id
        self._state = State(episode_id=str(uuid4()), step_count=0)
        # GOOD: Reset environment-specific state
        self._game_board = [[0] * 8 for _ in range(8)]
        return MyObservation(...)
```

### ❌ DON'T: Use Class Variables for State

```python
class MyEnvironment(Environment):
    # BAD: Class variable shared across all instances!
    _game_board = [[0] * 8 for _ in range(8)]

    def reset(self) -> Observation:
        # BAD: Modifying shared class variable
        MyEnvironment._game_board = [[0] * 8 for _ in range(8)]
        return MyObservation(...)
```

**Why it's bad:**
- All environment instances share the same `_game_board`
- Actions from one agent affect other agents' environments
- No isolation between episodes or instances

### ✅ DO: Always Increment Step Count

```python
def step(self, action: Action) -> Observation:
    # GOOD: Increment step count at start of step
    self._state.step_count += 1

    # ... rest of step logic ...

    return observation
```

### ❌ DON'T: Forget to Track Steps

```python
def step(self, action: Action) -> Observation:
    # BAD: Forgot to increment step_count!
    # This breaks episode tracking and max_steps logic

    # ... step logic ...

    return observation
```

### ✅ DO: Generate New Episode IDs on Reset

```python
from uuid import uuid4

def reset(self) -> Observation:
    # GOOD: Generate unique episode ID
    self._state = State(episode_id=str(uuid4()), step_count=0)
    return observation
```

### ❌ DON'T: Reuse Episode IDs

```python
def reset(self) -> Observation:
    # BAD: Reusing the same episode ID
    self._state = State(episode_id="fixed-id", step_count=0)
    return observation
```

**Why it's bad:**
- Can't distinguish between episodes in logs
- Breaks episode tracking in databases
- Confuses monitoring and debugging tools

### ✅ DO: Return Immutable Observations

```python
from dataclasses import dataclass

@dataclass(kw_only=True, frozen=True)  # frozen=True makes it immutable
class MyObservation(Observation):
    message: str
    count: int
```

### ❌ DON'T: Allow Observation Mutation

```python
@dataclass(kw_only=True)
class MyObservation(Observation):
    message: str
    count: int

# Later, agent code could do:
obs = env.step(action).observation
obs.message = "Modified!"  # BAD: Mutating observation!
```

**Why it's bad:**
- Breaks reproducibility
- Can cause subtle bugs in multi-threaded scenarios
- Violates functional programming principles

---

## Summary

**State Transition** in OpenEnv:
- Implemented via `reset()` and `step()` methods
- Follows Gymnasium API pattern
- Each transition updates `step_count` and environment-specific state
- Returns structured `Observation` with reward and done flag

**State Isolation** in OpenEnv:
- Achieved via instance variables (`self._state`)
- Each environment instance has independent state
- Episode IDs (UUID) enable tracking and correlation
- Container-level isolation for distributed deployments

**Critical Implementation Points:**
1. Use instance variables for state (never class variables)
2. Generate new `episode_id` on each `reset()`
3. Increment `step_count` on every `step()`
4. Return immutable observations (use `frozen=True`)
5. Implement proper termination checking (`done` flag)

**File Map:**
- `interfaces.py`: State transition interface definition
- `types.py`: State data structure (`State`, `Action`, `Observation`)
- `echo_environment.py`: Reference implementation
- `http_server.py`: HTTP routing for state transitions
- `http_env_client.py`: Client-side state transition API

This design ensures that OpenEnv environments are **safe**, **isolated**, **reproducible**, and **composable** - essential properties for training and evaluating LLM agents.
