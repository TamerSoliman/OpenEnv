# Agent-Environment Data Flow Guide

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Data Flow: Single Turn Simulation](#data-flow-single-turn-simulation)
- [Action Object Structure](#action-object-structure)
- [Observation Object Structure](#observation-object-structure)
- [StepResult Tuple Structure](#stepresult-tuple-structure)
- [Serialization and Deserialization](#serialization-and-deserialization)
- [Complete Examples](#complete-examples)
- [Advanced Topics](#advanced-topics)

---

## Overview

This guide describes the **complete data flow** of a single turn in OpenEnv's agent-environment simulation. You'll learn:
- The structure of **Action** objects sent by agents
- The structure of **Observation/Reward/Done** tuples returned by environments
- How data flows through the HTTP layer
- How serialization/deserialization works

---

## System Architecture

OpenEnv uses a **client-server architecture** with HTTP as the communication protocol:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐                              ┌──────────────┐     │
│  │              │                              │              │     │
│  │    AGENT     │                              │  ENVIRONMENT │     │
│  │   (Python)   │                              │   (Docker)   │     │
│  │              │                              │              │     │
│  └──────┬───────┘                              └──────▲───────┘     │
│         │                                             │             │
│         │  Action (Python object)                    │             │
│         │  EchoAction(message="Hello")               │             │
│         │                                             │             │
│         ▼                                             │             │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │           HTTPEnvClient (Agent Side)                     │       │
│  │  - Serializes Action → JSON                             │       │
│  │  - Sends HTTP POST                                       │       │
│  │  - Receives HTTP Response                                │       │
│  │  - Deserializes JSON → StepResult[Observation]           │       │
│  └──────┬───────────────────────────────────────────▲───────┘       │
│         │                                            │               │
│         │  HTTP POST /step                          │               │
│         │  {                                        │               │
│         │    "action": {"message": "Hello"},        │               │
│         │    "timeout_s": 15                        │               │
│         │  }                                        │               │
│         │                                            │               │
│         │                                            │  HTTP 200     │
│         │                                            │  {            │
│         │                                            │    "observation": {...},
│         │                                            │    "reward": 0.5,
│         │                                            │    "done": false
│         │                                            │  }            │
│         │                                            │               │
│         ▼                                            │               │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                   HTTP NETWORK                           │       │
│  │              (localhost or network)                      │       │
│  └──────┬───────────────────────────────────────────▲───────┘       │
│         │                                            │               │
│         │  JSON Payload                             │               │
│         │                                            │  JSON Response│
│         │                                            │               │
│         ▼                                            │               │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │           HTTPEnvServer (Environment Side)               │       │
│  │  - Receives HTTP POST                                    │       │
│  │  - Deserializes JSON → Action object                     │       │
│  │  - Calls env.step(action)                                │       │
│  │  - Serializes Observation → JSON                         │       │
│  └──────┬───────────────────────────────────────────▲───────┘       │
│         │                                            │               │
│         │  Action (deserialized)                    │               │
│         │  EchoAction(message="Hello")              │               │
│         │                                            │               │
│         │                                            │  Observation  │
│         │                                            │  EchoObservation(...)
│         │                                            │               │
│         ▼                                            │               │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │              Environment (Business Logic)                │       │
│  │  - Processes action                                      │       │
│  │  - Updates state                                         │       │
│  │  - Computes reward                                       │       │
│  │  - Creates observation                                   │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Layers:**
1. **Agent Layer**: Business logic for action selection
2. **Client Layer**: Serialization and HTTP communication
3. **Network Layer**: HTTP protocol (localhost or distributed)
4. **Server Layer**: HTTP routing and deserialization
5. **Environment Layer**: State transition and reward computation

---

## Data Flow: Single Turn Simulation

Let's trace a complete turn from `reset()` to `step()`:

### Part 1: Reset - Episode Initialization

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RESET FLOW (Episode Initialization)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  AGENT CODE                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  client = EchoEnv.from_docker_image("echo-env:latest")       │   │
│  │  result = client.reset()                                     │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  CLIENT: HTTPEnvClient.reset()                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  body = {}  # Empty payload for reset                        │   │
│  │  response = POST http://localhost:8000/reset                 │   │
│  │              Headers: {"Content-Type": "application/json"}   │   │
│  │              Body: {}                                        │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  HTTP REQUEST                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  POST /reset HTTP/1.1                                        │   │
│  │  Host: localhost:8000                                        │   │
│  │  Content-Type: application/json                              │   │
│  │  Content-Length: 2                                           │   │
│  │                                                              │   │
│  │  {}                                                          │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  SERVER: HTTPEnvServer /reset endpoint                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  @app.post("/reset")                                         │   │
│  │  async def reset(request: Dict = Body(default={})):          │   │
│  │      observation = self.env.reset()  # Call environment     │   │
│  │      return self._serialize_observation(observation)         │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  ENVIRONMENT: EchoEnvironment.reset()                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  def reset(self) -> EchoObservation:                         │   │
│  │      # 1. Create new episode state                           │   │
│  │      self._state = State(                                    │   │
│  │          episode_id=str(uuid4()),  # "a1b2c3d4-..."         │   │
│  │          step_count=0                                        │   │
│  │      )                                                       │   │
│  │      self._reset_count += 1                                  │   │
│  │                                                              │   │
│  │      # 2. Create initial observation                         │   │
│  │      return EchoObservation(                                 │   │
│  │          echoed_message="Echo environment ready!",           │   │
│  │          message_length=0,                                   │   │
│  │          done=False,                                         │   │
│  │          reward=0.0,                                         │   │
│  │      )                                                       │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  SERVER: Serialize observation                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  def _serialize_observation(self, obs):                      │   │
│  │      obs_dict = asdict(obs)                                  │   │
│  │      # obs_dict = {                                          │   │
│  │      #   "echoed_message": "Echo environment ready!",        │   │
│  │      #   "message_length": 0,                                │   │
│  │      #   "done": False,                                      │   │
│  │      #   "reward": 0.0,                                      │   │
│  │      #   "metadata": {}                                      │   │
│  │      # }                                                     │   │
│  │                                                              │   │
│  │      reward = obs_dict.pop("reward", None)                   │   │
│  │      done = obs_dict.pop("done", False)                      │   │
│  │      obs_dict.pop("metadata", None)                          │   │
│  │                                                              │   │
│  │      return {                                                │   │
│  │          "observation": obs_dict,  # {"echoed_message": ..., "message_length": ...}
│  │          "reward": reward,         # 0.0                     │   │
│  │          "done": done,             # False                   │   │
│  │      }                                                       │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  HTTP RESPONSE                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  HTTP/1.1 200 OK                                             │   │
│  │  Content-Type: application/json                              │   │
│  │                                                              │   │
│  │  {                                                           │   │
│  │    "observation": {                                          │   │
│  │      "echoed_message": "Echo environment ready!",            │   │
│  │      "message_length": 0                                     │   │
│  │    },                                                        │   │
│  │    "reward": 0.0,                                            │   │
│  │    "done": false                                             │   │
│  │  }                                                           │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  CLIENT: Parse response                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  def _parse_result(self, payload: dict):                     │   │
│  │      obs_data = payload["observation"]                       │   │
│  │      # obs_data = {                                          │   │
│  │      #   "echoed_message": "Echo environment ready!",        │   │
│  │      #   "message_length": 0                                 │   │
│  │      # }                                                     │   │
│  │                                                              │   │
│  │      observation = EchoObservation(                          │   │
│  │          **obs_data,                                         │   │
│  │          done=payload["done"],                               │   │
│  │          reward=payload["reward"]                            │   │
│  │      )                                                       │   │
│  │                                                              │   │
│  │      return StepResult(                                      │   │
│  │          observation=observation,                            │   │
│  │          reward=payload["reward"],                           │   │
│  │          done=payload["done"]                                │   │
│  │      )                                                       │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  AGENT CODE: Receive result                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  result = StepResult[EchoObservation](                       │   │
│  │      observation=EchoObservation(                            │   │
│  │          echoed_message="Echo environment ready!",           │   │
│  │          message_length=0,                                   │   │
│  │          done=False,                                         │   │
│  │          reward=0.0                                          │   │
│  │      ),                                                      │   │
│  │      reward=0.0,                                             │   │
│  │      done=False                                              │   │
│  │  )                                                           │   │
│  │                                                              │   │
│  │  # Agent can now access:                                     │   │
│  │  print(result.observation.echoed_message)  # "Echo environment ready!"
│  │  print(result.reward)                      # 0.0            │   │
│  │  print(result.done)                        # False          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Part 2: Step - State Transition

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STEP FLOW (State Transition)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  AGENT CODE                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  action = EchoAction(message="Hello, World!")                │   │
│  │  result = client.step(action)                                │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  CLIENT: HTTPEnvClient.step(action)                                 │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  def step(self, action: EchoAction):                         │   │
│  │      # Serialize action                                      │   │
│  │      action_payload = self._step_payload(action)             │   │
│  │      # action_payload = {"message": "Hello, World!"}         │   │
│  │                                                              │   │
│  │      body = {                                                │   │
│  │          "action": action_payload,                           │   │
│  │          "timeout_s": 15                                     │   │
│  │      }                                                       │   │
│  │                                                              │   │
│  │      response = POST http://localhost:8000/step              │   │
│  │                  json=body                                   │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  HTTP REQUEST                                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  POST /step HTTP/1.1                                         │   │
│  │  Host: localhost:8000                                        │   │
│  │  Content-Type: application/json                              │   │
│  │                                                              │   │
│  │  {                                                           │   │
│  │    "action": {                                               │   │
│  │      "message": "Hello, World!"                              │   │
│  │    },                                                        │   │
│  │    "timeout_s": 15                                           │   │
│  │  }                                                           │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  SERVER: HTTPEnvServer /step endpoint                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  @app.post("/step")                                          │   │
│  │  async def step(request: Dict):                              │   │
│  │      # Extract action data                                   │   │
│  │      action_data = request.get("action", {})                 │   │
│  │      # action_data = {"message": "Hello, World!"}            │   │
│  │                                                              │   │
│  │      # Deserialize action                                    │   │
│  │      action = self._deserialize_action(action_data)          │   │
│  │      # action = EchoAction(message="Hello, World!")          │   │
│  │                                                              │   │
│  │      # Execute step                                          │   │
│  │      observation = self.env.step(action)                     │   │
│  │                                                              │   │
│  │      # Serialize and return                                  │   │
│  │      return self._serialize_observation(observation)         │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  ENVIRONMENT: EchoEnvironment.step(action)                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  def step(self, action: EchoAction) -> EchoObservation:      │   │
│  │      # 1. Increment step count                               │   │
│  │      self._state.step_count += 1  # 0 → 1                    │   │
│  │                                                              │   │
│  │      # 2. Process action                                     │   │
│  │      message = action.message  # "Hello, World!"             │   │
│  │      length = len(message)     # 13                          │   │
│  │                                                              │   │
│  │      # 3. Compute reward                                     │   │
│  │      reward = length * 0.1     # 13 * 0.1 = 1.3              │   │
│  │                                                              │   │
│  │      # 4. Create observation                                 │   │
│  │      return EchoObservation(                                 │   │
│  │          echoed_message=message,       # "Hello, World!"     │   │
│  │          message_length=length,        # 13                  │   │
│  │          done=False,                                         │   │
│  │          reward=reward,                # 1.3                 │   │
│  │          metadata={"step": self._state.step_count}  # {"step": 1}
│  │      )                                                       │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  SERVER: Serialize observation                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  obs_dict = asdict(observation)                              │   │
│  │  # obs_dict = {                                              │   │
│  │  #   "echoed_message": "Hello, World!",                      │   │
│  │  #   "message_length": 13,                                   │   │
│  │  #   "done": False,                                          │   │
│  │  #   "reward": 1.3,                                          │   │
│  │  #   "metadata": {"step": 1}                                 │   │
│  │  # }                                                         │   │
│  │                                                              │   │
│  │  reward = obs_dict.pop("reward", None)                       │   │
│  │  done = obs_dict.pop("done", False)                          │   │
│  │  obs_dict.pop("metadata", None)                              │   │
│  │                                                              │   │
│  │  return {                                                    │   │
│  │      "observation": {                                        │   │
│  │          "echoed_message": "Hello, World!",                  │   │
│  │          "message_length": 13                                │   │
│  │      },                                                      │   │
│  │      "reward": 1.3,                                          │   │
│  │      "done": False                                           │   │
│  │  }                                                           │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  HTTP RESPONSE                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  HTTP/1.1 200 OK                                             │   │
│  │  Content-Type: application/json                              │   │
│  │                                                              │   │
│  │  {                                                           │   │
│  │    "observation": {                                          │   │
│  │      "echoed_message": "Hello, World!",                      │   │
│  │      "message_length": 13                                    │   │
│  │    },                                                        │   │
│  │    "reward": 1.3,                                            │   │
│  │    "done": false                                             │   │
│  │  }                                                           │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  CLIENT: Parse response                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  observation = EchoObservation(                              │   │
│  │      echoed_message="Hello, World!",                         │   │
│  │      message_length=13,                                      │   │
│  │      done=False,                                             │   │
│  │      reward=1.3                                              │   │
│  │  )                                                           │   │
│  │                                                              │   │
│  │  return StepResult(                                          │   │
│  │      observation=observation,                                │   │
│  │      reward=1.3,                                             │   │
│  │      done=False                                              │   │
│  │  )                                                           │   │
│  └──────┬───────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
│  AGENT CODE: Receive result                                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  result = StepResult[EchoObservation](...)                   │   │
│  │                                                              │   │
│  │  # Agent processes result:                                   │   │
│  │  observation = result.observation                            │   │
│  │  reward = result.reward          # 1.3                       │   │
│  │  done = result.done              # False                     │   │
│  │                                                              │   │
│  │  # Access observation fields:                                │   │
│  │  message = observation.echoed_message  # "Hello, World!"     │   │
│  │  length = observation.message_length   # 13                  │   │
│  │                                                              │   │
│  │  # Agent decision loop continues...                          │   │
│  │  if not done:                                                │   │
│  │      next_action = agent.decide(observation, reward)         │   │
│  │      next_result = client.step(next_action)                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Action Object Structure

### Base Action Class

**File**: `/home/user/OpenEnv/src/core/env_server/types.py`

```python
@dataclass(kw_only=True)
class Action:
    """Base class for all environment actions."""
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Structure:**
- `metadata`: Optional dictionary for debugging, tracing, etc.

### Environment-Specific Action Classes

Environments define their own Action subclasses:

#### Example 1: Echo Environment Action

**File**: `/home/user/OpenEnv/src/envs/echo_env/models.py`

```python
@dataclass(kw_only=True)
class EchoAction(Action):
    """Action for the Echo environment - just a message to echo."""
    message: str
```

**Structure:**
- `message`: The text message to echo back
- `metadata`: (inherited) Optional dict

**Example Instance:**
```python
action = EchoAction(message="Hello, World!")

# JSON serialization:
{
    "message": "Hello, World!",
    "metadata": {}
}
```

#### Example 2: Connect4 Action

**File**: `/home/user/OpenEnv/src/envs/connect4_env/models.py`

```python
@dataclass(kw_only=True)
class Connect4Action(Action):
    """Action for Connect4 - drop a piece in a column."""
    column: int
```

**Structure:**
- `column`: Column index (0-6) to drop piece
- `metadata`: (inherited) Optional dict

**Example Instance:**
```python
action = Connect4Action(column=3)

# JSON serialization:
{
    "column": 3,
    "metadata": {}
}
```

#### Example 3: Coding Environment Action

**File**: `/home/user/OpenEnv/src/envs/coding_env/models.py`

```python
@dataclass(kw_only=True)
class CodeAction(Action):
    """Action for code execution environment."""
    code: str
    language: str = "python"
```

**Structure:**
- `code`: Source code to execute
- `language`: Programming language (default: "python")
- `metadata`: (inherited) Optional dict

**Example Instance:**
```python
action = CodeAction(
    code="print('Hello, World!')",
    language="python"
)

# JSON serialization:
{
    "code": "print('Hello, World!')",
    "language": "python",
    "metadata": {}
}
```

### Action Serialization

**Client Side** (`HTTPEnvClient._step_payload()`):
```python
def _step_payload(self, action: EchoAction) -> dict:
    """Convert Action object to JSON dict."""
    return asdict(action)
```

**Server Side** (`HTTPEnvServer._deserialize_action()`):
```python
def _deserialize_action(self, action_data: dict) -> Action:
    """Convert JSON dict to Action object."""
    metadata = action_data.pop("metadata", {})
    action = self.action_cls(**action_data)
    action.metadata = metadata
    return action
```

---

## Observation Object Structure

### Base Observation Class

**File**: `/home/user/OpenEnv/src/core/env_server/types.py`

```python
@dataclass(kw_only=True)
class Observation:
    """Base class for all environment observations."""
    done: bool = False
    reward: Union[bool, int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Required Fields:**
- `done`: Is the episode finished?
- `reward`: Scalar reward for the transition
- `metadata`: Optional diagnostic information

### Environment-Specific Observation Classes

#### Example 1: Echo Environment Observation

**File**: `/home/user/OpenEnv/src/envs/echo_env/models.py`

```python
@dataclass(kw_only=True)
class EchoObservation(Observation):
    """Observation from the Echo environment - the echoed message."""
    echoed_message: str
    message_length: int = 0
```

**Complete Structure:**
```python
EchoObservation(
    echoed_message="Hello, World!",  # Environment-specific
    message_length=13,                # Environment-specific
    done=False,                       # Inherited from Observation
    reward=1.3,                       # Inherited from Observation
    metadata={"step": 1}              # Inherited from Observation
)
```

**JSON Serialization** (for HTTP):
```json
{
  "observation": {
    "echoed_message": "Hello, World!",
    "message_length": 13
  },
  "reward": 1.3,
  "done": false
}
```

**Note**: `metadata` is stripped during serialization (not sent over HTTP).

#### Example 2: Connect4 Observation

```python
@dataclass(kw_only=True)
class Connect4Observation(Observation):
    """Observation from Connect4 environment."""
    board: List[List[int]]    # 6x7 grid (0=empty, 1=player1, 2=player2)
    next_player: int          # 1 or 2
    winner: Optional[int]     # None, 1, 2, or 0 (draw)
```

**Example Instance:**
```python
Connect4Observation(
    board=[
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]
    ],
    next_player=2,
    winner=None,
    done=False,
    reward=0.0
)
```

#### Example 3: Coding Environment Observation

```python
@dataclass(kw_only=True)
class CodeObservation(Observation):
    """Observation from code execution."""
    stdout: str
    stderr: str
    exit_code: int
```

**Example Instance:**
```python
CodeObservation(
    stdout="Hello, World!\n",
    stderr="",
    exit_code=0,
    done=True,   # Code execution finished
    reward=1.0   # Success reward
)
```

### Observation Serialization

**Server Side** (`HTTPEnvServer._serialize_observation()`):
```python
def _serialize_observation(self, observation: Observation) -> dict:
    """Convert Observation to JSON-compatible dict."""
    obs_dict = asdict(observation)

    # Extract fields that go at top level
    reward = obs_dict.pop("reward", None)
    done = obs_dict.pop("done", False)
    obs_dict.pop("metadata", None)  # Remove metadata

    return {
        "observation": obs_dict,  # Environment-specific fields
        "reward": reward,          # Top-level reward
        "done": done,              # Top-level done flag
    }
```

**Client Side** (`HTTPEnvClient._parse_result()`):
```python
def _parse_result(self, payload: dict) -> StepResult[ObsT]:
    """Convert JSON response to StepResult[Observation]."""
    obs_data = payload["observation"]

    observation = self.observation_cls(
        **obs_data,
        done=payload["done"],
        reward=payload["reward"]
    )

    return StepResult(
        observation=observation,
        reward=payload["reward"],
        done=payload["done"]
    )
```

---

## StepResult Tuple Structure

**File**: `/home/user/OpenEnv/src/core/client_types.py`

```python
@dataclass
class StepResult(Generic[ObsT]):
    """
    Represents the result of one environment step.

    Attributes:
        observation: The environment's observation after the action.
        reward: Scalar reward for this step (optional).
        done: Whether the episode is finished.
    """
    observation: ObsT
    reward: Optional[float] = None
    done: bool = False
```

### Usage Pattern

```python
# After reset()
result: StepResult[EchoObservation] = client.reset()

# Access fields
observation = result.observation  # Type: EchoObservation
reward = result.reward            # Type: float | None
done = result.done                # Type: bool

# Access observation fields
message = observation.echoed_message
length = observation.message_length

# After step()
action = EchoAction(message="Test")
result: StepResult[EchoObservation] = client.step(action)

# Same access pattern
print(f"Reward: {result.reward}")
print(f"Done: {result.done}")
print(f"Message: {result.observation.echoed_message}")
```

### Type Safety

The `Generic[ObsT]` allows type checkers to know the observation type:

```python
# Type checker knows this
result: StepResult[EchoObservation] = client.reset()
message: str = result.observation.echoed_message  # ✅ Type-safe

# Type checker catches this
length: int = result.observation.nonexistent_field  # ❌ Error!
```

---

## Serialization and Deserialization

### JSON Schema

#### Reset Response
```json
{
  "observation": {
    // Environment-specific fields only
    "echoed_message": "Echo environment ready!",
    "message_length": 0
  },
  "reward": 0.0,
  "done": false
}
```

#### Step Request
```json
{
  "action": {
    // Environment-specific fields
    "message": "Hello, World!"
  },
  "timeout_s": 15
}
```

#### Step Response
```json
{
  "observation": {
    // Environment-specific fields
    "echoed_message": "Hello, World!",
    "message_length": 13
  },
  "reward": 1.3,
  "done": false
}
```

#### State Query Response
```json
{
  "episode_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "step_count": 5
}
```

### Serialization Pipeline

```
Python Object → asdict() → dict → json.dumps() → JSON String → HTTP
```

### Deserialization Pipeline

```
HTTP → JSON String → json.loads() → dict → **kwargs → Python Object
```

---

## Complete Examples

### Example 1: Echo Environment - Full Interaction

```python
from envs.echo_env import EchoEnv, EchoAction

# 1. Create client
client = EchoEnv.from_docker_image("echo-env:latest")

# 2. Reset
result = client.reset()
print(f"Initial observation: {result.observation.echoed_message}")
# Output: "Echo environment ready!"

# 3. Step 1
action1 = EchoAction(message="Hello")
result1 = client.step(action1)
print(f"Echoed: {result1.observation.echoed_message}")
print(f"Reward: {result1.reward}")
# Output: Echoed: Hello
#         Reward: 0.5

# 4. Step 2
action2 = EchoAction(message="Testing the environment")
result2 = client.step(action2)
print(f"Echoed: {result2.observation.echoed_message}")
print(f"Length: {result2.observation.message_length}")
print(f"Reward: {result2.reward}")
# Output: Echoed: Testing the environment
#         Length: 23
#         Reward: 2.3

# 5. Query state
state = client.state()
print(f"Episode ID: {state.episode_id}")
print(f"Step count: {state.step_count}")
# Output: Episode ID: a1b2c3d4-...
#         Step count: 2

# 6. Cleanup
client.close()
```

### Example 2: Connect4 Environment

```python
from envs.connect4_env import Connect4Env, Connect4Action

client = Connect4Env.from_docker_image("connect4-env:latest")

# Reset
result = client.reset()
print(f"Board shape: {len(result.observation.board)}x{len(result.observation.board[0])}")
print(f"Next player: {result.observation.next_player}")
# Output: Board shape: 6x7
#         Next player: 1

# Player 1 drops in column 3
action = Connect4Action(column=3)
result = client.step(action)
print(f"Next player: {result.observation.next_player}")
print(f"Winner: {result.observation.winner}")
print(f"Done: {result.done}")
# Output: Next player: 2
#         Winner: None
#         Done: False

# Continue playing...
client.close()
```

### Example 3: Coding Environment

```python
from envs.coding_env import CodingEnv, CodeAction

client = CodingEnv.from_docker_image("coding-env:latest")

# Reset
result = client.reset()

# Execute code
action = CodeAction(
    code="print('Hello from Python!')\nprint(2 + 2)",
    language="python"
)
result = client.step(action)
print(f"stdout: {result.observation.stdout}")
print(f"stderr: {result.observation.stderr}")
print(f"exit_code: {result.observation.exit_code}")
print(f"reward: {result.reward}")
# Output: stdout: Hello from Python!
#                 4
#         stderr:
#         exit_code: 0
#         reward: 1.0

client.close()
```

---

## Advanced Topics

### Transform Pipeline

Transforms can modify observations before they reach the agent:

```python
class RewardScalingTransform(Transform):
    def __call__(self, obs: Observation) -> Observation:
        # Scale reward to [0, 1]
        if obs.reward is not None:
            obs.reward = min(max(obs.reward / 10.0, 0.0), 1.0)
        return obs

# Use with environment
env = EchoEnvironment(transform=RewardScalingTransform())
```

**Data flow with transform:**
```
env.step(action) → raw_observation → transform(obs) → transformed_obs → client
```

### Error Handling

```python
try:
    result = client.step(action)
except requests.HTTPError as e:
    # Handle HTTP errors (400, 500, etc.)
    print(f"HTTP error: {e}")
except requests.Timeout:
    # Handle timeout
    print("Request timed out")
except Exception as e:
    # Handle other errors
    print(f"Unexpected error: {e}")
```

### Custom Serialization

Some environments may need custom serialization:

```python
class MyEnvClient(HTTPEnvClient[MyAction, MyObservation]):
    def _step_payload(self, action: MyAction) -> dict:
        # Custom serialization for complex types
        return {
            "field1": action.field1,
            "field2": action.field2.to_dict(),  # Custom serialization
        }

    def _parse_result(self, payload: dict) -> StepResult[MyObservation]:
        # Custom deserialization
        obs_data = payload["observation"]
        observation = MyObservation(
            field1=obs_data["field1"],
            field2=ComplexType.from_dict(obs_data["field2"]),
            done=payload["done"],
            reward=payload["reward"]
        )
        return StepResult(observation=observation, ...)
```

---

## Summary

**Action Object:**
- Base class: `Action` with `metadata` field
- Environment-specific subclasses add custom fields
- Serialized to JSON dict for HTTP transport

**Observation Object:**
- Base class: `Observation` with `done`, `reward`, `metadata`
- Environment-specific subclasses add custom fields
- Serialized to `{observation: {...}, reward: ..., done: ...}` structure

**StepResult Tuple:**
- Wrapper containing `observation`, `reward`, `done`
- Type-safe with `Generic[ObsT]`
- Same structure for both `reset()` and `step()` returns

**Data Flow:**
1. Agent creates Action object
2. Client serializes to JSON and sends HTTP POST
3. Server deserializes to Action object
4. Environment processes action, returns Observation
5. Server serializes Observation to JSON
6. Client deserializes to StepResult[Observation]
7. Agent receives typed result

This architecture enables **type-safe**, **distributed**, **language-agnostic** agent-environment interactions while maintaining clean separation of concerns.
