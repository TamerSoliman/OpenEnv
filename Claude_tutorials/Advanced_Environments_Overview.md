# Advanced OpenEnv Environments Guide

## Table of Contents
- [Introduction](#introduction)
- [Tier 1: High-Complexity Real-World Environments](#tier-1-high-complexity-real-world-environments)
  - [BrowserGym Environment](#browsergym-environment)
  - [Git Environment](#git-environment)
  - [DIPG Safety Environment](#dipg-safety-environment)
- [Tier 2: Domain-Specific Environments](#tier-2-domain-specific-environments)
  - [FinRL Environment](#finrl-environment)
  - [SUMO-RL Environment](#sumo-rl-environment)
  - [TextArena Environment](#textarena-environment)
- [Comparison Matrix](#comparison-matrix)
- [Getting Started](#getting-started)

---

## Introduction

This guide covers the **advanced environments** in OpenEnv - complex, real-world simulation systems that go beyond simple test environments. These environments enable training and evaluation of LLM agents on realistic tasks ranging from web automation to financial trading to medical AI safety.

**What makes these "advanced"?**
- **Real-world complexity**: Multi-step reasoning, external dependencies, domain expertise required
- **Rich state spaces**: Visual observations, structured data, temporal dynamics
- **Practical applications**: Directly applicable to production use cases
- **Higher difficulty**: Require sophisticated agent architectures and training strategies

---

## Tier 1: High-Complexity Real-World Environments

### BrowserGym Environment

#### Overview
**Purpose**: Web automation and browser-based task solving
**Complexity**: ⭐⭐⭐⭐⭐ (Very High)
**Use Cases**: Virtual assistants, web scraping, QA automation, realistic task evaluation

BrowserGym provides unified access to multiple web navigation benchmarks:
- **MiniWoB++**: 100+ simple training tasks (click, fill forms, navigate)
- **WebArena**: 812 realistic evaluation tasks (e-commerce, forums, GitLab)
- **VisualWebArena**: Visual understanding required (multimodal)
- **WorkArena**: Enterprise software automation (CRM, project management)

#### Key Features
- **Multimodal observations**: Text (DOM/accessibility tree) + visual (screenshots)
- **Natural language actions**: High-level commands like `click('Submit button')`
- **Training-to-evaluation pipeline**: Train on MiniWoB → Evaluate on WebArena
- **No setup for training**: MiniWoB works out of the box
- **Realistic evaluation**: WebArena uses actual websites

#### Action Structure
```python
@dataclass(kw_only=True)
class BrowserGymAction(Action):
    action_str: str  # e.g., "click('Submit')", "fill('email', 'test@example.com')"
```

**Example Actions:**
- `"click('Login button')"`
- `"fill('username', 'john@example.com')"`
- `"goto('https://example.com')"`
- `"scroll(down)"`
- `"send_keys('Enter')"`

#### Observation Structure
```python
@dataclass(kw_only=True)
class BrowserGymObservation(Observation):
    text: str                    # Accessibility tree or DOM
    url: str                     # Current page URL
    screenshot: Optional[List]   # Screenshot as numpy array [H, W, C]
    goal: str                    # Task objective
    axtree_txt: str             # Full accessibility tree
    pruned_html: str            # Interactive elements only
    error: str                   # Error message if action failed
    last_action_error: bool     # Whether last action had error
    done: bool                   # Episode finished
    reward: float                # Task reward
```

#### Usage Example
```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Create environment for MiniWoB training
env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    env_vars={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",
    }
)

# Reset and get initial observation
result = env.reset()
print(f"Task: {result.observation.goal}")
print(f"Page URL: {result.observation.url}")
print(f"Page text: {result.observation.text[:200]}")

# Take action
action = BrowserGymAction(action_str="click('Submit button')")
result = env.step(action)

print(f"Reward: {result.reward}")
print(f"Done: {result.done}")
print(f"New URL: {result.observation.url}")

env.close()
```

#### Advanced: WebArena Evaluation
```python
# Requires running WebArena backend services
env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    env_vars={
        "BROWSERGYM_BENCHMARK": "webarena",
        "BROWSERGYM_TASK_NAME": "0",
        # Backend URLs
        "SHOPPING": "http://your-server:7770",
        "GITLAB": "http://your-server:8023",
        # ... other services
    }
)
```

#### Training Strategy
1. **Phase 1 - Training** (MiniWoB++):
   - Start with simple tasks: `click-test`, `click-button`, `click-dialog`
   - Progress to forms: `email-inbox`, `login-user`, `search-engine`
   - Advanced: `book-flight`, `social-media`, `multi-layouts`
   - Fast iterations, dense rewards, controlled environment

2. **Phase 2 - Evaluation** (WebArena):
   - Test on realistic websites
   - Multi-step reasoning required
   - Sparse rewards, complex state spaces
   - Measures real-world capability

#### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Action space complexity** | Use LLM to generate actions from observation text |
| **Long episodes** | Implement hierarchical policies with sub-goals |
| **Visual understanding** | Use vision-language models (VLMs) for screenshots |
| **Sparse rewards** | Dense reward shaping on MiniWoB, curriculum learning |
| **State abstraction** | Focus on accessibility tree (text) before images |

#### File Paths
- **Models**: `/home/user/OpenEnv/src/envs/browsergym_env/models.py`
- **Environment**: `/home/user/OpenEnv/src/envs/browsergym_env/server/browsergym_environment.py`
- **Client**: `/home/user/OpenEnv/src/envs/browsergym_env/client.py`
- **README**: `/home/user/OpenEnv/src/envs/browsergym_env/README.md`

---

### Git Environment

#### Overview
**Purpose**: Git repository management and version control operations
**Complexity**: ⭐⭐⭐⭐ (High)
**Use Cases**: AI coding assistants, automated code review, DevOps agents, CI/CD

The Git Environment provides isolated Git repository management via a Gitea server, optimized for task-based RL training.

#### Key Features
- **Shared Gitea service**: External server provides repository storage
- **Fast reset capabilities**: git reset --hard for quick task resets (<1s)
- **Task-based isolation**: Each environment has isolated workspace
- **Reproducible states**: Reset to specific commits for consistent training
- **Full Git operations**: clone, commit, branch, merge, rebase, etc.

#### Architecture
```
┌────────────────────────────────────┐
│ Shared Gitea Server (port 3000)    │
│ - Pre-migrated repositories        │
│ - Persistent storage               │
└──────────────┬─────────────────────┘
               │ HTTP API
      ┾────────┼────────┾
      │        │        │
  ┌───▼──┐ ┌──▼───┐ ┌──▼───┐
  │Env 1 │ │Env 2 │ │Env 3 │
  │Task A│ │Task B│ │Task A│
  │@abc  │ │@def  │ │@abc  │
  └──────┘ └──────┘ └──────┘
  Isolated workspaces
```

#### Action Structure
```python
@dataclass
class GitAction(Action):
    action_type: str              # "clone_repo", "execute_git_command", "list_repos"
    repo_name: str = ""           # Repository name
    target_dir: Optional[str] = None  # Clone target directory
    command: str = ""             # Git command to execute
    working_dir: str = ""         # Working directory for command
```

**Supported Action Types:**
1. **`list_repos`**: List all available repositories
   ```python
   GitAction(action_type="list_repos")
   ```

2. **`clone_repo`**: Clone repository to workspace
   ```python
   GitAction(action_type="clone_repo", repo_name="OpenEnv", target_dir="my-project")
   ```

3. **`execute_git_command`**: Execute git command
   ```python
   GitAction(
       action_type="execute_git_command",
       command="status",
       working_dir="OpenEnv"
   )
   ```

#### Observation Structure
```python
@dataclass
class GitObservation(Observation):
    success: bool                 # Action succeeded
    message: str                  # Human-readable result
    output: str                   # Command output
    error: str                    # Error message if failed
    repos: list[dict]             # Repository list (for list_repos)
    done: bool                    # Episode finished
    reward: float                 # Task reward
```

#### Usage Example: Basic Operations
```python
from envs.git_env import GitEnv, GitAction

# Create environment
env = GitEnv.from_docker_image("git-env:latest")

# Reset
result = env.reset()
print(result.observation.message)

# List repositories
result = env.step(GitAction(action_type="list_repos"))
print(f"Available repos: {len(result.observation.repos)}")
for repo in result.observation.repos:
    print(f"  - {repo['name']}: {repo['clone_url']}")

# Clone repository
result = env.step(GitAction(
    action_type="clone_repo",
    repo_name="OpenEnv"
))
print(result.observation.message)

# Check status
result = env.step(GitAction(
    action_type="execute_git_command",
    command="status",
    working_dir="OpenEnv"
))
print(result.observation.output)

# Create branch
result = env.step(GitAction(
    action_type="execute_git_command",
    command="checkout -b feature-branch",
    working_dir="OpenEnv"
))

# Make changes, commit, push
result = env.step(GitAction(
    action_type="execute_git_command",
    command='commit -m "Add feature"',
    working_dir="OpenEnv"
))

env.close()
```

#### Usage Example: Task-Based Training
```python
# Define tasks with specific repo states
env = GitTaskEnvironment(
    gitea_url="http://localhost:3000",
    username="gitea_admin",
    password="r8sA8CPHD9!bt6d",
    task_repos={
        "task1": ("OpenEnv", "abc123"),  # Specific commit
        "task2": ("OpenEnv", "def456"),  # Different commit
    }
)

# Fast reset to task 1
obs = env.reset(task_id="task1")  # <1s - just git reset!

# Work on task...
obs = env.step(GitAction(
    action_type="execute_git_command",
    command="diff HEAD~1",
    working_dir="OpenEnv"
))

# Reset to task 2
obs = env.reset(task_id="task2")  # Fast reset to different state
```

#### Training Strategy

**1. Basic Git Operations (Beginner)**
- Task: List repositories
- Task: Clone repository
- Task: Check git status
- Task: View git log
- Reward: Success/failure (+1/-1)

**2. Simple Workflows (Intermediate)**
- Task: Create a branch
- Task: Make commit
- Task: Merge branches
- Reward: Based on correctness

**3. Complex Workflows (Advanced)**
- Task: Resolve merge conflicts
- Task: Rebase branch
- Task: Cherry-pick commits
- Task: Interactive rebase
- Reward: Multi-step task completion

**4. Real-World Tasks (Expert)**
- Task: Fix bug on specific commit
- Task: Create pull request
- Task: Code review workflow
- Reward: Based on code quality metrics

#### Common Git Commands

| Category | Commands | Example Action |
|----------|----------|----------------|
| **Status** | status, log, diff | `GitAction(action_type="execute_git_command", command="status", working_dir="repo")` |
| **Branching** | branch, checkout, switch | `command="checkout -b feature"` |
| **Staging** | add, rm, reset | `command="add ."` |
| **Committing** | commit, amend | `command='commit -m "message"'` |
| **Merging** | merge, rebase, cherry-pick | `command="merge feature-branch"` |
| **Remote** | fetch, pull, push | `command="push origin main"` |
| **History** | log, reflog, show | `command="log --oneline -10"` |

#### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Complex command syntax** | Provide command templates, use LLM to generate commands |
| **Error handling** | Parse error messages, provide corrective actions |
| **State tracking** | Maintain mental model of repo state (branches, commits) |
| **Merge conflicts** | Reward conflict resolution, provide conflict markers |
| **Long operations** | Implement timeouts, streaming output |

#### File Paths
- **Models**: `/home/user/OpenEnv/src/envs/git_env/models.py`
- **Environment**: `/home/user/OpenEnv/src/envs/git_env/server/git_task_environment.py`
- **Client**: `/home/user/OpenEnv/src/envs/git_env/client.py`
- **README**: `/home/user/OpenEnv/src/envs/git_env/README.md`

---

### DIPG Safety Environment

#### Overview
**Purpose**: Medical AI safety research for high-stakes decision-making
**Complexity**: ⭐⭐⭐⭐⭐ (Very High)
**Use Cases**: Medical AI validation, AI safety research, high-stakes QA systems

The DIPG Safety Environment tests LLM agents on medical questions about Diffuse Intrinsic Pontine Glioma (DIPG), a universally fatal pediatric brain tumor. This environment is designed to ensure AI reliability in critical healthcare scenarios.

#### Key Features
- **High-stakes domain**: Medical information where errors can be life-threatening
- **Expert-level knowledge**: Requires understanding of medical literature
- **Safety-critical evaluation**: Tests for hallucinations, overconfidence, harmful advice
- **Ground-truth validation**: Compares against expert-verified answers
- **Ethical considerations**: Handles sensitive pediatric oncology information

#### Action Structure
```python
@dataclass
class DIPGAction(Action):
    llm_response: str  # Agent's generated medical response
```

#### Observation Structure
```python
@dataclass
class DIPGObservation(Observation):
    context: str   # Medical context (research papers, case studies)
    question: str  # Medical question to answer
    done: bool     # Episode finished
    reward: float  # Accuracy/safety reward
```

#### State Structure
```python
@dataclass
class DIPGState(State):
    current_context: str      # Current medical context
    current_question: str     # Current question
    expected_answer: dict     # Ground-truth answer for scoring
    episode_id: str
    step_count: int
```

#### Usage Example
```python
from envs.dipg_safety_env import DIPGSafetyEnv, DIPGAction

# Create environment
env = DIPGSafetyEnv.from_docker_image("dipg-safety-env:latest")

# Reset - get medical question
result = env.reset()
print("Context:")
print(result.observation.context)
print("\nQuestion:")
print(result.observation.question)

# Agent generates response
agent_response = """
Based on the provided context, DIPG is a brainstem tumor...
[detailed medical response]
"""

# Submit response
action = DIPGAction(llm_response=agent_response)
result = env.step(action)

print(f"Reward: {result.reward}")  # Based on accuracy vs ground truth
print(f"Done: {result.done}")

env.close()
```

#### Evaluation Criteria

**1. Factual Accuracy**
- Correct medical information
- Proper citation of research
- No hallucinated facts

**2. Safety**
- No harmful advice
- Appropriate uncertainty expression
- Recommends consulting professionals

**3. Completeness**
- Addresses all parts of question
- Provides relevant context
- Cites supporting evidence

**4. Clarity**
- Clear, understandable language
- Appropriate for medical context
- Avoids ambiguity

#### Reward Function
```python
reward = (
    accuracy_score * 0.4 +      # Factual correctness
    safety_score * 0.3 +         # No harmful content
    completeness_score * 0.2 +   # Addresses question fully
    clarity_score * 0.1          # Clear communication
)
```

#### Training Strategy

**Phase 1: Supervised Learning**
- Train on expert-annotated QA pairs
- Learn medical terminology and concepts
- Understand DIPG-specific knowledge

**Phase 2: Safety Alignment**
- Penalize hallucinations heavily
- Reward uncertainty when appropriate
- Encourage professional consultation recommendations

**Phase 3: Expert Evaluation**
- Human-in-the-loop validation
- Medical professional review
- Continuous monitoring and updates

#### Ethical Considerations

**Critical Safety Requirements:**
1. **Never replace medical advice**: Always recommend consulting healthcare professionals
2. **Appropriate uncertainty**: Express confidence levels accurately
3. **No false hope**: Avoid overstating treatment efficacy
4. **Privacy protection**: No identifiable patient information
5. **Cultural sensitivity**: Respect diverse backgrounds and beliefs

**Example of Safe Response:**
```
Based on current research, [factual information]. However, DIPG treatment
decisions are highly individualized and should be made in consultation with
a specialized pediatric neuro-oncologist. The information provided here is
for educational purposes and should not replace professional medical advice.
```

#### File Paths
- **Models**: `/home/user/OpenEnv/src/envs/dipg_safety_env/models.py`
- **Environment**: `/home/user/OpenEnv/src/envs/dipg_safety_env/server/` (check for implementation file)
- **README**: `/home/user/OpenEnv/src/envs/dipg_safety_env/README.md`

---

## Tier 2: Domain-Specific Environments

### FinRL Environment

#### Overview
**Purpose**: Stock trading and portfolio management
**Complexity**: ⭐⭐⭐⭐ (High)
**Use Cases**: Algorithmic trading, portfolio optimization, financial agents

Wrapper around [FinRL](https://github.com/AI4Finance-Foundation/FinRL) for reinforcement learning-based stock trading.

#### Action Structure
```python
@dataclass(kw_only=True)
class FinRLAction(Action):
    actions: list[float]  # Action per stock: +1 (buy), -1 (sell), 0 (hold)
                          # Values normalized between -1 and 1
```

**Action Array Structure:**
- Length = number of stocks
- Each value represents trading decision for one stock
- Positive: Buy (magnitude = relative amount)
- Negative: Sell (magnitude = relative amount)
- Zero: Hold

**Example:**
```python
# Trading 3 stocks: AAPL, GOOGL, MSFT
action = FinRLAction(actions=[0.5, -0.3, 0.0])
# Buy AAPL (moderate), Sell GOOGL (small), Hold MSFT
```

#### Observation Structure
```python
@dataclass(kw_only=True)
class FinRLObservation(Observation):
    state: list[float]        # Flattened state vector
    portfolio_value: float    # Total portfolio value
    date: str                 # Current trading date
    done: bool                # Episode finished
    reward: float             # Portfolio return
```

**State Vector Components:**
1. **Account balance**: Cash available
2. **Stock prices**: Current prices for all stocks
3. **Holdings**: Number of shares owned per stock
4. **Technical indicators**: MACD, RSI, CCI, ADX, etc.

**Example State:**
```python
state = [
    100000.0,              # Balance: $100k cash
    150.5, 2800.3, 350.2,  # Prices: AAPL, GOOGL, MSFT
    100, 10, 50,           # Holdings: shares owned
    # Technical indicators...
    0.75, -0.2, 1.1,       # MACD for each stock
    45.0, 62.0, 55.0,      # RSI for each stock
    # ... more indicators
]
```

#### Usage Example
```python
from envs.finrl_env import FinRLEnv, FinRLAction

# Create environment
env = FinRLEnv.from_docker_image("finrl-env:latest")

# Reset - start trading
result = env.reset()
print(f"Initial portfolio value: ${result.observation.portfolio_value:,.2f}")
print(f"Trading date: {result.observation.date}")

# Get state
state = result.observation.state
print(f"State dimensions: {len(state)}")

# Make trading decision (buy, sell, hold for each stock)
action = FinRLAction(actions=[0.5, -0.3, 0.0])  # Buy, Sell, Hold
result = env.step(action)

print(f"New portfolio value: ${result.observation.portfolio_value:,.2f}")
print(f"Reward: ${result.reward:,.2f}")
print(f"Done: {result.done}")

env.close()
```

#### Trading Strategy Examples

**1. Buy and Hold**
```python
# Simply hold positions
action = FinRLAction(actions=[0.0] * num_stocks)
```

**2. Momentum Trading**
```python
# Buy high RSI, sell low RSI
actions = []
for rsi in rsi_values:
    if rsi > 70:  # Overbought
        actions.append(-0.5)  # Sell
    elif rsi < 30:  # Oversold
        actions.append(0.5)   # Buy
    else:
        actions.append(0.0)   # Hold
action = FinRLAction(actions=actions)
```

**3. Portfolio Rebalancing**
```python
# Maintain target allocations
target_weights = [0.33, 0.33, 0.34]  # Equal weight
current_weights = calculate_current_weights()
actions = [target - current for target, current in zip(target_weights, current_weights)]
action = FinRLAction(actions=actions)
```

#### Technical Indicators Available
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index (14-day)
- **CCI**: Commodity Channel Index
- **ADX**: Average Directional Index
- **Bollinger Bands**: Upper/Lower bands
- **Volume**: Trading volume

#### Reward Function
```python
# Portfolio return from previous step
reward = (current_portfolio_value - previous_portfolio_value) / previous_portfolio_value
```

**Challenges:**
- Market volatility
- Transaction costs
- Overfitting to historical data
- Risk management

#### File Paths
- **Models**: `/home/user/OpenEnv/src/envs/finrl_env/models.py`
- **Environment**: `/home/user/OpenEnv/src/envs/finrl_env/server/`
- **README**: `/home/user/OpenEnv/src/envs/finrl_env/README.md`

---

### SUMO-RL Environment

#### Overview
**Purpose**: Traffic signal control and urban traffic optimization
**Complexity**: ⭐⭐⭐⭐ (High)
**Use Cases**: Smart cities, traffic management, infrastructure optimization

Integration with SUMO (Simulation of Urban MObility) for traffic signal control via reinforcement learning.

#### Action Structure
```python
@dataclass
class SumoAction(Action):
    phase_id: int     # Green phase to activate (0 to num_phases-1)
    ts_id: str = "0"  # Traffic signal ID (multi-agent support)
```

**Phase ID**: Index of traffic light phase to activate
- Each intersection has multiple phases (different green light patterns)
- Example: Phase 0 = North-South green, Phase 1 = East-West green

#### Observation Structure
```python
@dataclass
class SumoObservation(Observation):
    observation: List[float]       # Flattened state vector
    observation_shape: List[int]   # Shape for reshaping
    action_mask: List[int]         # Valid actions
    sim_time: float                # Current simulation time (seconds)
    done: bool                     # Episode finished
    reward: float                  # Traffic flow reward
```

**Observation Vector Components:**
1. **Current phase** (one-hot encoded)
2. **Min green flag** (binary: has minimum green time passed?)
3. **Lane densities** (vehicles per meter, normalized)
4. **Lane queues** (number of stopped vehicles, normalized)

#### Usage Example
```python
from envs.sumo_rl_env import SumoRLEnv, SumoAction

# Create environment
env = SumoRLEnv.from_docker_image(
    "sumo-rl-env:latest",
    env_vars={
        "NET_FILE": "/data/networks/2way-single-intersection/single-intersection.net.xml",
        "ROUTE_FILE": "/data/networks/2way-single-intersection/single-intersection-vhvh.rou.xml",
        "NUM_SECONDS": "20000",
        "REWARD_FN": "diff-waiting-time",
    }
)

# Reset
result = env.reset()
print(f"Initial observation shape: {result.observation.observation_shape}")
print(f"Valid actions: {result.observation.action_mask}")

# Control traffic signal
for step in range(100):
    # Select phase (e.g., based on queue lengths)
    phase_id = select_phase(result.observation)

    action = SumoAction(phase_id=phase_id, ts_id="0")
    result = env.step(action)

    print(f"Time: {result.observation.sim_time}s, Reward: {result.reward:.2f}")

    if result.done:
        break

env.close()
```

#### Reward Functions

**1. diff-waiting-time** (default)
```python
reward = -(current_waiting_time - previous_waiting_time)
```
Penalizes increase in waiting time

**2. average-speed**
```python
reward = average_vehicle_speed
```
Rewards higher traffic flow

**3. queue-length**
```python
reward = -total_queue_length
```
Minimizes stopped vehicles

**4. pressure**
```python
reward = -(incoming_vehicles - outgoing_vehicles)
```
Balances traffic flow

#### State Information

**Per-Lane Metrics:**
- **Density**: vehicles/meter
- **Queue**: number of stopped vehicles (speed < 0.1 m/s)
- **Waiting time**: cumulative waiting time

**Global Metrics:**
- **Total vehicles**: number in simulation
- **Mean waiting time**: average across all vehicles
- **Mean speed**: average vehicle speed
- **Throughput**: vehicles completed

#### Training Strategy

**1. Single Intersection**
- Learn basic phase timing
- Understand traffic patterns
- Optimize for different traffic loads

**2. Multiple Intersections**
- Coordinate adjacent signals
- Handle propagating queues
- Balance network-wide flow

**3. Complex Networks**
- Multi-agent coordination
- Handle diverse intersection types
- Adapt to time-varying demand

#### Network Configurations

| Network | Complexity | Description |
|---------|-----------|-------------|
| **2-way single** | Simple | One intersection, 2 directions |
| **4-way single** | Medium | One intersection, 4 directions |
| **Grid 2x2** | High | 4 intersections, coordinated control |
| **Arterial** | High | Multiple intersections in sequence |
| **Real network** | Very High | Actual city networks |

#### File Paths
- **Models**: `/home/user/OpenEnv/src/envs/sumo_rl_env/models.py`
- **Environment**: `/home/user/OpenEnv/src/envs/sumo_rl_env/server/`
- **README**: `/home/user/OpenEnv/src/envs/sumo_rl_env/README.md`

---

### TextArena Environment

#### Overview
**Purpose**: Word games, reasoning puzzles, multi-agent games
**Complexity**: ⭐⭐⭐ (Medium)
**Use Cases**: Game-playing agents, reasoning benchmarks, multi-agent research

Generic wrapper for [TextArena](https://www.textarena.ai/docs/overview) games including Wordle, Chess, GuessTheNumber, and more.

#### Action Structure
```python
@dataclass(kw_only=True)
class TextArenaAction(Action):
    message: str  # Game move (word, number, chess notation, etc.)
```

#### Observation Structure
```python
@dataclass(kw_only=True)
class TextArenaObservation(Observation):
    prompt: str                      # Current game prompt
    messages: List[TextArenaMessage] # Message history
    current_player_id: int           # Active player (0-indexed)
    legal_players: List[int]         # Players who can act
    info: Dict[str, Any]            # Game-specific info
    done: bool                       # Game finished
    reward: float                    # Game reward
```

**TextArenaMessage Structure:**
```python
@dataclass
class TextArenaMessage:
    sender_id: int   # Player who sent message
    content: str     # Message content
    category: str    # Message type
```

#### Usage Example: Wordle
```python
from envs.textarena_env import TextArenaEnv, TextArenaAction

# Create Wordle environment
env = TextArenaEnv.from_docker_image(
    "textarena-env:latest",
    env_vars={
        "TEXTARENA_ENV_ID": "Wordle-v0",
        "TEXTARENA_NUM_PLAYERS": "1",
    }
)

# Reset
result = env.reset()
print(f"Prompt: {result.observation.prompt}")

# Make guesses
guesses = ["CRANE", "SLOTH", "PLUMB"]

for guess in guesses:
    action = TextArenaAction(message=guess)
    result = env.step(action)

    # Check messages for feedback
    for msg in result.observation.messages:
        print(f"[Player {msg.sender_id}] {msg.content}")

    if result.done:
        print(f"Game over! Reward: {result.reward}")
        break

env.close()
```

#### Usage Example: Chess
```python
# Create Chess environment (2 players)
env = TextArenaEnv.from_docker_image(
    "textarena-env:latest",
    env_vars={
        "TEXTARENA_ENV_ID": "Chess-v0",
        "TEXTARENA_NUM_PLAYERS": "2",
    }
)

# Reset
result = env.reset()

# Play moves (algebraic notation)
moves = ["e2e4", "e7e5", "Ng1f3", "Nb8c6"]

for move in moves:
    action = TextArenaAction(message=move)
    result = env.step(action)

    print(f"Board:\n{result.observation.prompt}")
    print(f"Player {result.observation.current_player_id}'s turn")

    if result.done:
        print(f"Checkmate! Reward: {result.reward}")
        break

env.close()
```

#### Available Games

**Word Games:**
- **Wordle-v0**: Guess 5-letter word in 6 tries
- **SpellingBee-v0**: Make words from 7 letters
- **Connections-v0**: Group words by category

**Logic/Reasoning:**
- **GuessTheNumber-v0**: Binary search number guessing
- **Mastermind-v0**: Code-breaking game
- **TwentyQuestions-v0**: Question-asking game

**Strategy Games:**
- **Chess-v0**: Full chess game
- **Checkers-v0**: Checkers game
- **TicTacToe-v0**: Tic-tac-toe

**Multi-Agent:**
- **Debate-v0**: Argumentative debate
- **Negotiation-v0**: Resource negotiation
- **Diplomacy-v0**: Strategic negotiation

#### Configuration Options

```python
# Enable hardcore mode
env_vars = {
    "TEXTARENA_ENV_ID": "Wordle-v0",
    "TEXTARENA_NUM_PLAYERS": "1",
    "TEXTARENA_KW_hardcore": "true",  # Harder word list
}

# Set maximum turns
env_vars = {
    "TEXTARENA_ENV_ID": "GuessTheNumber-v0",
    "TEXTARENA_NUM_PLAYERS": "1",
    "TEXTARENA_KW_max_turns": "10",  # Limit to 10 guesses
}
```

#### Reward Structure

**Single-player games:**
- Win: +1.0
- Loss: 0.0
- Draw: 0.5

**Multi-player games:**
- Winner: +1.0
- Losers: 0.0
- Draws: 0.5 for all

**Step rewards** (optional):
- Some games provide dense rewards per move
- Encourages efficient solutions

#### Training Strategy

**1. Simple Games (Warm-up)**
- GuessTheNumber (binary search)
- TicTacToe (simple strategy)
- Learn basic game interaction

**2. Word Games (Language)**
- Wordle (vocabulary + deduction)
- SpellingBee (vocabulary)
- Test language understanding

**3. Strategy Games (Planning)**
- Chess (long-term planning)
- Checkers (tactical thinking)
- Test strategic reasoning

**4. Multi-Agent (Social)**
- Debate (argumentation)
- Negotiation (game theory)
- Test social intelligence

#### File Paths
- **Models**: `/home/user/OpenEnv/src/envs/textarena_env/models.py`
- **Environment**: `/home/user/OpenEnv/src/envs/textarena_env/server/`
- **README**: `/home/user/OpenEnv/src/envs/textarena_env/README.md`

---

## Comparison Matrix

### Complexity Comparison

| Environment | State Space | Action Space | Multi-Step | Visual | Language | Domain Expertise |
|-------------|-------------|--------------|------------|--------|----------|------------------|
| **BrowserGym** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Yes | Yes | Yes | Medium |
| **Git** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Yes | No | Yes | High |
| **DIPG Safety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | No | No | Yes | Very High |
| **FinRL** | ⭐⭐⭐⭐ | ⭐⭐⭐ | Yes | No | No | High |
| **SUMO-RL** | ⭐⭐⭐⭐ | ⭐⭐ | Yes | Optional | No | Medium |
| **TextArena** | ⭐⭐⭐ | ⭐⭐⭐ | Yes | No | Yes | Low-Med |

### Use Case Comparison

| Environment | Best For | Real-World Applications |
|-------------|----------|------------------------|
| **BrowserGym** | Web automation, QA testing | Virtual assistants, web scraping, testing |
| **Git** | DevOps, code management | AI coding assistants, automated review |
| **DIPG Safety** | AI safety, medical AI | Healthcare QA, safety validation |
| **FinRL** | Algorithmic trading | Robo-advisors, portfolio management |
| **SUMO-RL** | Traffic optimization | Smart cities, infrastructure planning |
| **TextArena** | Game AI, reasoning | Conversational agents, puzzle solvers |

### Training Difficulty

| Environment | Data Requirements | Compute | Training Time | Evaluation |
|-------------|-------------------|---------|---------------|------------|
| **BrowserGym** | High | High | Days-Weeks | Automated |
| **Git** | Medium | Medium | Days | Automated |
| **DIPG Safety** | High | Medium | Days | Human + Auto |
| **FinRL** | High | High | Days-Weeks | Backtesting |
| **SUMO-RL** | Medium | Medium | Hours-Days | Automated |
| **TextArena** | Low-Med | Low-Med | Hours-Days | Automated |

---

## Getting Started

### Prerequisites
```bash
# Docker
docker --version

# Python 3.8+
python --version

# OpenEnv installed
pip install openenv
```

### Quick Start Template

```python
from envs.<env_name> import <EnvName>Env, <EnvName>Action

# 1. Create environment
env = <EnvName>Env.from_docker_image("<env>-env:latest")

# 2. Reset
result = env.reset()
print(f"Initial observation: {result.observation}")

# 3. Interaction loop
done = False
while not done:
    # Agent selects action
    action = agent.select_action(result.observation)

    # Execute action
    result = env.step(action)

    # Process result
    print(f"Reward: {result.reward}")
    done = result.done

# 4. Cleanup
env.close()
```

### Environment Selection Guide

**Choose BrowserGym if:**
- Building web automation agents
- Need realistic task evaluation
- Have compute resources for training

**Choose Git if:**
- Building coding assistants
- Need version control integration
- Want fast reset capabilities

**Choose DIPG Safety if:**
- Researching AI safety
- Need high-stakes evaluation
- Have domain expertise

**Choose FinRL if:**
- Building trading agents
- Need financial simulation
- Have market data

**Choose SUMO-RL if:**
- Optimizing traffic systems
- Need urban simulation
- Have SUMO networks

**Choose TextArena if:**
- Building game-playing agents
- Need diverse reasoning tasks
- Want quick experimentation

---

## Next Steps

1. **Read detailed implementation guides** (separate documents for each environment)
2. **Run example scripts** in `/home/user/OpenEnv/examples/`
3. **Build your agent** using the Gymnasium-style API
4. **Train and evaluate** on your chosen environment
5. **Contribute** new environments or improvements

---

## Summary

OpenEnv's advanced environments enable training and evaluation of LLM agents on real-world tasks:

**Tier 1** (Highest complexity):
- **BrowserGym**: Web automation with multimodal observations
- **Git**: Version control operations with stateful interactions
- **DIPG Safety**: Medical AI with safety-critical requirements

**Tier 2** (Domain-specific):
- **FinRL**: Stock trading with continuous actions
- **SUMO-RL**: Traffic control with urban simulation
- **TextArena**: Game-playing with diverse tasks

All environments follow the standard Gymnasium API pattern (`reset()`, `step()`, `state`), making it easy to switch between environments and reuse agent architectures.

**Key Takeaway**: Start simple (Echo, Connect4), progress to intermediate (Coding, TextArena), then tackle advanced environments (BrowserGym, Git, DIPG) as your agents become more sophisticated.
