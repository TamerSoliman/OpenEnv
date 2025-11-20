# BrowserGym Environment: Complete Implementation Guide

## Table of Contents
- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Action Space Deep Dive](#action-space-deep-dive)
- [Observation Space Deep Dive](#observation-space-deep-dive)
- [Complete Implementation Examples](#complete-implementation-examples)
- [Agent Architectures](#agent-architectures)
- [Training Strategies](#training-strategies)
- [Advanced Topics](#advanced-topics)
- [Troubleshooting](#troubleshooting)

---

## Introduction

BrowserGym is the **most complex environment** in OpenEnv, providing a unified framework for training and evaluating web agents. This guide covers everything from basic usage to advanced agent architectures.

**What makes BrowserGym unique:**
- **Multimodal observations**: Text (DOM/AXTree) + Visual (screenshots)
- **Natural language actions**: High-level commands instead of low-level clicks
- **Multi-benchmark support**: Train on simple tasks, evaluate on complex ones
- **Realistic evaluation**: Actual websites, not synthetic simulations

**Learning Path:**
1. Basic interaction (this guide)
2. Simple agents (template-based)
3. LLM-based agents (GPT-4, Claude)
4. Multimodal agents (vision-language models)
5. Advanced agents (hierarchical, memory-augmented)

---

## Environment Setup

### Docker Installation

```bash
# Clone OpenEnv
git clone https://github.com/meta-ai/OpenEnv.git
cd OpenEnv

# Build BrowserGym environment
docker build -f src/envs/browsergym_env/server/Dockerfile \
    -t browsergym-env:latest .

# Test installation
docker run -p 8000:8000 \
    -e BROWSERGYM_BENCHMARK=miniwob \
    -e BROWSERGYM_TASK_NAME=click-test \
    browsergym-env:latest
```

### Python Client Setup

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

# Create environment (automatically handles Docker)
env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    env_vars={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",
    }
)
```

### Benchmark-Specific Setup

#### MiniWoB++ (No Setup Required!)
```python
env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    env_vars={
        "BROWSERGYM_BENCHMARK": "miniwob",
        "BROWSERGYM_TASK_NAME": "click-test",  # or any of 100+ tasks
    }
)
```

#### WebArena (Requires Backend Services)
```bash
# Start WebArena backend (separate terminal)
cd webarena-setup
docker-compose up -d

# Wait for services to be ready (~5 minutes)
# Services: shopping, reddit, gitlab, wikipedia, map, homepage
```

```python
env = BrowserGymEnv.from_docker_image(
    "browsergym-env:latest",
    env_vars={
        "BROWSERGYM_BENCHMARK": "webarena",
        "BROWSERGYM_TASK_NAME": "0",  # Task ID
        # Backend URLs
        "SHOPPING": "http://your-server:7770",
        "REDDIT": "http://your-server:9999",
        "GITLAB": "http://your-server:8023",
        "WIKIPEDIA": "http://your-server:8888",
        "MAP": "http://your-server:3000",
        "HOMEPAGE": "http://your-server:4399",
    }
)
```

---

## Action Space Deep Dive

### Action Format

BrowserGym actions are **natural language strings** that describe browser operations:

```python
@dataclass(kw_only=True)
class BrowserGymAction(Action):
    action_str: str  # Natural language command
```

### Action Categories

#### 1. Navigation Actions

**goto(url)**
```python
# Navigate to URL
action = BrowserGymAction(action_str="goto('https://example.com')")

# Navigate to relative URL
action = BrowserGymAction(action_str="goto('/login')")
```

**go_back()**
```python
# Browser back button
action = BrowserGymAction(action_str="go_back()")
```

**go_forward()**
```python
# Browser forward button
action = BrowserGymAction(action_str="go_forward()")
```

#### 2. Click Actions

**click(element_description)**
```python
# Click by text content
action = BrowserGymAction(action_str="click('Submit button')")

# Click by accessibility label
action = BrowserGymAction(action_str="click('Login')")

# Click specific element
action = BrowserGymAction(action_str="click('button[type=submit]')")
```

**double_click(element_description)**
```python
action = BrowserGymAction(action_str="double_click('File icon')")
```

**right_click(element_description)**
```python
action = BrowserGymAction(action_str="right_click('Context menu')")
```

#### 3. Input Actions

**fill(element_description, text)**
```python
# Fill text input
action = BrowserGymAction(action_str="fill('username', 'john@example.com')")

# Fill password
action = BrowserGymAction(action_str="fill('password', 'secret123')")

# Fill textarea
action = BrowserGymAction(action_str="fill('comment', 'This is a comment')")
```

**press(key)**
```python
# Press Enter
action = BrowserGymAction(action_str="press('Enter')")

# Press Tab
action = BrowserGymAction(action_str="press('Tab')")

# Press Escape
action = BrowserGymAction(action_str="press('Escape')")
```

**send_keys(keys)**
```python
# Type text
action = BrowserGymAction(action_str="send_keys('Hello World')")

# Special keys
action = BrowserGymAction(action_str="send_keys('Control+A')")  # Select all
action = BrowserGymAction(action_str="send_keys('Control+C')")  # Copy
```

#### 4. Selection Actions

**select_option(element_description, option)**
```python
# Select dropdown option by text
action = BrowserGymAction(action_str="select_option('country', 'United States')")

# Select by value
action = BrowserGymAction(action_str="select_option('size', 'large')")
```

**check(element_description)**
```python
# Check checkbox
action = BrowserGymAction(action_str="check('I agree to terms')")
```

**uncheck(element_description)**
```python
# Uncheck checkbox
action = BrowserGymAction(action_str="uncheck('Subscribe to newsletter')")
```

#### 5. Scroll Actions

**scroll(direction)**
```python
# Scroll down
action = BrowserGymAction(action_str="scroll(down)")

# Scroll up
action = BrowserGymAction(action_str="scroll(up)")

# Scroll to element
action = BrowserGymAction(action_str="scroll('footer')")
```

#### 6. Wait Actions

**wait(milliseconds)**
```python
# Wait for page load
action = BrowserGymAction(action_str="wait(1000)")  # 1 second
```

### Action Composition

You can chain actions using multiple step calls:

```python
# Login workflow
result = env.step(BrowserGymAction(action_str="fill('username', 'john@example.com')"))
result = env.step(BrowserGymAction(action_str="fill('password', 'secret123')"))
result = env.step(BrowserGymAction(action_str="click('Login button')"))
result = env.step(BrowserGymAction(action_str="wait(2000)"))  # Wait for redirect
```

---

## Observation Space Deep Dive

### Observation Structure

```python
@dataclass(kw_only=True)
class BrowserGymObservation(Observation):
    # Primary observation modalities
    text: str                      # Accessibility tree or DOM text
    url: str                       # Current page URL
    screenshot: Optional[List]     # Screenshot as array [H, W, C]

    # Task information
    goal: str                      # Task objective/instruction

    # Detailed representations
    axtree_txt: str               # Full accessibility tree
    pruned_html: str              # Simplified HTML (interactive elements)

    # Error handling
    error: str                     # Error message if action failed
    last_action_error: bool       # Whether last action had error

    # Episode information
    done: bool                     # Task completed
    reward: float                  # Task reward (0.0 or 1.0 typically)

    # Additional data
    metadata: dict                 # Raw BrowserGym obs/info
```

### Observation Modalities

#### 1. Text Observations

**Accessibility Tree** (`axtree_txt`)
- **What**: Hierarchical representation of page elements
- **Format**: Indented text with element types and attributes
- **Best for**: Understanding page structure, finding interactive elements
- **Size**: Typically 1-5KB, can be 50KB+ for complex pages

**Example:**
```
RootWebArea 'Login Page'
  heading 'Welcome Back'
  form 'login-form'
    textbox 'Email' [value='']
    textbox 'Password' [type='password'] [value='']
    button 'Submit' [clickable]
  link 'Forgot Password?' [clickable]
```

**Pruned HTML** (`pruned_html`)
- **What**: Simplified HTML with only interactive elements
- **Format**: HTML tags with attributes
- **Best for**: Understanding page semantics
- **Size**: Smaller than full HTML

**Example:**
```html
<form id="login-form">
  <input name="email" type="text" placeholder="Email">
  <input name="password" type="password" placeholder="Password">
  <button type="submit">Submit</button>
</form>
<a href="/forgot-password">Forgot Password?</a>
```

#### 2. Visual Observations

**Screenshots** (`screenshot`)
- **What**: RGB image of current viewport
- **Format**: Numpy array [height, width, 3]
- **Best for**: Visual understanding, handling images/charts
- **Size**: ~2-5MB per screenshot (1280x720 RGB)

**When to use screenshots:**
- Image-heavy tasks (e.g., "click the red button")
- Visual similarity tasks (e.g., CAPTCHA)
- Layout understanding
- Multimodal agents (vision-language models)

**When to skip screenshots:**
- Text-only tasks (saves compute)
- Accessibility tree sufficient
- Training on limited resources

#### 3. URL Information

**Current URL** (`url`)
- Track navigation flow
- Detect successful navigation
- Handle redirects
- Understand page context

**Example Usage:**
```python
result = env.step(action)
if result.observation.url == "https://example.com/dashboard":
    print("Successfully logged in!")
```

#### 4. Task Goals

**Goal String** (`goal`)
- Natural language task description
- What the agent should accomplish
- Success criteria (implicit)

**Example Goals:**
- "Click the 'Submit' button"
- "Fill out the login form and submit"
- "Find the cheapest laptop and add to cart"
- "Create a merge request for issue #42"

### Choosing Observation Modality

| Modality | Pros | Cons | Best For |
|----------|------|------|----------|
| **AXTree** | Compact, semantic, LLM-friendly | Limited visual info | Text-based tasks, structure understanding |
| **HTML** | Semantic tags, attributes | Can be large | Understanding relationships |
| **Screenshot** | Complete visual info | Large, needs VLM | Visual tasks, image understanding |
| **URL** | Simple, reliable | Limited info | Navigation tracking |

**Recommendation:**
- **Start with**: AXTree only (lightweight, fast)
- **Add screenshots**: For vision-dependent tasks
- **Add HTML**: For complex semantic understanding

---

## Complete Implementation Examples

### Example 1: Simple Click Task (MiniWoB)

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction

def run_click_test():
    """Simple MiniWoB click-test task."""

    # Create environment
    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        env_vars={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": "click-test",
        }
    )

    # Reset
    result = env.reset()
    print(f"Goal: {result.observation.goal}")
    print(f"URL: {result.observation.url}")
    print(f"\nPage content:\n{result.observation.text}")

    # Parse goal to find button text
    # Goal format: "Click on the button labeled 'Submit'."
    import re
    match = re.search(r"labeled '([^']+)'", result.observation.goal)
    if match:
        button_text = match.group(1)
        print(f"\nTarget button: {button_text}")

        # Click the button
        action = BrowserGymAction(action_str=f"click('{button_text}')")
        result = env.step(action)

        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")
        print(f"Success: {result.reward > 0}")
    else:
        print("Could not parse button text from goal")

    env.close()

if __name__ == "__main__":
    run_click_test()
```

### Example 2: Form Filling (MiniWoB)

```python
def run_login_user():
    """MiniWoB login-user task."""

    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        env_vars={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": "login-user",
        }
    )

    result = env.reset()
    print(f"Goal: {result.observation.goal}")

    # Goal typically includes username and password
    # Example: "Login with username 'admin' and password 'pass123'"
    import re
    username_match = re.search(r"username '([^']+)'", result.observation.goal)
    password_match = re.search(r"password '([^']+)'", result.observation.goal)

    if username_match and password_match:
        username = username_match.group(1)
        password = password_match.group(1)

        print(f"Username: {username}")
        print(f"Password: {password}")

        # Fill form
        result = env.step(BrowserGymAction(
            action_str=f"fill('username', '{username}')"
        ))
        print(f"Filled username, error: {result.observation.last_action_error}")

        result = env.step(BrowserGymAction(
            action_str=f"fill('password', '{password}')"
        ))
        print(f"Filled password, error: {result.observation.last_action_error}")

        # Submit
        result = env.step(BrowserGymAction(
            action_str="click('Login')"
        ))

        print(f"\nFinal result:")
        print(f"  Reward: {result.reward}")
        print(f"  Done: {result.done}")
        print(f"  Success: {result.reward > 0}")

    env.close()
```

### Example 3: Multi-Step Task with LLM Agent

```python
from envs.browsergym_env import BrowserGymEnv, BrowserGymAction
import anthropic

def run_llm_agent(task_name: str, max_steps: int = 10):
    """Run LLM-based agent on BrowserGym task."""

    # Setup environment
    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        env_vars={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": task_name,
        }
    )

    # Setup LLM client
    client = anthropic.Anthropic()

    # Reset
    result = env.reset()
    print(f"Goal: {result.observation.goal}\n")

    # Interaction loop
    for step in range(max_steps):
        # Create prompt for LLM
        prompt = f"""You are a web automation agent. Your goal is:
{result.observation.goal}

Current page URL: {result.observation.url}

Page content (accessibility tree):
{result.observation.text}

Based on the current page and your goal, what action should you take next?

Respond with a single action in this format:
ACTION: <action_string>

Available actions:
- click('element_description')
- fill('element', 'text')
- goto('url')
- press('key')
- select_option('dropdown', 'option')
- scroll('down' or 'up')

Your response:"""

        # Get LLM response
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text
        print(f"Step {step + 1}:")
        print(f"  LLM response: {response_text}")

        # Parse action from response
        import re
        action_match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", response_text)
        if action_match:
            action_str = action_match.group(1).strip()
            print(f"  Parsed action: {action_str}")

            # Execute action
            action = BrowserGymAction(action_str=action_str)
            result = env.step(action)

            print(f"  Result: reward={result.reward}, done={result.done}, error={result.observation.last_action_error}")

            if result.observation.last_action_error:
                print(f"  Error: {result.observation.error}")

            if result.done:
                print(f"\nTask completed!")
                print(f"  Final reward: {result.reward}")
                print(f"  Success: {result.reward > 0}")
                break
        else:
            print("  Could not parse action from LLM response")
            break

        print()

    env.close()

if __name__ == "__main__":
    run_llm_agent("click-dialog")
```

### Example 4: Curriculum Learning Across Multiple Tasks

```python
def run_curriculum_training():
    """Train agent on progressively harder tasks."""

    curriculum = [
        # Level 1: Basic clicks
        ("click-test", 5),
        ("click-button", 5),
        ("click-dialog", 5),

        # Level 2: Simple forms
        ("click-checkboxes", 10),
        ("click-radio", 10),
        ("enter-text", 10),

        # Level 3: Complex forms
        ("login-user", 15),
        ("email-inbox", 20),
        ("search-engine", 20),

        # Level 4: Multi-step
        ("book-flight", 30),
        ("social-media", 30),
    ]

    for task_name, max_steps in curriculum:
        print(f"\n{'='*60}")
        print(f"Training on: {task_name}")
        print(f"{'='*60}\n")

        # Run multiple episodes
        successes = 0
        episodes = 10

        for episode in range(episodes):
            print(f"Episode {episode + 1}/{episodes}")
            success = run_single_episode(task_name, max_steps)
            if success:
                successes += 1

        success_rate = successes / episodes
        print(f"\nTask {task_name}: {success_rate * 100:.1f}% success rate")

        # Move to next level if success rate > 70%
        if success_rate < 0.7:
            print(f"Stuck on {task_name}, continuing training...")
            # In real training, you'd continue on this task
        else:
            print(f"Passed {task_name}, moving to next level!")

def run_single_episode(task_name: str, max_steps: int) -> bool:
    """Run single episode, return success."""
    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        env_vars={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": task_name,
        }
    )

    result = env.reset()

    for step in range(max_steps):
        # Your agent logic here
        action = your_agent.select_action(result.observation)
        result = env.step(action)

        if result.done:
            break

    success = result.reward > 0
    env.close()
    return success
```

---

## Agent Architectures

### 1. Template-Based Agent (Baseline)

```python
class TemplateAgent:
    """Simple rule-based agent using templates."""

    def __init__(self):
        self.templates = {
            "click": r"Click.*?['\"]([^'\"]+)['\"]",
            "fill": r"(?:Enter|Type|Fill).*?['\"]([^'\"]+)['\"].*?['\"]([^'\"]+)['\"]",
            "select": r"Select.*?['\"]([^'\"]+)['\"]",
        }

    def select_action(self, obs: BrowserGymObservation) -> BrowserGymAction:
        goal = obs.goal.lower()

        # Try to match templates
        for action_type, pattern in self.templates.items():
            match = re.search(pattern, obs.goal, re.IGNORECASE)
            if match:
                if action_type == "click":
                    element = match.group(1)
                    return BrowserGymAction(action_str=f"click('{element}')")
                elif action_type == "fill":
                    field = match.group(1)
                    value = match.group(2)
                    return BrowserGymAction(action_str=f"fill('{field}', '{value}')")
                elif action_type == "select":
                    option = match.group(1)
                    return BrowserGymAction(action_str=f"select_option('dropdown', '{option}')")

        # Default: click first button
        return BrowserGymAction(action_str="click('button')")
```

### 2. LLM-Based Agent (Standard)

```python
class LLMAgent:
    """LLM-based agent using Claude or GPT-4."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.history = []

    def select_action(self, obs: BrowserGymObservation) -> BrowserGymAction:
        # Build prompt
        prompt = f"""You are a web automation expert.

Goal: {obs.goal}

Current page:
URL: {obs.url}

Page structure (accessibility tree):
{obs.text[:3000]}  # Truncate to fit context

Previous actions:
{self._format_history()}

What action should you take next to achieve the goal?
Respond with: ACTION: <action_string>

Examples:
- ACTION: click('Submit button')
- ACTION: fill('email', 'test@example.com')
- ACTION: goto('https://example.com')
"""

        # Get LLM response
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        response = message.content[0].text

        # Parse action
        match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", response)
        if match:
            action_str = match.group(1).strip()
        else:
            # Fallback
            action_str = "wait(1000)"

        action = BrowserGymAction(action_str=action_str)

        # Update history
        self.history.append((obs, action))

        return action

    def _format_history(self) -> str:
        if not self.history:
            return "None yet."

        lines = []
        for i, (obs, act) in enumerate(self.history[-5:]):  # Last 5 actions
            lines.append(f"{i+1}. {act.action_str}")
        return "\n".join(lines)
```

### 3. Multimodal Agent (Advanced)

```python
class MultimodalAgent:
    """Vision-language agent using screenshots."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-3-5-sonnet-20241022"

    def select_action(self, obs: BrowserGymObservation) -> BrowserGymAction:
        # Encode screenshot
        if obs.screenshot is not None:
            screenshot_base64 = self._encode_screenshot(obs.screenshot)
        else:
            screenshot_base64 = None

        # Build multimodal prompt
        content = [
            {
                "type": "text",
                "text": f"""You are a web automation agent.

Goal: {obs.goal}

Current URL: {obs.url}

See the screenshot of the current page below. What action should you take?

Respond with: ACTION: <action_string>"""
            }
        ]

        if screenshot_base64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_base64,
                }
            })

        # Get VLM response
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{"role": "user", "content": content}]
        )

        response = message.content[0].text

        # Parse action
        match = re.search(r"ACTION:\s*(.+?)(?:\n|$)", response)
        if match:
            action_str = match.group(1).strip()
        else:
            action_str = "wait(1000)"

        return BrowserGymAction(action_str=action_str)

    def _encode_screenshot(self, screenshot) -> str:
        """Convert numpy array to base64 PNG."""
        import io
        import base64
        from PIL import Image
        import numpy as np

        # Convert to PIL Image
        img = Image.fromarray(np.array(screenshot).astype('uint8'), 'RGB')

        # Encode as PNG
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        # Base64 encode
        return base64.b64encode(buffer.read()).decode('utf-8')
```

---

## Training Strategies

### Strategy 1: Behavior Cloning (Supervised Learning)

```python
def collect_demonstrations():
    """Collect expert demonstrations for behavior cloning."""

    dataset = []

    # Collect demos on MiniWoB tasks
    tasks = ["click-test", "click-button", "login-user"]

    for task in tasks:
        for episode in range(100):  # 100 demos per task
            env = BrowserGymEnv.from_docker_image(...)
            result = env.reset()

            done = False
            trajectory = []

            while not done:
                # Expert provides action (human or scripted)
                action = expert_policy(result.observation)

                # Record (state, action) pair
                trajectory.append({
                    "observation": result.observation,
                    "action": action.action_str,
                })

                result = env.step(action)
                done = result.done

            # Only keep successful trajectories
            if result.reward > 0:
                dataset.extend(trajectory)

            env.close()

    # Save dataset
    import json
    with open("browsergym_demonstrations.json", "w") as f:
        json.dump(dataset, f)

    return dataset

def train_bc_agent(dataset):
    """Train agent via behavior cloning."""

    # Fine-tune LLM on demonstrations
    # (Pseudo-code - actual implementation depends on your setup)

    training_data = []
    for example in dataset:
        obs = example["observation"]
        action = example["action"]

        prompt = f"Goal: {obs.goal}\nPage: {obs.text}\n\nAction:"
        completion = action

        training_data.append({"prompt": prompt, "completion": completion})

    # Fine-tune (e.g., via OpenAI API, Anthropic, or local)
    # fine_tuning_job = openai.FineTuning.create(...)

    return trained_model
```

### Strategy 2: Reinforcement Learning (RL)

```python
def train_rl_agent():
    """Train agent with RL (e.g., PPO, DQN)."""

    import torch
    from stable_baselines3 import PPO

    # Create vectorized environment
    def make_env(task_name):
        def _init():
            env = BrowserGymEnv.from_docker_image(
                "browsergym-env:latest",
                env_vars={
                    "BROWSERGYM_BENCHMARK": "miniwob",
                    "BROWSERGYM_TASK_NAME": task_name,
                }
            )
            return env
        return _init

    # Note: BrowserGym obs is complex (text + images)
    # You'll need a custom wrapper to convert to RL-friendly format

    # Train
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=1000000)

    return model
```

### Strategy 3: Curriculum Learning

**Phase 1: Simple Tasks (1-2 steps)**
- click-test
- click-button
- focus-text

**Phase 2: Forms (2-5 steps)**
- enter-text
- login-user
- click-checkboxes

**Phase 3: Search (5-10 steps)**
- search-engine
- email-inbox
- click-tab

**Phase 4: Complex (10+ steps)**
- book-flight
- social-media
- multi-layouts

### Strategy 4: Self-Improvement

```python
def self_improvement_loop():
    """Agent improves by learning from its own experience."""

    agent = LLMAgent()
    success_buffer = []
    failure_buffer = []

    for iteration in range(100):
        print(f"Iteration {iteration}")

        # Collect experience
        for episode in range(10):
            result, trajectory = run_episode(agent)

            if result.reward > 0:
                success_buffer.append(trajectory)
            else:
                failure_buffer.append(trajectory)

        # Learn from successes
        if len(success_buffer) >= 50:
            # Fine-tune on successful trajectories
            train_on_data(success_buffer)
            success_buffer = []

        # Analyze failures
        if len(failure_buffer) >= 50:
            # Use LLM to analyze what went wrong
            analyze_failures(failure_buffer)
            failure_buffer = []
```

---

## Advanced Topics

### Error Handling

```python
def robust_action_execution(env, action, max_retries=3):
    """Execute action with retries on error."""

    for attempt in range(max_retries):
        result = env.step(action)

        if not result.observation.last_action_error:
            return result

        print(f"Action failed (attempt {attempt + 1}): {result.observation.error}")

        # Try alternative action
        if "element not found" in result.observation.error.lower():
            # Wait and retry
            env.step(BrowserGymAction(action_str="wait(1000)"))
        elif "invalid selector" in result.observation.error.lower():
            # Modify action
            action = simplify_action(action)
        else:
            # Give up
            break

    return result
```

### State Abstraction

```python
def extract_key_elements(obs: BrowserGymObservation) -> dict:
    """Extract structured information from observation."""

    # Parse accessibility tree
    elements = []
    for line in obs.axtree_txt.split("\n"):
        # Extract element type, text, attributes
        match = re.match(r"(\s*)(\w+)\s+'([^']*)'(.*)", line)
        if match:
            indent, elem_type, text, attrs = match.groups()
            elements.append({
                "type": elem_type,
                "text": text,
                "attrs": attrs,
                "level": len(indent) // 2,
            })

    # Find interactive elements
    interactive = [e for e in elements if e["type"] in ["button", "link", "textbox"]]

    return {
        "url": obs.url,
        "goal": obs.goal,
        "interactive_elements": interactive,
        "num_elements": len(elements),
    }
```

### Memory-Augmented Agents

```python
class MemoryAgent(LLMAgent):
    """Agent with episodic memory."""

    def __init__(self):
        super().__init__()
        self.memory_buffer = []  # Store (obs, action, result) tuples

    def select_action(self, obs: BrowserGymObservation) -> BrowserGymAction:
        # Retrieve relevant memories
        relevant_memories = self.retrieve_memories(obs)

        # Include memories in prompt
        memory_str = self._format_memories(relevant_memories)

        prompt = f"""You are a web automation agent with memory.

Current goal: {obs.goal}

Relevant past experiences:
{memory_str}

Current page:
{obs.text[:2000]}

What action should you take?
ACTION: <action_string>"""

        # ... rest of LLM call ...

        return action

    def retrieve_memories(self, obs: BrowserGymObservation, top_k=5):
        """Retrieve top-k relevant memories."""
        # Simple: retrieve memories with similar goals
        scored_memories = []
        for memory in self.memory_buffer:
            similarity = compute_similarity(obs.goal, memory[0].goal)
            scored_memories.append((similarity, memory))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scored_memories[:top_k]]

    def _format_memories(self, memories):
        lines = []
        for i, (obs, action, result) in enumerate(memories):
            lines.append(f"{i+1}. Goal: {obs.goal}")
            lines.append(f"   Action: {action.action_str}")
            lines.append(f"   Success: {result.reward > 0}")
        return "\n".join(lines)
```

---

## Troubleshooting

### Common Issues

**1. Action Execution Fails**
- **Symptom**: `last_action_error=True`, error message in observation
- **Causes**: Element not found, invalid selector, timing issues
- **Solutions**:
  - Add `wait()` actions before clicking
  - Use more specific element descriptions
  - Check accessibility tree for actual element text

**2. Slow Performance**
- **Symptom**: Each step takes >5 seconds
- **Causes**: Large observations, screenshot processing, network latency
- **Solutions**:
  - Disable screenshots if not needed
  - Truncate accessibility tree
  - Use faster model/smaller LLM
  - Run locally instead of Docker

**3. Task Not Completing**
- **Symptom**: Agent gets stuck, `done=False` after many steps
- **Causes**: Wrong actions, goal misunderstanding, page not loading
- **Solutions**:
  - Print observation after each step
  - Check URL changes
  - Add timeout/max steps
  - Implement fallback actions

**4. Low Success Rate**
- **Symptom**: Agent succeeds <50% of time
- **Causes**: Insufficient training, poor prompting, task too hard
- **Solutions**:
  - Start with simpler tasks (curriculum)
  - Improve prompts (add examples)
  - Collect demonstrations
  - Fine-tune model

### Debugging Tips

```python
def debug_episode(task_name: str):
    """Run episode with detailed debugging."""

    env = BrowserGymEnv.from_docker_image(
        "browsergym-env:latest",
        env_vars={
            "BROWSERGYM_BENCHMARK": "miniwob",
            "BROWSERGYM_TASK_NAME": task_name,
        }
    )

    result = env.reset()

    print("="*60)
    print(f"Task: {task_name}")
    print(f"Goal: {result.observation.goal}")
    print("="*60)

    for step in range(10):
        print(f"\n--- Step {step + 1} ---")
        print(f"URL: {result.observation.url}")
        print(f"Text (first 500 chars):\n{result.observation.text[:500]}")

        # Your agent
        action = agent.select_action(result.observation)
        print(f"\nAction: {action.action_str}")

        result = env.step(action)

        print(f"Reward: {result.reward}")
        print(f"Done: {result.done}")
        print(f"Error: {result.observation.last_action_error}")
        if result.observation.last_action_error:
            print(f"Error message: {result.observation.error}")

        if result.done:
            print(f"\n{'SUCCESS' if result.reward > 0 else 'FAILURE'}")
            break

    env.close()
```

---

## Summary

**BrowserGym is the most powerful environment in OpenEnv** for training web agents. This guide covered:

✅ Setup and configuration (MiniWoB, WebArena)
✅ Action space (all action types and examples)
✅ Observation space (text, visual, multimodal)
✅ Complete implementation examples
✅ Agent architectures (template, LLM, multimodal)
✅ Training strategies (BC, RL, curriculum, self-improvement)
✅ Advanced topics (memory, state abstraction)
✅ Troubleshooting and debugging

**Next Steps:**
1. Start with MiniWoB simple tasks (click-test, enter-text)
2. Build template agent → LLM agent → multimodal agent
3. Use curriculum learning for progressive difficulty
4. Evaluate on WebArena for real-world performance

**Key Insight**: Start simple (text-only, MiniWoB), then add complexity (screenshots, WebArena) as your agent improves. The training-to-evaluation pipeline (MiniWoB → WebArena) is the key to building robust web agents.
