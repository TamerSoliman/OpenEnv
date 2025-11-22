# Git Environment: Complete Implementation Guide

## Table of Contents
- [Introduction](#introduction)
- [Architecture Overview](#architecture-overview)
- [Environment Setup](#environment-setup)
- [Action Space Complete Reference](#action-space-complete-reference)
- [Complete Implementation Examples](#complete-implementation-examples)
- [Training Scenarios](#training-scenarios)
- [Advanced Workflows](#advanced-workflows)
- [Best Practices](#best-practices)

---

## Introduction

The Git Environment enables training LLM agents on version control operations - essential for building AI coding assistants and DevOps automation tools.

**Why Git Environment?**
- **Fast reset**: <1s reset via git operations (vs minutes for Docker restart)
- **Task-based isolation**: Each environment has isolated workspace
- **Reproducible states**: Reset to specific commits for consistent training
- **Real Git operations**: Full command support, no simulation

**Use Cases:**
- AI coding assistants (GitHub Copilot-style)
- Automated code review agents
- CI/CD automation
- DevOps workflow automation
- Git education tools

---

## Architecture Overview

### Components

```
┌──────────────────────────────────────────────────────────────┐
│                    SHARED GITEA SERVER                        │
│                    (External, port 3000)                      │
│                                                               │
│  • Pre-migrated repositories                                  │
│  • User authentication                                        │
│  • Git HTTP/SSH endpoints                                     │
│  • Web UI (optional)                                          │
└──────────────────┬───────────────────────────────────────────┘
                   │ HTTP API (REST + Git protocol)
          ┌────────┼────────┬────────┬────────┐
          │        │        │        │        │
     ┌────▼───┐ ┌─▼────┐ ┌─▼────┐ ┌─▼────┐ ┌─▼────┐
     │ Env 1  │ │ Env 2│ │ Env 3│ │ Env 4│ │ Env 5│
     │ Task A │ │ Task B│ │ Task A│ │ Task C│ │ Task A│
     │@commit1│ │@commit2│@commit1│@commit3│@commit1│
     └────────┘ └──────┘ └──────┘ └──────┘ └──────┘
     Isolated workspaces (/workspace/<repo>)
```

**Key Design Principles:**
1. **Shared Gitea**: One server, many environments (efficient resource use)
2. **Isolated workspaces**: Each env has separate `/workspace` directory
3. **Task-based config**: Pre-configure repo states for fast reset
4. **Stateless environments**: Can spin up/down without losing repository data

---

## Environment Setup

### Gitea Server Setup

```bash
# Start Gitea server (do this ONCE, shared across all environments)
docker run -d \
  --name gitea \
  -p 3000:3000 \
  -p 222:22 \
  -v gitea-data:/data \
  gitea/gitea:latest

# Wait for Gitea to be ready
curl -f http://localhost:3000/api/v1/version

# Create admin user
curl -X POST http://localhost:3000/api/v1/admin/users \
  -H "Content-Type: application/json" \
  -d '{
    "username": "gitea_admin",
    "password": "r8sA8CPHD9!bt6d",
    "email": "admin@example.com"
  }'

# Migrate repositories (one-time setup)
# Option 1: Via UI at http://localhost:3000
# Option 2: Via API
curl -X POST http://localhost:3000/api/v1/repos/migrate \
  -u gitea_admin:r8sA8CPHD9!bt6d \
  -H "Content-Type: application/json" \
  -d '{
    "clone_addr": "https://github.com/meta-ai/OpenEnv.git",
    "repo_name": "OpenEnv",
    "private": false
  }'
```

### Environment Client Setup

```python
from envs.git_env import GitEnv, GitAction

# Basic setup (automatic repository access)
env = GitEnv.from_docker_image(
    "git-env:latest",
    env_vars={
        "GITEA_URL": "http://host.docker.internal:3000",
        "GITEA_USERNAME": "gitea_admin",
        "GITEA_PASSWORD": "r8sA8CPHD9!bt6d",
    }
)
```

### Task-Based Setup (Advanced)

```python
from envs.git_env.server.git_task_environment import GitTaskEnvironment

# Configure tasks with specific repo states
env = GitTaskEnvironment(
    gitea_url="http://localhost:3000",
    username="gitea_admin",
    password="r8sA8CPHD9!bt6d",
    workspace_dir="/workspace",
    task_repos={
        # Task ID: (repo_name, commit_hash)
        "fix_bug_123": ("OpenEnv", "abc123"),
        "add_feature_auth": ("OpenEnv", "def456"),
        "refactor_models": ("OpenEnv", "789ghi"),
    }
)

# Fast reset to specific task (<1s!)
obs = env.reset(task_id="fix_bug_123")
```

---

## Action Space Complete Reference

### Action Structure

```python
@dataclass
class GitAction(Action):
    action_type: str                  # Type of operation
    repo_name: str = ""               # Repository name
    target_dir: Optional[str] = None  # Target directory (for clone)
    command: str = ""                 # Git command (for execute)
    working_dir: str = ""             # Working directory (for execute)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Action Types

#### 1. `list_repos` - List Available Repositories

**Purpose**: Get list of all repositories on Gitea server

**Example:**
```python
action = GitAction(action_type="list_repos")
result = env.step(action)

# Result
print(result.observation.repos)
# [
#   {'name': 'OpenEnv', 'clone_url': 'http://...', 'description': '...'},
#   {'name': 'my-project', 'clone_url': 'http://...', 'description': '...'},
# ]
```

**Observation Fields:**
- `repos`: List[Dict] with `name`, `clone_url`, `description`
- `success`: bool
- `message`: str (e.g., "Found 5 repositories")

#### 2. `clone_repo` - Clone Repository to Workspace

**Purpose**: Clone repository from Gitea to local workspace

**Example:**
```python
# Basic clone
action = GitAction(
    action_type="clone_repo",
    repo_name="OpenEnv"
)
result = env.step(action)

# Clone to specific directory
action = GitAction(
    action_type="clone_repo",
    repo_name="OpenEnv",
    target_dir="my-project"
)
result = env.step(action)
```

**Observation Fields:**
- `success`: bool
- `message`: str (e.g., "Successfully cloned OpenEnv")
- `output`: str (clone path and commit info)

**Notes:**
- Clones to `/workspace/<repo_name>` by default
- If `target_dir` specified, clones to `/workspace/<target_dir>`
- For task-based envs, uses pre-configured commit

#### 3. `execute_git_command` - Execute Git Command

**Purpose**: Run any git command in workspace

**Example:**
```python
# Git status
action = GitAction(
    action_type="execute_git_command",
    command="status",
    working_dir="OpenEnv"
)
result = env.step(action)

# Git log
action = GitAction(
    action_type="execute_git_command",
    command="log --oneline -10",
    working_dir="OpenEnv"
)
result = env.step(action)

# Create branch
action = GitAction(
    action_type="execute_git_command",
    command="checkout -b feature-branch",
    working_dir="OpenEnv"
)
result = env.step(action)
```

**Observation Fields:**
- `success`: bool (exit_code == 0)
- `message`: str (success/failure message)
- `output`: str (stdout from command)
- `error`: str (stderr from command)

### Complete Git Command Reference

#### Status & Info
```python
# Status
GitAction(action_type="execute_git_command", command="status", working_dir="repo")

# Log
GitAction(action_type="execute_git_command", command="log --oneline -20", working_dir="repo")

# Show commit
GitAction(action_type="execute_git_command", command="show HEAD", working_dir="repo")

# Diff
GitAction(action_type="execute_git_command", command="diff", working_dir="repo")
GitAction(action_type="execute_git_command", command="diff HEAD~1", working_dir="repo")

# Branch list
GitAction(action_type="execute_git_command", command="branch -a", working_dir="repo")

# Current branch
GitAction(action_type="execute_git_command", command="rev-parse --abbrev-ref HEAD", working_dir="repo")
```

#### Branching
```python
# Create branch
GitAction(action_type="execute_git_command", command="checkout -b feature-branch", working_dir="repo")

# Switch branch
GitAction(action_type="execute_git_command", command="checkout main", working_dir="repo")

# Delete branch
GitAction(action_type="execute_git_command", command="branch -d feature-branch", working_dir="repo")

# Merge branch
GitAction(action_type="execute_git_command", command="merge feature-branch", working_dir="repo")
```

#### Staging & Committing
```python
# Stage all files
GitAction(action_type="execute_git_command", command="add .", working_dir="repo")

# Stage specific file
GitAction(action_type="execute_git_command", command="add path/to/file.py", working_dir="repo")

# Commit
GitAction(action_type="execute_git_command", command='commit -m "Fix bug"', working_dir="repo")

# Amend commit
GitAction(action_type="execute_git_command", command='commit --amend -m "Updated message"', working_dir="repo")

# Reset staging
GitAction(action_type="execute_git_command", command="reset HEAD file.py", working_dir="repo")
```

#### Remote Operations
```python
# Fetch
GitAction(action_type="execute_git_command", command="fetch origin", working_dir="repo")

# Pull
GitAction(action_type="execute_git_command", command="pull origin main", working_dir="repo")

# Push
GitAction(action_type="execute_git_command", command="push origin feature-branch", working_dir="repo")

# Remote info
GitAction(action_type="execute_git_command", command="remote -v", working_dir="repo")
```

#### Advanced
```python
# Rebase
GitAction(action_type="execute_git_command", command="rebase main", working_dir="repo")

# Cherry-pick
GitAction(action_type="execute_git_command", command="cherry-pick abc123", working_dir="repo")

# Stash
GitAction(action_type="execute_git_command", command="stash", working_dir="repo")
GitAction(action_type="execute_git_command", command="stash pop", working_dir="repo")

# Blame
GitAction(action_type="execute_git_command", command="blame path/to/file.py", working_dir="repo")

# Reflog
GitAction(action_type="execute_git_command", command="reflog", working_dir="repo")
```

---

## Complete Implementation Examples

### Example 1: Basic Git Operations

```python
from envs.git_env import GitEnv, GitAction

def basic_git_workflow():
    """Demonstrate basic Git operations."""

    # Setup
    env = GitEnv.from_docker_image("git-env:latest")

    # Reset
    result = env.reset()
    print(f"Environment ready: {result.observation.message}")

    # 1. List repositories
    print("\n1. Listing repositories...")
    result = env.step(GitAction(action_type="list_repos"))
    print(f"Found {len(result.observation.repos)} repositories:")
    for repo in result.observation.repos:
        print(f"  - {repo['name']}: {repo['clone_url']}")

    # 2. Clone repository
    print("\n2. Cloning OpenEnv...")
    result = env.step(GitAction(
        action_type="clone_repo",
        repo_name="OpenEnv"
    ))
    print(f"Clone result: {result.observation.message}")

    # 3. Check status
    print("\n3. Checking status...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="status",
        working_dir="OpenEnv"
    ))
    print(f"Status:\n{result.observation.output}")

    # 4. View log
    print("\n4. Viewing recent commits...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="log --oneline -5",
        working_dir="OpenEnv"
    ))
    print(f"Recent commits:\n{result.observation.output}")

    # 5. Check branches
    print("\n5. Listing branches...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="branch -a",
        working_dir="OpenEnv"
    ))
    print(f"Branches:\n{result.observation.output}")

    env.close()

if __name__ == "__main__":
    basic_git_workflow()
```

### Example 2: Feature Branch Workflow

```python
def feature_branch_workflow():
    """Implement a complete feature branch workflow."""

    env = GitEnv.from_docker_image("git-env:latest")
    result = env.reset()

    # Clone repository
    print("Cloning repository...")
    result = env.step(GitAction(
        action_type="clone_repo",
        repo_name="OpenEnv"
    ))

    # Create feature branch
    print("\nCreating feature branch...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="checkout -b feature/add-docs",
        working_dir="OpenEnv"
    ))
    print(f"Branch created: {result.observation.success}")

    # Simulate making changes (in real scenario, agent would edit files)
    # For this example, we'll just create a dummy commit
    print("\nSimulating file changes...")

    # Stage changes
    print("Staging changes...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="add .",
        working_dir="OpenEnv"
    ))

    # Commit changes
    print("Committing changes...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command='commit -m "Add documentation for Git environment" --allow-empty',
        working_dir="OpenEnv"
    ))
    print(f"Commit result: {result.observation.message}")
    print(f"Commit output:\n{result.observation.output}")

    # View commit
    print("\nViewing commit...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="show HEAD --stat",
        working_dir="OpenEnv"
    ))
    print(f"Commit details:\n{result.observation.output}")

    # Switch back to main
    print("\nSwitching to main branch...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="checkout main",
        working_dir="OpenEnv"
    ))

    # Merge feature branch
    print("Merging feature branch...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="merge feature/add-docs",
        working_dir="OpenEnv"
    ))
    print(f"Merge result: {result.observation.message}")

    # Clean up branch
    print("\nDeleting feature branch...")
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="branch -d feature/add-docs",
        working_dir="OpenEnv"
    ))

    env.close()
```

### Example 3: LLM-Based Git Agent

```python
import anthropic

class GitAgent:
    """LLM-based agent for Git operations."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-3-5-sonnet-20241022"

    def execute_task(self, task_description: str, repo_name: str):
        """Execute a Git task using LLM reasoning."""

        env = GitEnv.from_docker_image("git-env:latest")
        result = env.reset()

        # Clone repository
        result = env.step(GitAction(
            action_type="clone_repo",
            repo_name=repo_name
        ))

        # Get repository state
        result = env.step(GitAction(
            action_type="execute_git_command",
            command="status",
            working_dir=repo_name
        ))
        status = result.observation.output

        result = env.step(GitAction(
            action_type="execute_git_command",
            command="log --oneline -10",
            working_dir=repo_name
        ))
        log = result.observation.output

        result = env.step(GitAction(
            action_type="execute_git_command",
            command="branch -a",
            working_dir=repo_name
        ))
        branches = result.observation.output

        # LLM decides what to do
        prompt = f"""You are a Git expert. Execute this task:

Task: {task_description}

Current repository state:
Repository: {repo_name}

Status:
{status}

Recent commits:
{log}

Branches:
{branches}

What Git commands should you execute to complete this task?

Respond with a list of commands, one per line, in this format:
COMMAND: <git command>

Example:
COMMAND: checkout -b feature-branch
COMMAND: add .
COMMAND: commit -m "message"

Your response:"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        response = message.content[0].text
        print(f"LLM plan:\n{response}\n")

        # Parse and execute commands
        import re
        commands = re.findall(r"COMMAND:\s*(.+)", response)

        for i, command in enumerate(commands):
            print(f"Executing ({i+1}/{len(commands)}): {command}")

            result = env.step(GitAction(
                action_type="execute_git_command",
                command=command,
                working_dir=repo_name
            ))

            if result.observation.success:
                print(f"  ✓ Success: {result.observation.message}")
                if result.observation.output:
                    print(f"  Output: {result.observation.output[:200]}")
            else:
                print(f"  ✗ Failed: {result.observation.error}")
                break

        env.close()

# Usage
agent = GitAgent()
agent.execute_task(
    task_description="Create a feature branch called 'add-tests', switch to it, and show the current branch",
    repo_name="OpenEnv"
)
```

---

## Training Scenarios

### Scenario 1: Basic Git Literacy

**Learning Objectives:**
- Understand git status, log, diff
- Navigate branches
- View commit history

**Tasks:**
```python
tasks = [
    ("Show current status", "Check the git status"),
    ("List branches", "List all branches"),
    ("View recent commits", "Show last 5 commits"),
    ("Show current branch", "Display the current branch name"),
    ("View specific commit", "Show details of commit abc123"),
]
```

**Evaluation:**
- Command correctness (did they use the right command?)
- Output parsing (can they extract information?)
- Efficiency (fewest commands to complete task)

### Scenario 2: Feature Development

**Learning Objectives:**
- Create and switch branches
- Stage and commit changes
- Merge branches

**Tasks:**
```python
tasks = [
    ("Create feature branch", "Create a branch called 'feature/new-feature'"),
    ("Switch to branch", "Switch to the feature branch"),
    ("Make changes", "Stage all changes and commit with message 'Add feature'"),
    ("Merge to main", "Switch to main and merge the feature branch"),
    ("Clean up", "Delete the feature branch"),
]
```

**Evaluation:**
- Workflow correctness
- Commit message quality
- Branch hygiene

### Scenario 3: Code Review & History

**Learning Objectives:**
- Inspect changes
- Understand history
- Find specific commits

**Tasks:**
```python
tasks = [
    ("Find bug introduction", "Find when bug.py was last modified"),
    ("View file history", "Show commit history for specific file"),
    ("Compare branches", "Show differences between main and feature branch"),
    ("Blame analysis", "Show who last modified each line of file"),
]
```

**Evaluation:**
- Investigation strategy
- Command sophistication
- Information extraction

### Scenario 4: Conflict Resolution

**Learning Objectives:**
- Detect conflicts
- Understand conflict markers
- Resolve conflicts

**Tasks:**
```python
tasks = [
    ("Detect conflict", "Try to merge branch-with-conflict and detect conflict"),
    ("Identify conflicted files", "List files with merge conflicts"),
    ("View conflict markers", "Show the conflict in conflicted file"),
    ("Abort merge", "Abort the merge to return to safe state"),
]
```

**Evaluation:**
- Conflict detection
- Understanding of merge process
- Recovery actions

---

## Advanced Workflows

### Workflow 1: Cherry-Pick Specific Commits

```python
def cherry_pick_workflow():
    """Cherry-pick commits from one branch to another."""

    env = GitEnv.from_docker_image("git-env:latest")
    result = env.reset()

    # Clone
    env.step(GitAction(action_type="clone_repo", repo_name="OpenEnv"))

    # Create target branch
    env.step(GitAction(
        action_type="execute_git_command",
        command="checkout -b backport-branch",
        working_dir="OpenEnv"
    ))

    # Cherry-pick specific commit
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="cherry-pick abc123",
        working_dir="OpenEnv"
    ))

    if result.observation.success:
        print("Cherry-pick successful!")
    else:
        print(f"Cherry-pick failed: {result.observation.error}")

    env.close()
```

### Workflow 2: Interactive Rebase Simulation

```python
def rebase_workflow():
    """Rebase feature branch onto updated main."""

    env = GitEnv.from_docker_image("git-env:latest")
    result = env.reset()

    # Clone and switch to feature branch
    env.step(GitAction(action_type="clone_repo", repo_name="OpenEnv"))
    env.step(GitAction(
        action_type="execute_git_command",
        command="checkout feature-branch",
        working_dir="OpenEnv"
    ))

    # Fetch latest main
    env.step(GitAction(
        action_type="execute_git_command",
        command="fetch origin main",
        working_dir="OpenEnv"
    ))

    # Rebase onto main
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="rebase origin/main",
        working_dir="OpenEnv"
    ))

    if result.observation.success:
        print("Rebase successful!")
    else:
        print(f"Rebase conflict: {result.observation.error}")
        # Handle conflict...

    env.close()
```

### Workflow 3: Bisect to Find Bug

```python
def bisect_workflow():
    """Use git bisect to find bug introduction."""

    env = GitEnv.from_docker_image("git-env:latest")
    result = env.reset()

    env.step(GitAction(action_type="clone_repo", repo_name="OpenEnv"))

    # Start bisect
    env.step(GitAction(
        action_type="execute_git_command",
        command="bisect start",
        working_dir="OpenEnv"
    ))

    # Mark bad commit (current)
    env.step(GitAction(
        action_type="execute_git_command",
        command="bisect bad",
        working_dir="OpenEnv"
    ))

    # Mark good commit (known working)
    env.step(GitAction(
        action_type="execute_git_command",
        command="bisect good abc123",
        working_dir="OpenEnv"
    ))

    # Bisect will checkout middle commit
    # Test code, then mark as good/bad
    # (Simplified - in reality, agent would run tests)

    for i in range(10):  # Max 10 iterations
        # Test current commit
        is_bug_present = test_code()  # Agent runs tests

        if is_bug_present:
            result = env.step(GitAction(
                action_type="execute_git_command",
                command="bisect bad",
                working_dir="OpenEnv"
            ))
        else:
            result = env.step(GitAction(
                action_type="execute_git_command",
                command="bisect good",
                working_dir="OpenEnv"
            ))

        # Check if bisect found the commit
        if "is the first bad commit" in result.observation.output:
            print(f"Found bad commit: {result.observation.output}")
            break

    # Reset bisect
    env.step(GitAction(
        action_type="execute_git_command",
        command="bisect reset",
        working_dir="OpenEnv"
    ))

    env.close()
```

---

## Best Practices

### 1. Error Handling

```python
def robust_git_operation(env, command, working_dir, max_retries=3):
    """Execute git command with error handling."""

    for attempt in range(max_retries):
        result = env.step(GitAction(
            action_type="execute_git_command",
            command=command,
            working_dir=working_dir
        ))

        if result.observation.success:
            return result

        print(f"Attempt {attempt + 1} failed: {result.observation.error}")

        # Handle specific errors
        if "not a git repository" in result.observation.error.lower():
            # Need to clone first
            env.step(GitAction(action_type="clone_repo", repo_name=working_dir))
        elif "already exists" in result.observation.error.lower():
            # Branch already exists, just switch to it
            return env.step(GitAction(
                action_type="execute_git_command",
                command=f"checkout {extract_branch_name(command)}",
                working_dir=working_dir
            ))
        elif "conflict" in result.observation.error.lower():
            # Merge conflict - abort and return
            env.step(GitAction(
                action_type="execute_git_command",
                command="merge --abort",
                working_dir=working_dir
            ))
            return result
        else:
            # Unknown error, retry
            continue

    return result
```

### 2. State Validation

```python
def validate_git_state(env, working_dir):
    """Validate git repository state before operations."""

    # Check if directory exists
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="status",
        working_dir=working_dir
    ))

    if not result.observation.success:
        return False, "Repository not found or not initialized"

    # Check for uncommitted changes
    if "Changes not staged" in result.observation.output:
        return False, "Uncommitted changes present"

    # Check for merge in progress
    if "You have unmerged paths" in result.observation.output:
        return False, "Merge in progress"

    return True, "Repository state is clean"
```

### 3. Command Composition

```python
def create_commit_with_validation(env, repo_name, message, files=None):
    """Create commit with proper validation."""

    # 1. Check status
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="status --short",
        working_dir=repo_name
    ))

    if not result.observation.output.strip():
        return None, "No changes to commit"

    # 2. Stage files
    if files:
        for file in files:
            env.step(GitAction(
                action_type="execute_git_command",
                command=f"add {file}",
                working_dir=repo_name
            ))
    else:
        env.step(GitAction(
            action_type="execute_git_command",
            command="add .",
            working_dir=repo_name
        ))

    # 3. Verify staging
    result = env.step(GitAction(
        action_type="execute_git_command",
        command="diff --cached --name-only",
        working_dir=repo_name
    ))

    staged_files = result.observation.output.strip().split('\n')
    print(f"Staging {len(staged_files)} files")

    # 4. Create commit
    result = env.step(GitAction(
        action_type="execute_git_command",
        command=f'commit -m "{message}"',
        working_dir=repo_name
    ))

    if result.observation.success:
        # Extract commit hash
        import re
        match = re.search(r'\[.+\s+([a-f0-9]+)\]', result.observation.output)
        if match:
            commit_hash = match.group(1)
            return commit_hash, "Commit created successfully"

    return None, result.observation.error
```

### 4. Logging & Monitoring

```python
class GitOperationLogger:
    """Log all Git operations for debugging and analysis."""

    def __init__(self):
        self.operations = []

    def log_operation(self, action, result):
        self.operations.append({
            "timestamp": time.time(),
            "action_type": action.action_type,
            "command": action.command if hasattr(action, "command") else None,
            "success": result.observation.success,
            "output": result.observation.output,
            "error": result.observation.error,
        })

    def get_failed_operations(self):
        return [op for op in self.operations if not op["success"]]

    def get_command_statistics(self):
        from collections import Counter
        commands = [op["command"] for op in self.operations if op["command"]]
        return Counter([cmd.split()[0] for cmd in commands])

    def save_log(self, filename):
        import json
        with open(filename, "w") as f:
            json.dump(self.operations, f, indent=2)
```

---

## Summary

The Git Environment provides a powerful platform for training agents on version control operations:

✅ **Fast reset** (<1s) for efficient training
✅ **Task-based isolation** for reproducible states
✅ **Full Git command support** for realistic workflows
✅ **Shared Gitea server** for efficient resource use

**Key Implementation Patterns:**
1. Use `list_repos` → `clone_repo` → `execute_git_command` workflow
2. Handle errors gracefully (repository not found, merge conflicts, etc.)
3. Validate state before operations
4. Log operations for debugging and analysis

**Training Progression:**
1. Basic commands (status, log, diff)
2. Branch management (create, switch, merge)
3. Staging & committing
4. Advanced workflows (rebase, cherry-pick, bisect)

**Next Steps:**
1. Start with basic command training
2. Build LLM-based agent with command generation
3. Implement multi-step workflows
4. Add code editing + Git operations for full coding assistant

**Key Insight**: The fast reset capability makes Git Environment ideal for curriculum learning - train on progressively complex Git workflows without the overhead of environment recreation.
