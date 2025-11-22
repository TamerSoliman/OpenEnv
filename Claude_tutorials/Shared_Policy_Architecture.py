"""
Shared Policy Architecture for Multi-Task Transfer Learning

This module implements policy architectures that enable knowledge sharing and transfer
across multiple OpenEnv environments. The key insight is that OpenEnv environments share
a common observation space (text) and often similar action spaces, allowing for
effective transfer learning through shared representations.

Key Architectures:
1. SharedEncoderPolicy: Shared text encoder + task-specific heads
2. UniversalValueFunction: Goal-conditioned Q-network for multi-task RL
3. ModularPolicy: Compositional policy with reusable skill modules

Research Foundation:
- "Universal Value Function Approximators" (Schaul et al., 2015)
- "Distral: Robust Multi-Task RL" (Teh et al., 2017)
- "Multi-Task Deep RL" (Caruana, 1997)

Author: Claude
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

@dataclass
class PolicyOutput:
    """
    Output from a policy forward pass.

    Attributes:
        action_logits: Logits for action selection
        value: State value estimate (optional, for actor-critic)
        features: Intermediate representations (for analysis)
    """
    action_logits: torch.Tensor
    value: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None


# =============================================================================
# ARCHITECTURE 1: SHARED ENCODER POLICY
# =============================================================================

class SharedEncoderPolicy(nn.Module):
    """
    Multi-task policy with shared text encoder and task-specific heads.

    Architecture:
        Text Observation
             ↓
        Shared Encoder (Transformer)
             ↓
        Shared Features (768-dim)
             ↓
        ┌────┼────┬────┐
        ↓    ↓    ↓    ↓
      Head1 Head2 Head3 Head4
      (Echo)(Code)(Git)(Browser)

    Why This Works:
    - All OpenEnv environments use text observations
    - Shared encoder learns general text understanding
    - Task-specific heads specialize for each environment
    - Positive transfer through shared representations

    When to Use:
    - Training on multiple OpenEnv environments simultaneously
    - Transfer learning (pre-train on source tasks, fine-tune on target)
    - Want to leverage pre-trained language models (BERT, GPT)
    """

    def __init__(
        self,
        task_names: List[str],
        vocab_size: int = 10000,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        action_dims: Dict[str, int] = None,
        use_pretrained: bool = False
    ):
        """
        Initialize shared encoder policy.

        Args:
            task_names: List of task identifiers (e.g., ["echo", "coding", "git"])
            vocab_size: Vocabulary size for text encoding
            embed_dim: Embedding dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            action_dims: Dict mapping task name to action dimension
            use_pretrained: Whether to use pre-trained transformer (requires transformers library)
        """
        super().__init__()

        self.task_names = task_names
        self.embed_dim = embed_dim
        self.use_pretrained = use_pretrained

        # WHY: Default action dimensions if not specified
        if action_dims is None:
            action_dims = {task: 10 for task in task_names}  # Default: 10 actions per task
        self.action_dims = action_dims

        # WHY: Shared encoder processes all text observations
        if use_pretrained:
            # Option 1: Use pre-trained transformer (requires transformers library)
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.encoder = AutoModel.from_pretrained("bert-base-uncased")
                self.embed_dim = 768  # BERT hidden size
            except ImportError:
                print("Warning: transformers library not available, using custom encoder")
                self.use_pretrained = False
                self._init_custom_encoder(vocab_size, embed_dim, num_heads, num_layers)
        else:
            # Option 2: Custom transformer encoder
            self._init_custom_encoder(vocab_size, embed_dim, num_heads, num_layers)

        # WHY: Task-specific heads map shared features to task-specific actions
        # Each task gets its own output layer, allowing specialization
        self.task_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, action_dims[task])
            )
            for task in task_names
        })

        # WHY: Value heads for actor-critic training (optional)
        self.value_heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(self.embed_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
            for task in task_names
        })

    def _init_custom_encoder(self, vocab_size, embed_dim, num_heads, num_layers):
        """Initialize custom transformer encoder."""
        # WHY: Embedding layer converts token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # WHY: Positional encoding for sequence position information
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, embed_dim))

        # WHY: Transformer encoder layers for text understanding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def encode_text(self, text_input):
        """
        Encode text observation to feature vector.

        Args:
            text_input: Either token IDs (custom encoder) or text string (pretrained)

        Returns:
            Feature vector of shape (batch_size, embed_dim)
        """
        if self.use_pretrained:
            # WHY: Use pre-trained BERT encoder
            outputs = self.encoder(**text_input)
            # WHY: Use [CLS] token representation as sentence embedding
            features = outputs.last_hidden_state[:, 0, :]
        else:
            # WHY: Custom encoder processing
            # Embed tokens
            embedded = self.embedding(text_input)

            # WHY: Add positional encoding
            seq_len = embedded.size(1)
            embedded = embedded + self.pos_encoding[:, :seq_len, :]

            # WHY: Pass through transformer layers
            encoded = self.encoder(embedded)

            # WHY: Pool to single vector (mean pooling)
            features = encoded.mean(dim=1)

        return features

    def forward(self, observation, task_id: str) -> PolicyOutput:
        """
        Forward pass for a specific task.

        Args:
            observation: Text observation (token IDs or text string)
            task_id: Task identifier (must be in task_names)

        Returns:
            PolicyOutput with action logits and value estimate
        """
        # WHY: Encode observation using shared encoder
        # This is where knowledge transfer happens - all tasks benefit from shared encoding
        features = self.encode_text(observation)

        # WHY: Get task-specific action logits
        action_logits = self.task_heads[task_id](features)

        # WHY: Get value estimate for actor-critic
        value = self.value_heads[task_id](features)

        return PolicyOutput(
            action_logits=action_logits,
            value=value,
            features=features
        )

    def freeze_encoder(self):
        """
        Freeze encoder weights (useful for fine-tuning).

        Why: When fine-tuning on a new task, sometimes you want to keep the shared
        encoder fixed and only train the new task head. This prevents catastrophic
        forgetting of previously learned representations.
        """
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def add_task(self, task_name: str, action_dim: int):
        """
        Add a new task head (for continual learning).

        Args:
            task_name: New task identifier
            action_dim: Action dimension for new task

        Why: Allows adding new tasks without retraining from scratch.
        The shared encoder already contains useful representations from previous tasks.
        """
        self.task_names.append(task_name)
        self.action_dims[task_name] = action_dim

        # WHY: Add new task head
        self.task_heads[task_name] = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        # WHY: Add new value head
        self.value_heads[task_name] = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


# =============================================================================
# ARCHITECTURE 2: UNIVERSAL VALUE FUNCTION
# =============================================================================

class UniversalValueFunction(nn.Module):
    """
    Universal value function for multi-goal RL.

    Architecture:
        (State, Goal)
             ↓
        ┌────┴────┐
        ↓         ↓
    State Enc  Goal Enc
        ↓         ↓
        └────┬────┘
             ↓
        Combined Features
             ↓
        Q-values

    Why This Works:
    - Condition on both state and goal
    - Single network handles all tasks (goals)
    - Easy to add new goals without retraining

    When to Use:
    - Multi-task RL where tasks differ in goals
    - Want to query value of state-action pairs for different goals
    - Hindsight Experience Replay (HER) style training

    Reference: "Universal Value Function Approximators" (Schaul et al., 2015)
    """

    def __init__(
        self,
        state_vocab_size: int = 10000,
        goal_vocab_size: int = 1000,
        embed_dim: int = 256,
        num_actions: int = 10
    ):
        """
        Initialize universal value function.

        Args:
            state_vocab_size: Vocabulary size for state encoding
            goal_vocab_size: Vocabulary size for goal encoding
            embed_dim: Embedding dimension
            num_actions: Number of discrete actions
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_actions = num_actions

        # WHY: Separate encoders for state and goal
        # Allows different levels of abstraction
        self.state_encoder = nn.Sequential(
            nn.Embedding(state_vocab_size, embed_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True),
                num_layers=2
            )
        )

        self.goal_encoder = nn.Sequential(
            nn.Embedding(goal_vocab_size, embed_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True),
                num_layers=2
            )
        )

        # WHY: Q-head predicts value for each action given (state, goal)
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),  # Concatenated state + goal
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, state, goal) -> torch.Tensor:
        """
        Compute Q-values for all actions given state and goal.

        Args:
            state: State observation (token IDs)
            goal: Goal description (token IDs)

        Returns:
            Q-values for each action: shape (batch_size, num_actions)
        """
        # WHY: Encode state and goal separately
        state_features = self.state_encoder(state).mean(dim=1)  # Pool to vector
        goal_features = self.goal_encoder(goal).mean(dim=1)

        # WHY: Concatenate state and goal representations
        # This allows Q-network to condition on both
        combined = torch.cat([state_features, goal_features], dim=-1)

        # WHY: Predict Q-values for all actions
        q_values = self.q_head(combined)

        return q_values


# =============================================================================
# ARCHITECTURE 3: MODULAR POLICY
# =============================================================================

class SkillModule(nn.Module):
    """
    A single reusable skill module.

    Represents a primitive action or skill that can be composed.
    """

    def __init__(self, skill_name: str, input_dim: int, output_dim: int):
        super().__init__()
        self.skill_name = skill_name
        self.skill_policy = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, features):
        return self.skill_policy(features)


class ModularPolicy(nn.Module):
    """
    Modular policy with compositional skills.

    Architecture:
        Observation
             ↓
        Feature Encoder
             ↓
        Skill Selector → Select skill
             ↓
        ┌────┼────┬────┐
        ↓    ↓    ↓    ↓
     Skill1 Skill2 Skill3 Skill4
    (navigate)(read)(write)(execute)
             ↓
        Action

    Why This Works:
    - Compositional: Combine basic skills for complex tasks
    - Reusable: Skills transfer across tasks
    - Interpretable: Know which skill is active

    When to Use:
    - Tasks can be decomposed into sub-skills
    - Want interpretable policies
    - Transfer learning through skill reuse
    """

    def __init__(
        self,
        skill_names: List[str],
        vocab_size: int = 10000,
        embed_dim: int = 256,
        action_dim: int = 10
    ):
        """
        Initialize modular policy.

        Args:
            skill_names: List of skill identifiers
            vocab_size: Vocabulary size for observation encoding
            embed_dim: Feature dimension
            action_dim: Action space dimension for each skill
        """
        super().__init__()

        self.skill_names = skill_names
        self.embed_dim = embed_dim

        # WHY: Shared feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True),
                num_layers=2
            )
        )

        # WHY: Skill selector decides which skill to use
        self.skill_selector = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(skill_names))
        )

        # WHY: Each skill has its own policy module
        self.skills = nn.ModuleDict({
            skill: SkillModule(skill, embed_dim, action_dim)
            for skill in skill_names
        })

    def forward(self, observation, skill_id: Optional[str] = None):
        """
        Forward pass.

        Args:
            observation: Text observation (token IDs)
            skill_id: Optional skill to use (if None, automatically select)

        Returns:
            Action logits and selected skill
        """
        # WHY: Encode observation to features
        features = self.feature_encoder(observation).mean(dim=1)

        # WHY: Select skill (either specified or automatically)
        if skill_id is None:
            skill_logits = self.skill_selector(features)
            skill_id = self.skill_names[skill_logits.argmax(dim=-1).item()]

        # WHY: Execute selected skill
        action_logits = self.skills[skill_id](features)

        return action_logits, skill_id


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def compute_multi_task_loss(
    policy: SharedEncoderPolicy,
    batch: Dict[str, Any],
    task_weights: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    """
    Compute multi-task training loss.

    Args:
        policy: SharedEncoderPolicy instance
        batch: Dict with keys "observations", "actions", "task_ids", "returns"
        task_weights: Optional weights for each task (for balancing)

    Returns:
        Total loss across all tasks
    """
    observations = batch["observations"]
    actions = batch["actions"]
    task_ids = batch["task_ids"]
    returns = batch["returns"]

    if task_weights is None:
        task_weights = {task: 1.0 for task in policy.task_names}

    total_loss = 0.0
    task_losses = {}

    # WHY: Compute loss for each task in the batch
    for task_id in policy.task_names:
        # WHY: Get samples for this task
        task_mask = task_ids == task_id
        if not task_mask.any():
            continue  # No samples for this task in batch

        task_obs = observations[task_mask]
        task_actions = actions[task_mask]
        task_returns = returns[task_mask]

        # WHY: Forward pass for this task
        output = policy(task_obs, task_id)

        # WHY: Policy loss (cross-entropy for discrete actions)
        policy_loss = F.cross_entropy(output.action_logits, task_actions)

        # WHY: Value loss (MSE for value prediction)
        value_loss = F.mse_loss(output.value.squeeze(), task_returns)

        # WHY: Combined loss
        task_loss = policy_loss + 0.5 * value_loss
        task_losses[task_id] = task_loss.item()

        # WHY: Weight and accumulate
        total_loss += task_weights[task_id] * task_loss

    return total_loss, task_losses


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_shared_encoder():
    """
    Example: Train shared encoder policy on multiple tasks.
    """
    print("=" * 80)
    print("EXAMPLE 1: Shared Encoder Multi-Task Policy")
    print("=" * 80)

    # WHY: Define tasks
    task_names = ["echo", "coding", "git"]
    action_dims = {"echo": 10, "coding": 20, "git": 15}

    # WHY: Create policy
    policy = SharedEncoderPolicy(
        task_names=task_names,
        vocab_size=10000,
        embed_dim=256,
        num_heads=4,
        num_layers=3,
        action_dims=action_dims,
        use_pretrained=False
    )

    # WHY: Example forward pass for each task
    batch_size = 4
    seq_len = 20

    for task in task_names:
        # WHY: Mock observation (token IDs)
        obs = torch.randint(0, 10000, (batch_size, seq_len))

        # WHY: Forward pass
        output = policy(obs, task)

        print(f"\nTask: {task}")
        print(f"  Action logits shape: {output.action_logits.shape}")
        print(f"  Value shape: {output.value.shape}")
        print(f"  Features shape: {output.features.shape}")

    # WHY: Demonstrate adding new task
    print("\nAdding new task: 'browser'")
    policy.add_task("browser", action_dim=25)

    obs = torch.randint(0, 10000, (batch_size, seq_len))
    output = policy(obs, "browser")
    print(f"  New task action logits shape: {output.action_logits.shape}")

    print()


def example_universal_value_function():
    """
    Example: Universal value function for multi-goal RL.
    """
    print("=" * 80)
    print("EXAMPLE 2: Universal Value Function")
    print("=" * 80)

    # WHY: Create UVF
    uvf = UniversalValueFunction(
        state_vocab_size=10000,
        goal_vocab_size=1000,
        embed_dim=256,
        num_actions=10
    )

    # WHY: Example state and goals
    batch_size = 4
    state = torch.randint(0, 10000, (batch_size, 20))  # State observation
    goal1 = torch.randint(0, 1000, (batch_size, 5))    # Goal 1: "complete echo task"
    goal2 = torch.randint(0, 1000, (batch_size, 5))    # Goal 2: "solve coding problem"

    # WHY: Compute Q-values for different goals
    q_values_goal1 = uvf(state, goal1)
    q_values_goal2 = uvf(state, goal2)

    print(f"\nQ-values for goal 1 shape: {q_values_goal1.shape}")
    print(f"Q-values for goal 2 shape: {q_values_goal2.shape}")

    # WHY: Select best action for each goal
    action_goal1 = q_values_goal1.argmax(dim=-1)
    action_goal2 = q_values_goal2.argmax(dim=-1)

    print(f"\nBest actions for goal 1: {action_goal1}")
    print(f"Best actions for goal 2: {action_goal2}")
    print()


def example_modular_policy():
    """
    Example: Modular policy with compositional skills.
    """
    print("=" * 80)
    print("EXAMPLE 3: Modular Policy with Skills")
    print("=" * 80)

    # WHY: Define skills
    skill_names = ["navigate", "read", "write", "execute"]

    # WHY: Create modular policy
    policy = ModularPolicy(
        skill_names=skill_names,
        vocab_size=10000,
        embed_dim=256,
        action_dim=10
    )

    # WHY: Example observations
    batch_size = 4
    obs = torch.randint(0, 10000, (batch_size, 20))

    # WHY: Automatic skill selection
    action_logits, selected_skill = policy(obs)
    print(f"\nAutomatic skill selection:")
    print(f"  Selected skill: {selected_skill}")
    print(f"  Action logits shape: {action_logits.shape}")

    # WHY: Manual skill selection
    for skill in skill_names:
        action_logits, _ = policy(obs, skill_id=skill)
        print(f"\nManual skill '{skill}':")
        print(f"  Action logits shape: {action_logits.shape}")

    print()


if __name__ == "__main__":
    """
    Run all examples demonstrating shared policy architectures.

    To run this file:
        python Claude_tutorials/Shared_Policy_Architecture.py

    To use in your own code:
        from Claude_tutorials.Shared_Policy_Architecture import SharedEncoderPolicy
        policy = SharedEncoderPolicy(task_names=["echo", "coding"], ...)
        output = policy(observation, task_id="echo")
    """
    example_shared_encoder()
    example_universal_value_function()
    example_modular_policy()

    print("=" * 80)
    print("All shared policy architecture examples completed successfully!")
    print("=" * 80)
