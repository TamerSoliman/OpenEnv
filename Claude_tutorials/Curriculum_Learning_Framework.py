"""
Curriculum Learning Framework for OpenEnv

This module provides a framework for curriculum learning - training agents by gradually
increasing task difficulty from easy to hard. This often leads to faster learning and
better final performance compared to training on hard tasks from the start.

Key Components:
1. CurriculumScheduler: Manages difficulty progression over training
2. Performance-based progression: Advance when agent shows mastery
3. Multi-environment scheduling: Balance training across difficulty levels
4. Integration with parallel execution for efficient data collection

Research Background:
- "Curriculum Learning" (Bengio et al., 2009)
- "Automatic Curriculum Learning For Deep RL" (Portelas et al., 2020)
- "Teacher-Student Curriculum Learning" (Matiisen et al., 2017)

Author: Claude
License: MIT
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple
from enum import Enum
import numpy as np


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class ProgressionStrategy(Enum):
    """
    Strategy for progressing through curriculum stages.

    LINEAR: Progress after fixed number of episodes
    THRESHOLD: Progress when performance exceeds threshold
    ADAPTIVE: Adjust difficulty dynamically based on performance
    MIXED: Combine multiple strategies
    """
    LINEAR = "linear"
    THRESHOLD = "threshold"
    ADAPTIVE = "adaptive"
    MIXED = "mixed"


@dataclass
class CurriculumStage:
    """
    A single stage in the curriculum.

    Attributes:
        name: Human-readable name (e.g., "easy", "medium", "hard")
        env_config: Configuration for environments at this difficulty
        target_performance: Performance threshold to advance (for threshold strategy)
        min_episodes: Minimum episodes before allowing progression
        weight: Proportion of training data from this stage (for mixed strategies)
    """
    name: str
    env_config: Dict[str, Any]
    target_performance: float = 0.0
    min_episodes: int = 0
    weight: float = 1.0


@dataclass
class CurriculumState:
    """
    Current state of curriculum learning.

    Tracks progress through stages and performance history.
    """
    current_stage_idx: int
    episodes_in_stage: int
    total_episodes: int
    performance_history: List[float]
    stage_performances: Dict[str, List[float]]  # Performance per stage


# =============================================================================
# CURRICULUM SCHEDULER
# =============================================================================

class CurriculumScheduler:
    """
    Manages curriculum learning progression across difficulty stages.

    Why Use This:
    - Easier learning: Start with simple tasks before hard ones
    - Faster convergence: Agents learn basic skills quickly
    - Better exploration: Avoid getting stuck in hard tasks early
    - Performance tracking: Monitor mastery of each difficulty level

    When to Use:
    - Training on tasks with clear difficulty levels
    - Agent struggles to learn from hard tasks directly
    - Want to maximize sample efficiency
    - Multi-task learning with varying complexity

    Example progression:
        Stage 1 (easy): 100 episodes → 80% success
        Stage 2 (medium): 200 episodes → 60% success
        Stage 3 (hard): 300 episodes → 40% success (final performance)
    """

    def __init__(
        self,
        stages: List[CurriculumStage],
        strategy: ProgressionStrategy = ProgressionStrategy.THRESHOLD,
        performance_window: int = 10
    ):
        """
        Initialize curriculum scheduler.

        Args:
            stages: List of curriculum stages (easy to hard)
            strategy: Progression strategy to use
            performance_window: Number of recent episodes for performance calculation
        """
        if not stages:
            raise ValueError("Must provide at least one curriculum stage")

        self.stages = stages
        self.strategy = strategy
        self.performance_window = performance_window

        # WHY: Track curriculum state throughout training
        self.state = CurriculumState(
            current_stage_idx=0,
            episodes_in_stage=0,
            total_episodes=0,
            performance_history=[],
            stage_performances={stage.name: [] for stage in stages}
        )

    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[self.state.current_stage_idx]

    def update(self, episode_return: float) -> bool:
        """
        Update curriculum state with new episode result.

        Args:
            episode_return: Total return from the episode

        Returns:
            True if advanced to next stage, False otherwise
        """
        # WHY: Record performance for progression decisions
        self.state.performance_history.append(episode_return)
        self.state.episodes_in_stage += 1
        self.state.total_episodes += 1

        current_stage = self.get_current_stage()
        self.state.stage_performances[current_stage.name].append(episode_return)

        # WHY: Check if should progress to next stage
        advanced = self._check_progression()

        return advanced

    def _check_progression(self) -> bool:
        """
        Check if agent should progress to next difficulty stage.

        Returns:
            True if progressed, False otherwise
        """
        # WHY: Don't progress if already at final stage
        if self.state.current_stage_idx >= len(self.stages) - 1:
            return False

        current_stage = self.get_current_stage()

        # WHY: Always enforce minimum episodes before allowing progression
        if self.state.episodes_in_stage < current_stage.min_episodes:
            return False

        # WHY: Apply strategy-specific progression logic
        should_progress = False

        if self.strategy == ProgressionStrategy.LINEAR:
            # WHY: Progress after fixed number of episodes
            should_progress = True

        elif self.strategy == ProgressionStrategy.THRESHOLD:
            # WHY: Progress when recent performance exceeds threshold
            recent_performance = self._get_recent_performance()
            should_progress = recent_performance >= current_stage.target_performance

        elif self.strategy == ProgressionStrategy.ADAPTIVE:
            # WHY: Progress when performance plateaus (no improvement)
            should_progress = self._detect_plateau()

        if should_progress:
            self._advance_stage()
            return True

        return False

    def _advance_stage(self):
        """Advance to next curriculum stage."""
        self.state.current_stage_idx += 1
        self.state.episodes_in_stage = 0
        print(f"\n{'='*60}")
        print(f"CURRICULUM PROGRESSION: Advanced to stage {self.state.current_stage_idx + 1}")
        print(f"New stage: {self.get_current_stage().name}")
        print(f"{'='*60}\n")

    def _get_recent_performance(self) -> float:
        """
        Get mean performance over recent episodes.

        Uses performance_window to calculate moving average.
        """
        if not self.state.performance_history:
            return 0.0

        # WHY: Use recent performance to avoid being influenced by early struggles
        recent = self.state.performance_history[-self.performance_window:]
        return np.mean(recent)

    def _detect_plateau(self) -> bool:
        """
        Detect if performance has plateaued (stopped improving).

        Simple heuristic: Compare first half vs second half of recent window.
        If no significant improvement, consider it a plateau.
        """
        if len(self.state.performance_history) < self.performance_window * 2:
            return False

        # WHY: Compare two halves of performance window
        recent = self.state.performance_history[-self.performance_window * 2:]
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])

        # WHY: If improvement is less than 5%, consider plateaued
        improvement = (second_half - first_half) / (abs(first_half) + 1e-8)
        return improvement < 0.05

    def get_stage_distribution(self) -> Dict[str, int]:
        """
        Get recommended environment distribution for mixed training.

        Useful for adaptive curriculum where you train on multiple stages simultaneously.

        Returns:
            Dict mapping stage name to number of environments
        """
        if self.strategy != ProgressionStrategy.MIXED:
            # WHY: For non-mixed strategies, focus only on current stage
            return {self.get_current_stage().name: 1}

        # WHY: For mixed strategy, use stage weights
        total_weight = sum(stage.weight for stage in self.stages)
        distribution = {}

        for stage in self.stages:
            proportion = stage.weight / total_weight
            distribution[stage.name] = proportion

        return distribution

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive curriculum statistics.

        Useful for monitoring, logging, and visualization.
        """
        current_stage = self.get_current_stage()

        stats = {
            "current_stage": current_stage.name,
            "current_stage_idx": self.state.current_stage_idx,
            "episodes_in_stage": self.state.episodes_in_stage,
            "total_episodes": self.state.total_episodes,
            "recent_performance": self._get_recent_performance(),
        }

        # WHY: Add per-stage statistics
        for stage_name, performances in self.state.stage_performances.items():
            if performances:
                stats[f"{stage_name}_mean"] = np.mean(performances)
                stats[f"{stage_name}_std"] = np.std(performances)
                stats[f"{stage_name}_count"] = len(performances)

        return stats


# =============================================================================
# INTEGRATION WITH PARALLEL EXECUTION
# =============================================================================

class CurriculumParallelTrainer:
    """
    Combines curriculum learning with parallel environment execution.

    Why Use This:
    - Efficient data collection across curriculum stages
    - Smooth difficulty transitions
    - Real-time performance tracking
    - Integrates with ParallelRolloutCollector

    Example workflow:
        1. Start with 8 "easy" environments
        2. Collect rollouts in parallel
        3. When performance threshold met, transition to "medium"
        4. Continue until "hard" stage mastered
    """

    def __init__(
        self,
        curriculum_scheduler: CurriculumScheduler,
        env_factory: Callable[[Dict[str, Any]], Any],
        num_parallel_envs: int = 8
    ):
        """
        Initialize curriculum parallel trainer.

        Args:
            curriculum_scheduler: CurriculumScheduler instance
            env_factory: Function that creates env client from config
            num_parallel_envs: Number of parallel environments to use
        """
        self.scheduler = curriculum_scheduler
        self.env_factory = env_factory
        self.num_parallel_envs = num_parallel_envs

        # WHY: Initialize with environments at first stage
        self._current_env_clients = self._create_stage_environments()

    def _create_stage_environments(self) -> List[Any]:
        """
        Create environment clients for current curriculum stage.

        Returns:
            List of environment client instances
        """
        current_stage = self.scheduler.get_current_stage()

        # WHY: Create num_parallel_envs environments with stage config
        env_clients = []
        for i in range(self.num_parallel_envs):
            env_client = self.env_factory(current_stage.env_config)
            env_clients.append(env_client)

        return env_clients

    def train_step(
        self,
        agent_fn: Callable,
        parallel_collector: Any
    ) -> Tuple[List[Any], Dict[str, Any], bool]:
        """
        Execute one training step with curriculum management.

        Args:
            agent_fn: Agent policy function
            parallel_collector: ParallelRolloutCollector instance

        Returns:
            Tuple of (rollouts, curriculum_stats, stage_changed)
        """
        # WHY: Collect rollouts from current stage environments
        rollouts, parallel_stats = parallel_collector.collect_with_stats(
            self._current_env_clients,
            agent_fn
        )

        # WHY: Update curriculum with performance
        stage_changed = self.scheduler.update(parallel_stats.mean_return)

        # WHY: If stage changed, create new environments
        if stage_changed:
            print(f"Creating environments for new stage: {self.scheduler.get_current_stage().name}")
            self._current_env_clients = self._create_stage_environments()

        # WHY: Get curriculum statistics for logging
        curriculum_stats = self.scheduler.get_statistics()

        return rollouts, curriculum_stats, stage_changed

    def get_current_environments(self) -> List[Any]:
        """Get current stage environment clients."""
        return self._current_env_clients


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_threshold_curriculum():
    """
    Example: Threshold-based curriculum progression.

    Agent progresses when it achieves target performance on current stage.
    """
    print("=" * 80)
    print("EXAMPLE 1: Threshold-Based Curriculum Learning")
    print("=" * 80)

    # WHY: Define curriculum stages with increasing difficulty
    stages = [
        CurriculumStage(
            name="easy",
            env_config={"difficulty": 1, "max_steps": 10},
            target_performance=15.0,  # Must achieve 15.0 return
            min_episodes=20  # Minimum 20 episodes before progression
        ),
        CurriculumStage(
            name="medium",
            env_config={"difficulty": 2, "max_steps": 20},
            target_performance=25.0,
            min_episodes=30
        ),
        CurriculumStage(
            name="hard",
            env_config={"difficulty": 3, "max_steps": 30},
            target_performance=35.0,
            min_episodes=50
        ),
    ]

    # WHY: Create scheduler with threshold strategy
    scheduler = CurriculumScheduler(
        stages=stages,
        strategy=ProgressionStrategy.THRESHOLD,
        performance_window=10  # Use last 10 episodes for threshold check
    )

    # WHY: Simulate training with improving performance
    print("\nSimulating training...")
    np.random.seed(42)

    for episode in range(200):
        # WHY: Simulate improving agent (returns increase over time)
        current_stage = scheduler.get_current_stage()
        base_return = 5.0 + episode * 0.15 + np.random.randn() * 2.0
        episode_return = base_return

        # WHY: Update curriculum and check for progression
        advanced = scheduler.update(episode_return)

        if episode % 10 == 0 or advanced:
            stats = scheduler.get_statistics()
            print(f"Episode {episode}: "
                  f"Stage={stats['current_stage']}, "
                  f"Return={episode_return:.2f}, "
                  f"Recent Avg={stats['recent_performance']:.2f}")

        if advanced:
            print(f"  → Advanced to {scheduler.get_current_stage().name}!")

    # WHY: Print final statistics
    final_stats = scheduler.get_statistics()
    print("\nFinal Curriculum Statistics:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print()


def example_adaptive_curriculum():
    """
    Example: Adaptive curriculum that progresses when performance plateaus.
    """
    print("=" * 80)
    print("EXAMPLE 2: Adaptive Curriculum Learning (Plateau Detection)")
    print("=" * 80)

    stages = [
        CurriculumStage(
            name="beginner",
            env_config={"difficulty": 1},
            min_episodes=30
        ),
        CurriculumStage(
            name="intermediate",
            env_config={"difficulty": 2},
            min_episodes=40
        ),
        CurriculumStage(
            name="advanced",
            env_config={"difficulty": 3},
            min_episodes=50
        ),
    ]

    scheduler = CurriculumScheduler(
        stages=stages,
        strategy=ProgressionStrategy.ADAPTIVE,
        performance_window=20  # Larger window for plateau detection
    )

    print("\nSimulating training with plateau detection...")
    np.random.seed(42)

    for episode in range(150):
        current_stage = scheduler.get_current_stage()

        # WHY: Simulate plateau - performance increases then flattens
        if episode < 40:
            base_return = 5.0 + episode * 0.3
        else:
            base_return = 17.0  # Plateau at ~17

        episode_return = base_return + np.random.randn() * 2.0

        advanced = scheduler.update(episode_return)

        if episode % 10 == 0 or advanced:
            stats = scheduler.get_statistics()
            print(f"Episode {episode}: "
                  f"Stage={stats['current_stage']}, "
                  f"Return={episode_return:.2f}")

        if advanced:
            print(f"  → Plateau detected! Advanced to {scheduler.get_current_stage().name}")

    print()


def example_mixed_curriculum():
    """
    Example: Mixed curriculum training on multiple stages simultaneously.
    """
    print("=" * 80)
    print("EXAMPLE 3: Mixed Curriculum (Multi-Stage Training)")
    print("=" * 80)

    stages = [
        CurriculumStage(
            name="easy",
            env_config={"difficulty": 1},
            weight=0.5  # 50% of training
        ),
        CurriculumStage(
            name="medium",
            env_config={"difficulty": 2},
            weight=0.3  # 30% of training
        ),
        CurriculumStage(
            name="hard",
            env_config={"difficulty": 3},
            weight=0.2  # 20% of training
        ),
    ]

    scheduler = CurriculumScheduler(
        stages=stages,
        strategy=ProgressionStrategy.MIXED
    )

    # WHY: Get recommended distribution of environments
    distribution = scheduler.get_stage_distribution()

    print("\nRecommended environment distribution:")
    for stage_name, weight in distribution.items():
        print(f"  {stage_name}: {weight * 100:.1f}%")

    print("\nWith 10 parallel environments:")
    for stage_name, weight in distribution.items():
        num_envs = int(weight * 10)
        print(f"  {stage_name}: {num_envs} environments")

    print()


if __name__ == "__main__":
    """
    Run all curriculum learning examples.

    To run this file:
        python Claude_tutorials/Curriculum_Learning_Framework.py

    To use in your own code:
        from Claude_tutorials.Curriculum_Learning_Framework import CurriculumScheduler
        scheduler = CurriculumScheduler(stages, strategy=ProgressionStrategy.THRESHOLD)
        for episode in range(1000):
            rollouts = collect_rollouts()
            advanced = scheduler.update(mean_return)
            if advanced:
                update_environments()
    """
    example_threshold_curriculum()
    example_adaptive_curriculum()
    example_mixed_curriculum()

    print("=" * 80)
    print("All curriculum learning examples completed successfully!")
    print("=" * 80)
