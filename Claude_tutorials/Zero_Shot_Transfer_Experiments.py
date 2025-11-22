"""
Zero-Shot Transfer Experiments for OpenEnv

This module provides a framework for evaluating zero-shot transfer learning across
OpenEnv environments. Zero-shot transfer means applying a policy trained on source
tasks directly to target tasks without any fine-tuning.

Key Components:
1. TransferExperiment: Framework for running transfer experiments
2. TransferMetrics: Jumpstart, asymptotic performance, time-to-threshold
3. TransferMatrix: Compute transfer quality between all task pairs
4. Visualization: Plot learning curves and transfer matrices

Research Background:
- "Transfer in Deep RL" (Taylor & Stone, 2009)
- "Measuring Transfer in RL" (Ferguson & Mahadevan, 2006)
- "Task Similarity in RL" (Lazaric, 2012)

Author: Claude
License: MIT
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass
import time


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

@dataclass
class TransferMetrics:
    """
    Metrics for evaluating transfer learning quality.

    Attributes:
        jumpstart: Initial performance on target task
        time_to_threshold: Episodes needed to reach threshold performance
        asymptotic_performance: Final performance after training
        transfer_ratio: Speedup compared to training from scratch
        source_task: Source task name
        target_task: Target task name
    """
    jumpstart: float
    time_to_threshold: Optional[int]
    asymptotic_performance: float
    transfer_ratio: Optional[float]
    source_task: str
    target_task: str


@dataclass
class ExperimentResult:
    """
    Complete results from a transfer experiment.

    Attributes:
        learning_curve: List of episode returns
        metrics: TransferMetrics
        total_time: Wall-clock time for experiment
    """
    learning_curve: List[float]
    metrics: TransferMetrics
    total_time: float


# =============================================================================
# TRANSFER EXPERIMENT FRAMEWORK
# =============================================================================

class TransferExperiment:
    """
    Framework for running zero-shot transfer experiments.

    Workflow:
    1. Train policy on source task
    2. Evaluate on target task without fine-tuning (zero-shot)
    3. Compare to baseline (train from scratch on target)
    4. Compute transfer metrics

    Why Use This:
    - Standardized evaluation protocol
    - Fair comparison between transfer and baseline
    - Reproducible results

    When to Use:
    - Evaluating whether transfer learning helps
    - Comparing different source tasks for same target
    - Building transfer matrix across all task pairs
    """

    def __init__(
        self,
        random_seed: int = 42,
        num_eval_episodes: int = 10,
        performance_threshold: float = 0.7
    ):
        """
        Initialize transfer experiment framework.

        Args:
            random_seed: Random seed for reproducibility
            num_eval_episodes: Number of episodes for evaluation
            performance_threshold: Target performance for time-to-threshold metric
        """
        self.random_seed = random_seed
        self.num_eval_episodes = num_eval_episodes
        self.performance_threshold = performance_threshold

        np.random.seed(random_seed)

    def run_zero_shot_transfer(
        self,
        source_policy: Any,
        source_task: str,
        target_env: Any,
        target_task: str,
        max_steps_per_episode: int = 100
    ) -> ExperimentResult:
        """
        Run zero-shot transfer experiment.

        Steps:
        1. Take pre-trained source_policy
        2. Evaluate on target_env without any fine-tuning
        3. Measure performance

        Args:
            source_policy: Policy trained on source task
            source_task: Source task name
            target_env: Target environment
            target_task: Target task name
            max_steps_per_episode: Max steps per episode

        Returns:
            ExperimentResult with learning curve and metrics
        """
        print(f"\n{'='*60}")
        print(f"Zero-Shot Transfer: {source_task} → {target_task}")
        print(f"{'='*60}")

        start_time = time.time()

        # WHY: Evaluate policy on target environment
        learning_curve = []

        for episode in range(self.num_eval_episodes):
            episode_return = self._run_episode(
                source_policy,
                target_env,
                max_steps_per_episode
            )
            learning_curve.append(episode_return)

            if episode % 5 == 0:
                print(f"Episode {episode}: Return = {episode_return:.2f}")

        # WHY: Compute metrics
        metrics = self._compute_metrics(
            learning_curve,
            source_task,
            target_task,
            baseline_curve=None  # No baseline for zero-shot
        )

        total_time = time.time() - start_time

        print(f"\nZero-Shot Performance:")
        print(f"  Jumpstart: {metrics.jumpstart:.3f}")
        print(f"  Asymptotic: {metrics.asymptotic_performance:.3f}")
        print(f"  Time: {total_time:.2f}s")

        return ExperimentResult(
            learning_curve=learning_curve,
            metrics=metrics,
            total_time=total_time
        )

    def run_transfer_with_baseline(
        self,
        source_policy: Any,
        source_task: str,
        target_env: Any,
        target_task: str,
        baseline_policy: Any,
        max_steps_per_episode: int = 100
    ) -> Tuple[ExperimentResult, ExperimentResult]:
        """
        Run transfer experiment with baseline comparison.

        Compare:
        - Transfer: Pre-trained source_policy on target_env
        - Baseline: Randomly initialized baseline_policy on target_env

        Args:
            source_policy: Pre-trained policy
            source_task: Source task name
            target_env: Target environment
            target_task: Target task name
            baseline_policy: Randomly initialized policy
            max_steps_per_episode: Max steps per episode

        Returns:
            Tuple of (transfer_result, baseline_result)
        """
        print(f"\n{'='*60}")
        print(f"Transfer Experiment: {source_task} → {target_task}")
        print(f"{'='*60}")

        # WHY: Run transfer evaluation
        print("\n[1/2] Evaluating transfer policy...")
        transfer_result = self.run_zero_shot_transfer(
            source_policy, source_task, target_env, target_task, max_steps_per_episode
        )

        # WHY: Run baseline evaluation
        print("\n[2/2] Evaluating baseline policy...")
        baseline_result = self.run_zero_shot_transfer(
            baseline_policy, "random", target_env, target_task, max_steps_per_episode
        )

        # WHY: Compute transfer ratio
        transfer_result.metrics.transfer_ratio = (
            transfer_result.metrics.asymptotic_performance /
            (baseline_result.metrics.asymptotic_performance + 1e-8)
        )

        print(f"\n{'='*60}")
        print("Transfer vs. Baseline Comparison")
        print(f"{'='*60}")
        print(f"Transfer Jumpstart: {transfer_result.metrics.jumpstart:.3f}")
        print(f"Baseline Jumpstart: {baseline_result.metrics.jumpstart:.3f}")
        print(f"Improvement: {transfer_result.metrics.jumpstart - baseline_result.metrics.jumpstart:.3f}")
        print(f"\nTransfer Ratio: {transfer_result.metrics.transfer_ratio:.2f}x")

        return transfer_result, baseline_result

    def _run_episode(
        self,
        policy: Any,
        env: Any,
        max_steps: int
    ) -> float:
        """
        Run a single episode and return total return.

        This is a mock implementation. In real usage, replace with actual
        environment interaction logic.

        Args:
            policy: Policy to evaluate
            env: Environment
            max_steps: Maximum steps per episode

        Returns:
            Total episode return
        """
        # WHY: Mock implementation for demonstration
        # In real code, this would be:
        # obs = env.reset()
        # for step in range(max_steps):
        #     action = policy.get_action(obs)
        #     result = env.step(action)
        #     obs = result.observation
        #     episode_return += result.reward
        #     if result.done:
        #         break

        # Simulate episode with random return
        episode_return = np.random.randn() * 2 + 5.0  # Mean=5, Std=2

        return episode_return

    def _compute_metrics(
        self,
        learning_curve: List[float],
        source_task: str,
        target_task: str,
        baseline_curve: Optional[List[float]] = None
    ) -> TransferMetrics:
        """
        Compute transfer learning metrics.

        Metrics:
        1. Jumpstart: Initial performance (first episode)
        2. Asymptotic: Final performance (mean of last N episodes)
        3. Time-to-threshold: Episodes needed to reach threshold

        Args:
            learning_curve: Episode returns for transfer policy
            source_task: Source task name
            target_task: Target task name
            baseline_curve: Optional baseline returns for comparison

        Returns:
            TransferMetrics
        """
        # WHY: Jumpstart = performance at episode 0
        jumpstart = learning_curve[0]

        # WHY: Asymptotic = mean of last 20% of episodes
        last_n = max(1, len(learning_curve) // 5)
        asymptotic_performance = np.mean(learning_curve[-last_n:])

        # WHY: Time-to-threshold = first episode above threshold
        time_to_threshold = None
        for i, perf in enumerate(learning_curve):
            if perf >= self.performance_threshold:
                time_to_threshold = i
                break

        # WHY: Transfer ratio computed externally if baseline provided
        transfer_ratio = None

        return TransferMetrics(
            jumpstart=jumpstart,
            time_to_threshold=time_to_threshold,
            asymptotic_performance=asymptotic_performance,
            transfer_ratio=transfer_ratio,
            source_task=source_task,
            target_task=target_task
        )


# =============================================================================
# TRANSFER MATRIX
# =============================================================================

class TransferMatrix:
    """
    Compute transfer quality between all task pairs.

    Transfer matrix M[i, j] = transfer quality from task i to task j

    Quality metrics:
    - Jumpstart improvement
    - Asymptotic performance ratio
    - Time-to-threshold reduction

    Useful for:
    - Identifying which tasks transfer well to each other
    - Selecting source tasks for curriculum learning
    - Understanding task relationships
    """

    def __init__(self, task_names: List[str]):
        """
        Initialize transfer matrix.

        Args:
            task_names: List of task identifiers
        """
        self.task_names = task_names
        self.n_tasks = len(task_names)

        # WHY: Initialize matrix to store transfer quality
        self.matrix = np.zeros((self.n_tasks, self.n_tasks))

    def compute_matrix(
        self,
        policies: Dict[str, Any],
        environments: Dict[str, Any],
        experiment: TransferExperiment,
        baseline_policy: Any,
        metric: str = "jumpstart"
    ):
        """
        Compute full transfer matrix.

        For each (source, target) pair:
        1. Evaluate source policy on target environment
        2. Compare to baseline
        3. Record transfer quality

        Args:
            policies: Dict mapping task name to trained policy
            environments: Dict mapping task name to environment
            experiment: TransferExperiment instance
            baseline_policy: Baseline policy (random or untrained)
            metric: Metric to use ("jumpstart", "asymptotic", "transfer_ratio")
        """
        print(f"\n{'='*60}")
        print(f"Computing Transfer Matrix ({metric})")
        print(f"{'='*60}")

        for i, source_task in enumerate(self.task_names):
            for j, target_task in enumerate(self.task_names):
                if i == j:
                    # WHY: No transfer needed for same task (set to 1.0 = neutral)
                    self.matrix[i, j] = 1.0
                    continue

                # WHY: Run transfer experiment
                transfer_result, baseline_result = experiment.run_transfer_with_baseline(
                    source_policy=policies[source_task],
                    source_task=source_task,
                    target_env=environments[target_task],
                    target_task=target_task,
                    baseline_policy=baseline_policy
                )

                # WHY: Extract metric value
                if metric == "jumpstart":
                    # Improvement over baseline
                    value = (transfer_result.metrics.jumpstart -
                             baseline_result.metrics.jumpstart)
                elif metric == "asymptotic":
                    value = transfer_result.metrics.asymptotic_performance
                elif metric == "transfer_ratio":
                    value = transfer_result.metrics.transfer_ratio
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                self.matrix[i, j] = value

                print(f"  {source_task} → {target_task}: {value:.3f}")

        print(f"\n{'='*60}")
        print("Transfer Matrix Complete")
        print(f"{'='*60}")

    def get_best_source_for_target(self, target_task: str) -> str:
        """
        Get best source task for a given target task.

        Args:
            target_task: Target task name

        Returns:
            Source task name with highest transfer quality
        """
        target_idx = self.task_names.index(target_task)

        # WHY: Find source task with highest transfer quality
        source_idx = np.argmax(self.matrix[:, target_idx])

        return self.task_names[source_idx]

    def print_matrix(self):
        """Print transfer matrix in readable format."""
        print("\nTransfer Matrix:")
        print("(rows = source, cols = target)\n")

        # Header
        print("         ", end="")
        for task in self.task_names:
            print(f"{task[:8]:>10}", end="")
        print()

        # Matrix
        for i, source in enumerate(self.task_names):
            print(f"{source[:8]:>8} ", end="")
            for j in range(self.n_tasks):
                print(f"{self.matrix[i, j]:>10.3f}", end="")
            print()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_zero_shot_transfer():
    """
    Example: Basic zero-shot transfer experiment.
    """
    print("=" * 80)
    print("EXAMPLE 1: Zero-Shot Transfer Experiment")
    print("=" * 80)

    # WHY: Mock policies and environments for demonstration
    class MockPolicy:
        def __init__(self, name):
            self.name = name

        def get_action(self, obs):
            return "mock_action"

    class MockEnv:
        def __init__(self, name):
            self.name = name

    # WHY: Create experiment
    experiment = TransferExperiment(
        random_seed=42,
        num_eval_episodes=20,
        performance_threshold=6.0
    )

    # WHY: Run zero-shot transfer
    source_policy = MockPolicy("echo_policy")
    target_env = MockEnv("coding_env")

    result = experiment.run_zero_shot_transfer(
        source_policy=source_policy,
        source_task="echo",
        target_env=target_env,
        target_task="coding",
        max_steps_per_episode=100
    )

    # WHY: Print results
    print("\nResults:")
    print(f"  Learning curve: {[f'{x:.2f}' for x in result.learning_curve[:5]]} ...")
    print(f"  Jumpstart: {result.metrics.jumpstart:.3f}")
    print(f"  Asymptotic: {result.metrics.asymptotic_performance:.3f}")
    print()


def example_transfer_with_baseline():
    """
    Example: Transfer experiment with baseline comparison.
    """
    print("=" * 80)
    print("EXAMPLE 2: Transfer vs. Baseline Comparison")
    print("=" * 80)

    class MockPolicy:
        def __init__(self, name):
            self.name = name

    class MockEnv:
        def __init__(self, name):
            self.name = name

    # WHY: Create experiment
    experiment = TransferExperiment(
        random_seed=42,
        num_eval_episodes=20,
        performance_threshold=6.0
    )

    # WHY: Run transfer vs baseline
    source_policy = MockPolicy("echo_policy")
    baseline_policy = MockPolicy("random_policy")
    target_env = MockEnv("coding_env")

    transfer_result, baseline_result = experiment.run_transfer_with_baseline(
        source_policy=source_policy,
        source_task="echo",
        target_env=target_env,
        target_task="coding",
        baseline_policy=baseline_policy,
        max_steps_per_episode=100
    )

    print("\nFinal Comparison:")
    print(f"  Transfer: {transfer_result.metrics.asymptotic_performance:.3f}")
    print(f"  Baseline: {baseline_result.metrics.asymptotic_performance:.3f}")
    print(f"  Ratio: {transfer_result.metrics.transfer_ratio:.2f}x")
    print()


def example_transfer_matrix():
    """
    Example: Compute transfer matrix for multiple tasks.
    """
    print("=" * 80)
    print("EXAMPLE 3: Transfer Matrix Computation")
    print("=" * 80)

    class MockPolicy:
        def __init__(self, name):
            self.name = name

    class MockEnv:
        def __init__(self, name):
            self.name = name

    # WHY: Define tasks
    task_names = ["echo", "coding", "git", "browser"]

    # WHY: Create mock policies and environments
    policies = {task: MockPolicy(f"{task}_policy") for task in task_names}
    environments = {task: MockEnv(f"{task}_env") for task in task_names}
    baseline_policy = MockPolicy("random")

    # WHY: Create transfer matrix
    transfer_matrix = TransferMatrix(task_names)

    # WHY: Create experiment
    experiment = TransferExperiment(
        random_seed=42,
        num_eval_episodes=10,  # Fewer for demo
        performance_threshold=6.0
    )

    # WHY: Compute matrix (this will take time with real environments)
    print("\nNote: Using mock policies/environments for demonstration.")
    print("In real usage, this would evaluate actual transfer quality.\n")

    transfer_matrix.compute_matrix(
        policies=policies,
        environments=environments,
        experiment=experiment,
        baseline_policy=baseline_policy,
        metric="jumpstart"
    )

    # WHY: Print matrix
    transfer_matrix.print_matrix()

    # WHY: Find best source for each target
    print("\nBest Source Tasks:")
    for target in task_names:
        best_source = transfer_matrix.get_best_source_for_target(target)
        if best_source != target:
            value = transfer_matrix.matrix[
                task_names.index(best_source),
                task_names.index(target)
            ]
            print(f"  For {target}: {best_source} (quality={value:.3f})")

    print()


if __name__ == "__main__":
    """
    Run all zero-shot transfer experiment examples.

    To run this file:
        python Claude_tutorials/Zero_Shot_Transfer_Experiments.py

    To use in your own code:
        from Claude_tutorials.Zero_Shot_Transfer_Experiments import TransferExperiment
        experiment = TransferExperiment()
        result = experiment.run_zero_shot_transfer(policy, "source", env, "target")
    """
    example_zero_shot_transfer()
    example_transfer_with_baseline()
    example_transfer_matrix()

    print("=" * 80)
    print("All zero-shot transfer examples completed successfully!")
    print("=" * 80)
