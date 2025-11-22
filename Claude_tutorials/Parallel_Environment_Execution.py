"""
Parallel Environment Execution for OpenEnv

This module provides utilities for executing multiple OpenEnv environments in parallel,
enabling faster data collection, curriculum learning, and multi-environment training.

Key Patterns:
1. ThreadPoolExecutor: For I/O-bound OpenEnv environments (network calls)
2. AsyncIO: For async/await parallel execution
3. ProcessPoolExecutor: For CPU-bound environments (not recommended for OpenEnv due to pickling)
4. Parallel Rollout Collection: Gather experience from multiple environments simultaneously

Author: Claude
License: MIT
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Callable, Optional, Tuple
from queue import Queue
import numpy as np


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

@dataclass
class Rollout:
    """
    A single episode rollout containing the trajectory of an agent in an environment.

    Attributes:
        observations: List of observations seen during the episode
        actions: List of actions taken during the episode
        rewards: List of rewards received during the episode
        dones: List of done flags for each step
        episode_return: Total cumulative reward for the episode
        episode_length: Number of steps in the episode
        env_name: Name/ID of the environment that generated this rollout
    """
    observations: List[Any]
    actions: List[Any]
    rewards: List[float]
    dones: List[bool]
    episode_return: float
    episode_length: int
    env_name: str


@dataclass
class ParallelRolloutStats:
    """
    Aggregated statistics from multiple parallel rollouts.

    Useful for monitoring training progress and curriculum learning decisions.
    """
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    mean_length: float
    total_steps: int
    num_episodes: int
    env_breakdown: Dict[str, float]  # mean_return per environment


# =============================================================================
# PATTERN 1: THREAD-BASED PARALLEL EXECUTION
# =============================================================================

class ParallelEnvironmentExecutor:
    """
    Execute multiple OpenEnv environments in parallel using ThreadPoolExecutor.

    Why Use This:
    - OpenEnv environments make HTTP requests (I/O-bound), so threading is efficient
    - Python's GIL doesn't hurt performance for I/O-bound tasks
    - Thread pools reuse threads, reducing overhead
    - Easier to share memory between threads (vs. processes)

    When to Use:
    - Collecting rollouts from multiple environment instances
    - Running different agents on the same environment
    - Curriculum learning with multiple difficulty levels
    - Multi-task training across different environments

    Performance:
    - Can achieve ~4-8x speedup with 8 threads for OpenEnv environments
    - Limited by network latency, not CPU
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of parallel threads (default: 4)
                        For OpenEnv, 4-8 workers is typically optimal
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def collect_rollout(
        self,
        env_client,
        agent_fn: Callable,
        max_steps: int = 1000,
        env_name: str = "unknown"
    ) -> Rollout:
        """
        Collect a single rollout from an environment using the given agent.

        This function is designed to be run in parallel by ThreadPoolExecutor.
        Each thread will have its own environment client instance.

        Args:
            env_client: OpenEnv client instance (e.g., HttpEnvClient)
            agent_fn: Function that takes (observation, done) and returns action
            max_steps: Maximum steps per episode
            env_name: Identifier for this environment

        Returns:
            Rollout object containing the complete episode trajectory
        """
        # WHY: Store all data from the episode for later training
        observations = []
        actions = []
        rewards = []
        dones = []

        # WHY: Reset environment to start a fresh episode
        # Each thread gets its own episode_id from the server
        obs = env_client.reset()
        observations.append(obs)

        done = False
        step_count = 0
        episode_return = 0.0

        # WHY: Run episode until done or max_steps reached
        while not done and step_count < max_steps:
            # WHY: Get action from agent policy
            action = agent_fn(obs, done)
            actions.append(action)

            # WHY: Take step in environment
            result = env_client.step(action)

            # WHY: Extract data from StepResult
            obs = result.observation
            reward = result.reward if hasattr(result, 'reward') else 0.0
            done = result.done

            # WHY: Store step data
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)

            episode_return += reward
            step_count += 1

        # WHY: Return complete rollout for analysis/training
        return Rollout(
            observations=observations[:-1],  # Exclude final obs (no action taken)
            actions=actions,
            rewards=rewards,
            dones=dones,
            episode_return=episode_return,
            episode_length=step_count,
            env_name=env_name
        )

    def collect_rollouts_parallel(
        self,
        env_clients: List[Any],
        agent_fn: Callable,
        max_steps: int = 1000,
        env_names: Optional[List[str]] = None
    ) -> List[Rollout]:
        """
        Collect rollouts from multiple environments in parallel.

        This is the main entry point for parallel rollout collection.

        How it works:
        1. Submit all rollout collection tasks to thread pool
        2. Threads execute collect_rollout() concurrently
        3. Wait for all tasks to complete
        4. Return all rollouts

        Args:
            env_clients: List of OpenEnv client instances
            agent_fn: Agent policy function
            max_steps: Max steps per episode
            env_names: Optional names for each environment

        Returns:
            List of Rollout objects, one per environment
        """
        if env_names is None:
            env_names = [f"env_{i}" for i in range(len(env_clients))]

        # WHY: Submit all tasks to thread pool
        # as_completed() yields futures as they finish (not in submission order)
        futures = []
        for env_client, env_name in zip(env_clients, env_names):
            future = self.executor.submit(
                self.collect_rollout,
                env_client,
                agent_fn,
                max_steps,
                env_name
            )
            futures.append(future)

        # WHY: Collect results as they complete
        # This allows early processing of fast-completing episodes
        rollouts = []
        for future in as_completed(futures):
            try:
                rollout = future.result()
                rollouts.append(rollout)
            except Exception as e:
                print(f"Error collecting rollout: {e}")

        return rollouts

    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)


# =============================================================================
# PATTERN 2: ASYNC/AWAIT PARALLEL EXECUTION
# =============================================================================

class AsyncParallelExecutor:
    """
    Execute multiple OpenEnv environments in parallel using asyncio.

    Why Use This:
    - More scalable than threading for very high concurrency (50+ environments)
    - Lower memory overhead (no thread stacks)
    - Better control flow with async/await syntax
    - Efficient for I/O-bound operations

    When to Use:
    - Large-scale data collection (100+ parallel episodes)
    - Real-time multi-agent systems
    - When you need fine-grained control over concurrency

    Note: Requires async-compatible OpenEnv client (or use asyncio.to_thread)
    """

    async def collect_rollout_async(
        self,
        env_client,
        agent_fn: Callable,
        max_steps: int = 1000,
        env_name: str = "unknown"
    ) -> Rollout:
        """
        Async version of rollout collection.

        If env_client doesn't support async, we can use:
        - asyncio.to_thread() to run sync code in thread pool
        - Or create async wrapper around HttpEnvClient

        For demonstration, this uses asyncio.to_thread for compatibility.
        """
        observations = []
        actions = []
        rewards = []
        dones = []

        # WHY: to_thread() runs synchronous env_client.reset() in executor thread
        # This prevents blocking the event loop
        obs = await asyncio.to_thread(env_client.reset)
        observations.append(obs)

        done = False
        step_count = 0
        episode_return = 0.0

        while not done and step_count < max_steps:
            # WHY: Run agent function in thread pool if it's CPU-intensive
            action = await asyncio.to_thread(agent_fn, obs, done)
            actions.append(action)

            # WHY: Run step in thread pool (synchronous I/O)
            result = await asyncio.to_thread(env_client.step, action)

            obs = result.observation
            reward = result.reward if hasattr(result, 'reward') else 0.0
            done = result.done

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)

            episode_return += reward
            step_count += 1

        return Rollout(
            observations=observations[:-1],
            actions=actions,
            rewards=rewards,
            dones=dones,
            episode_return=episode_return,
            episode_length=step_count,
            env_name=env_name
        )

    async def collect_rollouts_async(
        self,
        env_clients: List[Any],
        agent_fn: Callable,
        max_steps: int = 1000,
        env_names: Optional[List[str]] = None
    ) -> List[Rollout]:
        """
        Collect rollouts using asyncio.gather for concurrent execution.

        How it works:
        1. Create coroutine for each environment
        2. asyncio.gather() runs all coroutines concurrently
        3. Returns results in submission order (unlike as_completed)

        Args:
            env_clients: List of environment clients
            agent_fn: Agent policy function
            max_steps: Max steps per episode
            env_names: Optional environment names

        Returns:
            List of rollouts in same order as env_clients
        """
        if env_names is None:
            env_names = [f"env_{i}" for i in range(len(env_clients))]

        # WHY: Create list of coroutines for all environments
        tasks = [
            self.collect_rollout_async(client, agent_fn, max_steps, name)
            for client, name in zip(env_clients, env_names)
        ]

        # WHY: gather() runs all tasks concurrently and waits for all to complete
        # return_exceptions=True means exceptions are returned, not raised
        rollouts = await asyncio.gather(*tasks, return_exceptions=True)

        # WHY: Filter out any exceptions that occurred
        valid_rollouts = [r for r in rollouts if isinstance(r, Rollout)]

        return valid_rollouts


# =============================================================================
# PATTERN 3: PARALLEL ROLLOUT COLLECTION WITH STATISTICS
# =============================================================================

class ParallelRolloutCollector:
    """
    High-level interface for parallel rollout collection with statistics.

    Why Use This:
    - Combines rollout collection with statistical analysis
    - Useful for curriculum learning (track progress per environment)
    - Provides aggregated metrics for training monitoring
    - Handles both threading and async execution
    """

    def __init__(self, execution_mode: str = "thread", max_workers: int = 4):
        """
        Initialize the rollout collector.

        Args:
            execution_mode: "thread" or "async"
            max_workers: Number of parallel workers (for thread mode)
        """
        self.execution_mode = execution_mode
        if execution_mode == "thread":
            self.executor = ParallelEnvironmentExecutor(max_workers)
        elif execution_mode == "async":
            self.executor = AsyncParallelExecutor()
        else:
            raise ValueError(f"Unknown execution_mode: {execution_mode}")

    def collect_with_stats(
        self,
        env_clients: List[Any],
        agent_fn: Callable,
        max_steps: int = 1000,
        env_names: Optional[List[str]] = None
    ) -> Tuple[List[Rollout], ParallelRolloutStats]:
        """
        Collect rollouts and compute statistics.

        Returns:
            Tuple of (rollouts, stats)
        """
        # WHY: Collect rollouts using appropriate executor
        if self.execution_mode == "thread":
            rollouts = self.executor.collect_rollouts_parallel(
                env_clients, agent_fn, max_steps, env_names
            )
        elif self.execution_mode == "async":
            # WHY: Run async function in event loop
            rollouts = asyncio.run(
                self.executor.collect_rollouts_async(
                    env_clients, agent_fn, max_steps, env_names
                )
            )

        # WHY: Compute statistics for monitoring and curriculum learning
        stats = self._compute_stats(rollouts)

        return rollouts, stats

    def _compute_stats(self, rollouts: List[Rollout]) -> ParallelRolloutStats:
        """
        Compute aggregated statistics from rollouts.

        These statistics are useful for:
        - Monitoring training progress
        - Curriculum learning decisions
        - Multi-task balancing
        """
        if not rollouts:
            return ParallelRolloutStats(
                mean_return=0.0, std_return=0.0, min_return=0.0,
                max_return=0.0, mean_length=0.0, total_steps=0,
                num_episodes=0, env_breakdown={}
            )

        # WHY: Extract returns for statistical analysis
        returns = [r.episode_return for r in rollouts]
        lengths = [r.episode_length for r in rollouts]

        # WHY: Compute per-environment statistics for curriculum learning
        env_breakdown = {}
        env_rollouts = {}
        for rollout in rollouts:
            if rollout.env_name not in env_rollouts:
                env_rollouts[rollout.env_name] = []
            env_rollouts[rollout.env_name].append(rollout.episode_return)

        for env_name, env_returns in env_rollouts.items():
            env_breakdown[env_name] = np.mean(env_returns)

        return ParallelRolloutStats(
            mean_return=np.mean(returns),
            std_return=np.std(returns),
            min_return=np.min(returns),
            max_return=np.max(returns),
            mean_length=np.mean(lengths),
            total_steps=sum(lengths),
            num_episodes=len(rollouts),
            env_breakdown=env_breakdown
        )

    def shutdown(self):
        """Cleanup resources."""
        if self.execution_mode == "thread":
            self.executor.shutdown()


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_random_agent(observation, done):
    """
    Example random agent for demonstration.

    In real usage, replace this with your trained policy.
    """
    import random
    # WHY: Simple random policy for testing
    # Replace with: return policy(observation)
    return f"random_action_{random.randint(0, 10)}"


def example_thread_based_collection():
    """
    Example: Collect rollouts from multiple environments using threading.

    This is the recommended approach for OpenEnv environments.
    """
    print("=" * 80)
    print("EXAMPLE 1: Thread-based Parallel Rollout Collection")
    print("=" * 80)

    # WHY: This example shows how to collect data from multiple environments in parallel
    # In real usage, replace mock clients with actual HttpEnvClient instances

    # Step 1: Create multiple environment clients
    # In real code:
    # from core.http_env_client import HttpEnvClient
    # env_clients = [
    #     HttpEnvClient("http://localhost:8000"),
    #     HttpEnvClient("http://localhost:8001"),
    #     HttpEnvClient("http://localhost:8002"),
    #     HttpEnvClient("http://localhost:8003"),
    # ]

    # For demonstration, using mock objects
    class MockEnvClient:
        def __init__(self, env_id):
            self.env_id = env_id
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return f"Initial observation from env {self.env_id}"

        def step(self, action):
            self.step_count += 1
            from dataclasses import dataclass

            @dataclass
            class MockResult:
                observation: str
                reward: float
                done: bool

            # WHY: Simulate varying episode lengths
            done = self.step_count >= (10 + self.env_id * 2)
            reward = 1.0 if not done else 10.0

            return MockResult(
                observation=f"Obs {self.step_count} from env {self.env_id}",
                reward=reward,
                done=done
            )

    env_clients = [MockEnvClient(i) for i in range(4)]
    env_names = ["easy", "medium", "hard", "expert"]

    # Step 2: Create parallel executor
    executor = ParallelEnvironmentExecutor(max_workers=4)

    # Step 3: Collect rollouts in parallel
    start_time = time.time()
    rollouts = executor.collect_rollouts_parallel(
        env_clients=env_clients,
        agent_fn=example_random_agent,
        max_steps=100,
        env_names=env_names
    )
    elapsed = time.time() - start_time

    # Step 4: Analyze results
    print(f"\nCollected {len(rollouts)} rollouts in {elapsed:.2f} seconds")
    print("\nRollout Summary:")
    for rollout in rollouts:
        print(f"  {rollout.env_name}: "
              f"return={rollout.episode_return:.2f}, "
              f"length={rollout.episode_length}")

    # Step 5: Cleanup
    executor.shutdown()
    print("\nThread pool shutdown complete\n")


def example_async_collection():
    """
    Example: Collect rollouts using async/await.

    Use this for very high concurrency (50+ environments).
    """
    print("=" * 80)
    print("EXAMPLE 2: Async-based Parallel Rollout Collection")
    print("=" * 80)

    # WHY: Reuse mock client from previous example
    class MockEnvClient:
        def __init__(self, env_id):
            self.env_id = env_id
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return f"Initial observation from env {self.env_id}"

        def step(self, action):
            self.step_count += 1
            from dataclasses import dataclass

            @dataclass
            class MockResult:
                observation: str
                reward: float
                done: bool

            done = self.step_count >= (10 + self.env_id * 2)
            reward = 1.0 if not done else 10.0

            return MockResult(
                observation=f"Obs {self.step_count} from env {self.env_id}",
                reward=reward,
                done=done
            )

    # WHY: Create more environments to show async scalability
    env_clients = [MockEnvClient(i) for i in range(8)]
    env_names = [f"env_{i}" for i in range(8)]

    # Step 2: Create async executor
    executor = AsyncParallelExecutor()

    # Step 3: Collect rollouts asynchronously
    start_time = time.time()
    rollouts = asyncio.run(
        executor.collect_rollouts_async(
            env_clients=env_clients,
            agent_fn=example_random_agent,
            max_steps=100,
            env_names=env_names
        )
    )
    elapsed = time.time() - start_time

    # Step 4: Analyze results
    print(f"\nCollected {len(rollouts)} rollouts in {elapsed:.2f} seconds")
    print("\nRollout Summary:")
    for rollout in rollouts:
        print(f"  {rollout.env_name}: "
              f"return={rollout.episode_return:.2f}, "
              f"length={rollout.episode_length}")
    print()


def example_collection_with_stats():
    """
    Example: High-level rollout collection with statistics.

    This is the recommended approach for most use cases.
    """
    print("=" * 80)
    print("EXAMPLE 3: Rollout Collection with Statistics")
    print("=" * 80)

    # WHY: Reuse mock client
    class MockEnvClient:
        def __init__(self, env_id):
            self.env_id = env_id
            self.step_count = 0

        def reset(self):
            self.step_count = 0
            return f"Initial observation from env {self.env_id}"

        def step(self, action):
            self.step_count += 1
            from dataclasses import dataclass

            @dataclass
            class MockResult:
                observation: str
                reward: float
                done: bool

            done = self.step_count >= (10 + self.env_id * 2)
            reward = 1.0 if not done else 10.0

            return MockResult(
                observation=f"Obs {self.step_count} from env {self.env_id}",
                reward=reward,
                done=done
            )

    # WHY: Create environments with different difficulties
    env_clients = [MockEnvClient(i) for i in range(4)]
    env_names = ["easy", "medium", "hard", "expert"]

    # Step 2: Create high-level collector
    collector = ParallelRolloutCollector(
        execution_mode="thread",
        max_workers=4
    )

    # Step 3: Collect with statistics
    rollouts, stats = collector.collect_with_stats(
        env_clients=env_clients,
        agent_fn=example_random_agent,
        max_steps=100,
        env_names=env_names
    )

    # Step 4: Print detailed statistics
    print(f"\nCollected {stats.num_episodes} episodes")
    print(f"Total steps: {stats.total_steps}")
    print(f"Mean return: {stats.mean_return:.2f} ± {stats.std_return:.2f}")
    print(f"Return range: [{stats.min_return:.2f}, {stats.max_return:.2f}]")
    print(f"Mean episode length: {stats.mean_length:.2f}")

    print("\nPer-Environment Performance:")
    for env_name, mean_return in stats.env_breakdown.items():
        print(f"  {env_name}: {mean_return:.2f}")

    # Step 5: Use stats for curriculum learning decisions
    print("\nCurriculum Learning Decision:")
    if stats.env_breakdown.get("easy", 0) > 15.0:
        print("  ✓ Agent has mastered 'easy' environment")
        print("  → Recommend increasing 'medium' environment proportion")
    else:
        print("  ✗ Agent still learning 'easy' environment")
        print("  → Continue training on current curriculum")

    # Step 6: Cleanup
    collector.shutdown()
    print()


if __name__ == "__main__":
    """
    Run all examples to demonstrate parallel execution patterns.

    To run this file:
        python Claude_tutorials/Parallel_Environment_Execution.py

    To use in your own code:
        from Claude_tutorials.Parallel_Environment_Execution import ParallelRolloutCollector
        collector = ParallelRolloutCollector(execution_mode="thread", max_workers=8)
        rollouts, stats = collector.collect_with_stats(env_clients, agent_fn)
    """
    example_thread_based_collection()
    example_async_collection()
    example_collection_with_stats()

    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
