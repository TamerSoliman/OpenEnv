"""
================================================================================
OpenEnv to Gymnasium Wrapper - Universal Adapter
================================================================================

This module provides a wrapper that converts ANY OpenEnv environment into a
Gymnasium-compatible environment, enabling integration with the entire
ecosystem of Gymnasium-based RL libraries.

WHAT: Adapter class that implements the Gymnasium API using OpenEnv environments
WHY: Enables use of OpenEnv envs with Stable-Baselines3, CleanRL, RLlib, etc.
HOW: Wraps OpenEnv HTTP client and translates between OpenEnv and Gym interfaces

COMPATIBILITY:
- Stable-Baselines3 (PPO, SAC, DQN, etc.)
- CleanRL implementations
- Ray RLlib
- TorchRL
- Any library supporting Gymnasium environments

SOURCE: Based on Gymnasium API specification
        https://gymnasium.farama.org/api/env/

Author: Claude (Anthropic)
License: Same as OpenEnv
================================================================================
"""

from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# OpenEnv imports
from core.http_env_client import HTTPEnvClient
from core.client_types import StepResult


# ==============================================================================
# MAIN WRAPPER CLASS
# ==============================================================================

class OpenEnvGymnasiumWrapper(gym.Env):
    """
    Wrapper that converts an OpenEnv environment to Gymnasium-compatible format.

    This wrapper handles the translation between:
    - OpenEnv's text-based actions → Gymnasium's action space
    - OpenEnv's structured observations → Gymnasium's observation space
    - OpenEnv's StepResult format → Gymnasium's (obs, reward, terminated, truncated, info)

    USAGE:
        >>> from envs.echo_env import EchoEnv
        >>> openenv_client = EchoEnv.from_docker_image("echo-env:latest")
        >>> gym_env = OpenEnvGymnasiumWrapper(
        ...     openenv_client,
        ...     observation_mode="text",
        ...     max_episode_steps=100
        ... )
        >>> obs, info = gym_env.reset()
        >>> action = gym_env.action_space.sample()
        >>> obs, reward, terminated, truncated, info = gym_env.step(action)

    INTEGRATION EXAMPLE (Stable-Baselines3):
        >>> from stable_baselines3 import PPO
        >>> model = PPO("MlpPolicy", gym_env, verbose=1)
        >>> model.learn(total_timesteps=10000)

    Args:
        openenv_client: An initialized OpenEnv HTTP client
        observation_mode: How to represent observations ("text", "dict", "array")
        max_episode_steps: Maximum steps per episode (for truncation)
        action_mode: How to handle actions ("text", "discrete")
    """

    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    def __init__(
        self,
        openenv_client: HTTPEnvClient,
        observation_mode: str = "text",
        max_episode_steps: int = 1000,
        action_mode: str = "text",
        discrete_actions: Optional[list] = None,
    ):
        """
        Initialize the Gymnasium wrapper around an OpenEnv client.

        OBSERVATION MODES:
        - "text": Observation is a string (uses Gym's Text space)
        - "dict": Observation is a dictionary (uses Gym's Dict space)
        - "array": Observation converted to numpy array (uses Gym's Box space)

        ACTION MODES:
        - "text": Actions are text strings (uses Gym's Text space)
        - "discrete": Actions are discrete indices into action list (uses Discrete space)

        DESIGN DECISION: We default to "text" mode because OpenEnv environments
        are primarily text-based (LLM agents). For traditional RL algorithms
        that need numeric spaces, use "discrete" action mode.

        Args:
            openenv_client: Initialized OpenEnv client (e.g., EchoEnv, Connect4Env)
            observation_mode: How to represent observations
            max_episode_steps: Maximum steps before truncation
            action_mode: How to handle actions
            discrete_actions: List of valid actions (required if action_mode="discrete")
        """
        super().__init__()

        # Store OpenEnv client
        self.openenv_client = openenv_client
        self.observation_mode = observation_mode
        self.max_episode_steps = max_episode_steps
        self.action_mode = action_mode
        self.discrete_actions = discrete_actions

        # Episode tracking
        self._current_step = 0
        self._last_observation = None

        # Define action space based on mode
        if action_mode == "text":
            # Text-based actions (unbounded string)
            # NOTE: Gymnasium's Text space is for observation, not actions
            # For text actions with standard RL algorithms, use discrete mode
            self.action_space = spaces.Text(max_length=1000)
        elif action_mode == "discrete":
            # Discrete action space (integer indices)
            if discrete_actions is None:
                raise ValueError("discrete_actions list required for discrete action mode")
            self.action_space = spaces.Discrete(len(discrete_actions))
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        # Define observation space based on mode
        # NOTE: We set a generic space here. For environment-specific spaces,
        # subclass this wrapper and override _get_observation_space()
        if observation_mode == "text":
            # Text observation (variable length)
            self.observation_space = spaces.Text(max_length=10000)
        elif observation_mode == "dict":
            # Dictionary observation (requires environment-specific definition)
            # Default: assume reward and done fields
            self.observation_space = spaces.Dict({
                "reward": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "done": spaces.Discrete(2),  # 0 or 1
            })
        elif observation_mode == "array":
            # Array observation (requires environment-specific definition)
            # Default: 1D array of floats
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32
            )
        else:
            raise ValueError(f"Unknown observation_mode: {observation_mode}")

    # -------------------------------------------------------------------------
    # GYMNASIUM API: reset()
    # -------------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        GYMNASIUM CONTRACT:
        - Returns: (observation, info)
        - observation: First observation of the new episode
        - info: Dictionary with auxiliary information

        OPENENV TRANSLATION:
        1. Call openenv_client.reset()
        2. Extract observation from StepResult
        3. Convert to Gymnasium format based on observation_mode
        4. Return (obs, info)

        Args:
            seed: Random seed for reproducibility
            options: Additional options (passed to OpenEnv if supported)

        Returns:
            observation: Initial observation
            info: Auxiliary information dictionary
        """
        # Set random seed if provided
        if seed is not None:
            super().reset(seed=seed)

        # Reset episode counter
        self._current_step = 0

        # Call OpenEnv reset
        # NOTE: OpenEnv's reset() returns StepResult[Observation]
        step_result = self.openenv_client.reset()

        # Store last observation
        self._last_observation = step_result.observation

        # Convert observation to Gymnasium format
        gym_observation = self._convert_observation(step_result.observation)

        # Build info dictionary
        info = {
            "openenv_observation": step_result.observation,
            "openenv_reward": step_result.reward,
            "openenv_done": step_result.done,
        }

        return gym_observation, info

    # -------------------------------------------------------------------------
    # GYMNASIUM API: step()
    # -------------------------------------------------------------------------

    def step(
        self,
        action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        GYMNASIUM CONTRACT:
        - Input: action (type depends on action_space)
        - Returns: (observation, reward, terminated, truncated, info)
        - terminated: Episode ended due to environment dynamics (e.g., goal reached)
        - truncated: Episode ended due to time limit

        OPENENV TRANSLATION:
        1. Convert Gymnasium action to OpenEnv action
        2. Call openenv_client.step(action)
        3. Extract observation, reward, done from StepResult
        4. Map done flag to terminated/truncated
        5. Return Gymnasium tuple

        Args:
            action: Action to execute (format depends on action_mode)

        Returns:
            observation: New observation after action
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode ended due to time limit
            info: Auxiliary information
        """
        # Increment step counter
        self._current_step += 1

        # Convert Gymnasium action to OpenEnv action
        openenv_action = self._convert_action(action)

        # Execute action in OpenEnv
        step_result = self.openenv_client.step(openenv_action)

        # Store last observation
        self._last_observation = step_result.observation

        # Convert observation to Gymnasium format
        gym_observation = self._convert_observation(step_result.observation)

        # Extract reward (default to 0.0 if None)
        reward = step_result.reward if step_result.reward is not None else 0.0

        # Determine termination reason
        # - terminated: Environment says episode is done
        # - truncated: We hit max_episode_steps
        terminated = step_result.done
        truncated = self._current_step >= self.max_episode_steps

        # Build info dictionary
        info = {
            "openenv_observation": step_result.observation,
            "openenv_action": openenv_action,
            "step_count": self._current_step,
        }

        return gym_observation, reward, terminated, truncated, info

    # -------------------------------------------------------------------------
    # GYMNASIUM API: render() (optional)
    # -------------------------------------------------------------------------

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment (optional).

        IMPLEMENTATION: Most OpenEnv environments are text-based, so we
        return a string representation of the current observation.

        For visual environments (e.g., BrowserGym with screenshots),
        override this method in a subclass.

        Returns:
            String representation of current state
        """
        if self._last_observation is None:
            return "No observation yet - call reset() first"

        # Return string representation
        return str(self._last_observation)

    # -------------------------------------------------------------------------
    # GYMNASIUM API: close()
    # -------------------------------------------------------------------------

    def close(self):
        """
        Clean up environment resources.

        OPENENV TRANSLATION:
        Call openenv_client.close() to stop Docker container and cleanup.
        """
        if hasattr(self.openenv_client, 'close'):
            self.openenv_client.close()

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS: Action Conversion
    # -------------------------------------------------------------------------

    def _convert_action(self, gym_action: Any) -> Any:
        """
        Convert Gymnasium action to OpenEnv action format.

        ACTION TRANSLATION LOGIC:
        - text mode: action is already a string, pass through
        - discrete mode: action is int index, map to action from list

        Args:
            gym_action: Action from Gymnasium (int or str)

        Returns:
            OpenEnv action object

        IMPORTANT: This is a generic implementation. Environment-specific
        wrappers should override this to create the correct Action subclass.
        """
        if self.action_mode == "text":
            # Action is already text
            action_str = gym_action
        elif self.action_mode == "discrete":
            # Map discrete index to action string
            action_str = self.discrete_actions[gym_action]
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

        # For generic wrapper, we just return the string
        # Subclasses should create the appropriate Action object
        # e.g., EchoAction(message=action_str)
        return action_str

    # -------------------------------------------------------------------------
    # INTERNAL HELPERS: Observation Conversion
    # -------------------------------------------------------------------------

    def _convert_observation(self, openenv_obs: Any) -> Any:
        """
        Convert OpenEnv observation to Gymnasium observation format.

        OBSERVATION TRANSLATION LOGIC:
        - text mode: convert observation to string
        - dict mode: convert observation to dictionary
        - array mode: convert observation to numpy array

        Args:
            openenv_obs: OpenEnv observation object

        Returns:
            Gymnasium-compatible observation

        IMPORTANT: This is a generic implementation. For environment-specific
        conversions, override this method in a subclass.
        """
        if self.observation_mode == "text":
            # Convert observation to text
            return str(openenv_obs)

        elif self.observation_mode == "dict":
            # Convert to dictionary
            # Default: use observation's __dict__ if available
            if hasattr(openenv_obs, '__dict__'):
                obs_dict = openenv_obs.__dict__.copy()
                # Remove metadata to keep it simple
                obs_dict.pop('metadata', None)
                return obs_dict
            else:
                return {"observation": str(openenv_obs)}

        elif self.observation_mode == "array":
            # Convert to numpy array
            # Default: very simple encoding (hash of string)
            # Override this for meaningful array conversions
            obs_str = str(openenv_obs)
            # Simple encoding: character codes padded/truncated to fixed size
            char_codes = [ord(c) for c in obs_str[:100]]
            # Pad to 100 elements
            char_codes.extend([0] * (100 - len(char_codes)))
            return np.array(char_codes, dtype=np.float32)

        else:
            raise ValueError(f"Unknown observation_mode: {self.observation_mode}")


# ==============================================================================
# ENVIRONMENT-SPECIFIC WRAPPER EXAMPLE: EchoEnv
# ==============================================================================

class EchoEnvGymnasiumWrapper(OpenEnvGymnasiumWrapper):
    """
    Specialized Gymnasium wrapper for EchoEnv.

    This demonstrates how to create environment-specific wrappers that
    properly handle action/observation conversion.

    IMPROVEMENTS OVER GENERIC WRAPPER:
    - Creates proper EchoAction objects
    - Extracts meaningful fields from EchoObservation
    - Defines appropriate observation space

    USAGE:
        >>> from envs.echo_env import EchoEnv
        >>> client = EchoEnv.from_docker_image("echo-env:latest")
        >>> gym_env = EchoEnvGymnasiumWrapper(client)
        >>> obs, info = gym_env.reset()
    """

    def __init__(self, openenv_client, max_episode_steps: int = 100):
        """
        Initialize EchoEnv Gymnasium wrapper.

        Args:
            openenv_client: EchoEnv client instance
            max_episode_steps: Maximum steps per episode
        """
        # Define discrete actions for Echo environment
        discrete_actions = [
            "Hello",
            "Test message",
            "Echo this",
            "Longer message with more text",
            "Short",
        ]

        # Initialize parent with dict observation mode
        super().__init__(
            openenv_client=openenv_client,
            observation_mode="dict",
            max_episode_steps=max_episode_steps,
            action_mode="discrete",
            discrete_actions=discrete_actions,
        )

        # Define Echo-specific observation space
        self.observation_space = spaces.Dict({
            "echoed_message": spaces.Text(max_length=1000),
            "message_length": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
        })

    def _convert_action(self, gym_action: int):
        """
        Convert discrete action to EchoAction.

        Args:
            gym_action: Integer index into discrete_actions

        Returns:
            EchoAction object
        """
        from envs.echo_env import EchoAction

        # Get action string from discrete action list
        action_str = self.discrete_actions[gym_action]

        # Create EchoAction object
        return EchoAction(message=action_str)

    def _convert_observation(self, openenv_obs):
        """
        Convert EchoObservation to Gymnasium dict.

        Args:
            openenv_obs: EchoObservation object

        Returns:
            Dictionary with echoed_message and message_length
        """
        return {
            "echoed_message": openenv_obs.echoed_message,
            "message_length": np.array([openenv_obs.message_length], dtype=np.int32),
        }


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

def example_basic_usage():
    """
    Example 1: Basic usage with Echo environment.

    DEMONSTRATES:
    - Creating wrapped environment
    - Standard Gymnasium interaction loop
    - Accessing OpenEnv-specific info
    """
    from envs.echo_env import EchoEnv

    print("="*60)
    print("Example 1: Basic Gymnasium Wrapper Usage")
    print("="*60)

    # Create OpenEnv client
    print("\n1. Creating OpenEnv client...")
    client = EchoEnv.from_docker_image("echo-env:latest")

    # Wrap with Gymnasium interface
    print("2. Wrapping with Gymnasium interface...")
    gym_env = EchoEnvGymnasiumWrapper(client, max_episode_steps=10)

    # Reset
    print("\n3. Resetting environment...")
    observation, info = gym_env.reset()
    print(f"   Initial observation: {observation}")
    print(f"   Info: {info}")

    # Interaction loop
    print("\n4. Running episode...")
    for step in range(5):
        # Sample random action
        action = gym_env.action_space.sample()
        print(f"\n   Step {step + 1}: Action = {action} ({gym_env.discrete_actions[action]})")

        # Execute action
        obs, reward, terminated, truncated, info = gym_env.step(action)

        print(f"   Observation: {obs}")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}, Truncated: {truncated}")

        if terminated or truncated:
            print("   Episode ended!")
            break

    # Cleanup
    print("\n5. Closing environment...")
    gym_env.close()
    print("Done!")


def example_stable_baselines3_integration():
    """
    Example 2: Integration with Stable-Baselines3.

    DEMONSTRATES:
    - Using wrapped env with SB3 PPO algorithm
    - Training loop
    - Model evaluation

    NOTE: This example requires stable-baselines3:
        pip install stable-baselines3
    """
    from envs.echo_env import EchoEnv

    print("="*60)
    print("Example 2: Stable-Baselines3 Integration")
    print("="*60)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        print("\nStable-Baselines3 not installed!")
        print("Install with: pip install stable-baselines3")
        return

    # Create and wrap environment
    print("\n1. Creating wrapped environment...")
    client = EchoEnv.from_docker_image("echo-env:latest")
    gym_env = EchoEnvGymnasiumWrapper(client, max_episode_steps=20)

    # Check if environment follows Gym API
    print("2. Checking environment compatibility...")
    try:
        check_env(gym_env)
        print("   ✓ Environment is Gym-compatible!")
    except Exception as e:
        print(f"   ✗ Environment check failed: {e}")

    # Create PPO model
    print("\n3. Creating PPO model...")
    model = PPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for Dict observations
        gym_env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=32,
    )

    # Train model
    print("\n4. Training model...")
    model.learn(total_timesteps=1000)

    # Evaluate
    print("\n5. Evaluating model...")
    obs, info = gym_env.reset()
    for _ in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = gym_env.step(action)
        print(f"   Action: {action}, Reward: {reward}")
        if terminated or truncated:
            break

    # Cleanup
    gym_env.close()
    print("\nDone!")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("OpenEnv Gymnasium Wrapper Examples\n")

    # Run basic example
    example_basic_usage()

    print("\n" + "="*60 + "\n")

    # Run SB3 example
    # example_stable_baselines3_integration()  # Uncomment if SB3 installed
