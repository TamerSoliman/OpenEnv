"""
================================================================================
HEAVILY ANNOTATED: Environment Base Class (interfaces.py)
================================================================================

This file contains the CORE abstraction for all OpenEnv environments.
Understanding this class is essential to implementing custom environments.

WHAT: Defines the Gymnasium-style API contract that all environments must follow
HOW: Uses abstract base classes (ABC) to enforce implementation of core methods
WHY: Provides a standardized interface for agent-environment interaction
     enabling ANY agent to work with ANY environment

Source: /home/user/OpenEnv/src/core/env_server/interfaces.py
================================================================================
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypedDict

from .types import Action, Observation, State


# ==============================================================================
# SECTION 1: COMMUNICATION TYPES (Message & ModelTokenizer)
# ==============================================================================
# WHAT: Data structures for LLM-based agent communication
# WHY: Many environments in OpenEnv involve LLM agents that need to format
#      their inputs/outputs as conversational messages

class Message(TypedDict):
    """
    A message in a conversation.

    WHAT: Represents a single turn in a multi-turn conversation
    STRUCTURE:
        - role: str     → Who is speaking ("user", "assistant", "system")
        - content: str  → What they said

    WHY: Compatible with Huggingface chat template format, allowing seamless
         integration with pre-trained LLMs and their tokenizers

    EXAMPLE:
        {"role": "user", "content": "Solve this math problem: 2+2"}
        {"role": "assistant", "content": "The answer is 4"}
    """
    role: str
    content: str


class ModelTokenizer(Protocol):
    """
    Protocol for tokenizers that support chat templates.

    WHAT: An interface (Protocol) that defines what methods a tokenizer must have
    HOW: Uses Python's Protocol type to define a structural contract
    WHY: Allows OpenEnv to work with ANY tokenizer that implements these methods,
         without tight coupling to specific tokenizer implementations

    COMPATIBILITY: Works with Huggingface transformers tokenizers out of the box
    """

    def apply_chat_template(
        self,
        conversation: list[Message],
        tokenize: bool = True,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Apply a chat template to format and optionally tokenize a conversation.

        WHAT: Converts a list of Message dicts into model-ready input
        HOW:
            1. Takes conversation history as list[Message]
            2. Applies model-specific formatting (e.g., ChatML, Llama format)
            3. Optionally tokenizes the formatted text

        WHY: Different LLMs expect different conversation formats. This
             abstraction lets environments work with any LLM's format.

        Args:
            conversation: List of message dicts with 'role' and 'content'
            tokenize: Whether to tokenize the output (True = return token IDs)
            return_tensors: Format for returned tensors ('pt' for PyTorch)
            **kwargs: Model-specific arguments

        Returns:
            Formatted and optionally tokenized conversation
            - If tokenize=True: tensor of token IDs
            - If tokenize=False: formatted string
        """
        ...

    def decode(
        self, token_ids: Any, skip_special_tokens: bool = False, **kwargs: Any
    ) -> str:
        """
        Decode token IDs back to text.

        WHAT: Converts model output (token IDs) back to human-readable text
        HOW: Uses the tokenizer's vocabulary to map IDs → strings
        WHY: Agents output token IDs; environments need text to process actions

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens (e.g., <PAD>, <EOS>)
            **kwargs: Additional arguments

        Returns:
            Decoded text string
        """
        ...


# ==============================================================================
# SECTION 2: OBSERVATION TRANSFORMS
# ==============================================================================
# WHAT: Optional transformations applied to observations AFTER environment step
# WHY: Allows reward shaping, observation augmentation, and metric computation
#      without modifying the core environment logic

class Transform(ABC):
    """
    Transform observations to add rewards, metrics, or other modifications.

    WHAT: A post-processing layer that modifies observations after step()
    HOW: Takes an Observation, modifies it, returns the modified version
    WHY: Enables reward engineering and observation augmentation independently
         from the base environment logic

    DESIGN PATTERN: Follows the TorchRL Transform pattern

    USE CASES:
        1. Reward Shaping: Modify reward values based on domain knowledge
        2. Observation Augmentation: Add computed features to observations
        3. Metric Tracking: Inject custom metrics into observation metadata
        4. Multi-objective rewards: Combine multiple reward signals

    EXAMPLE:
        class RewardScalingTransform(Transform):
            def __call__(self, obs: Observation) -> Observation:
                # Scale rewards to be in [0, 1] range
                if obs.reward is not None:
                    obs.reward = obs.reward / 100.0
                return obs
    """

    @abstractmethod
    def __call__(self, observation: Observation) -> Observation:
        """
        Transform an observation.

        WHAT: The transformation function applied to each observation
        HOW: Receives observation from environment, modifies and returns it
        WHY: Called automatically after env.step() if transform is registered

        Args:
            observation: The raw observation from the environment

        Returns:
            The transformed observation (same type, potentially different values)

        IMPORTANT: Must return the SAME type of Observation (don't change schema)
        """
        pass


# ==============================================================================
# SECTION 3: ENVIRONMENT BASE CLASS - THE CORE ABSTRACTION
# ==============================================================================
# WHAT: The abstract base class that ALL OpenEnv environments must inherit from
# WHY: Enforces a standard interface (Gymnasium-style) for agent-environment
#      interaction, enabling interoperability

class Environment(ABC):
    """
    Base class for all environment servers following Gym/Gymnasium API.

    ┌─────────────────────────────────────────────────────────────────┐
    │                    GYMNASIUM API PATTERN                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  1. reset() → Observation                                        │
    │     - Initializes a new episode                                  │
    │     - Returns the initial state observation                      │
    │     - Creates new episode_id, sets step_count=0                  │
    │                                                                  │
    │  2. step(action) → Observation                                   │
    │     - Executes the agent's action                                │
    │     - Updates internal state                                     │
    │     - Returns: (observation, reward, done, metadata)             │
    │     - Increments step_count                                      │
    │                                                                  │
    │  3. state → State (property)                                     │
    │     - Returns episode metadata (episode_id, step_count)          │
    │     - Does NOT modify state (read-only property)                 │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

    WHAT THIS CLASS DOES:
        - Defines the contract that all environments must follow
        - Provides optional transform support for reward shaping
        - Enforces implementation of reset(), step(), and state

    HOW IT WORKS:
        1. Subclass Environment (e.g., class MyEnv(Environment))
        2. Implement reset(), step(), and state property
        3. HTTPEnvServer wraps your environment and exposes it via HTTP
        4. HTTPEnvClient talks to the server from agent code

    WHY THIS DESIGN:
        - STANDARDIZATION: Every environment has the same interface
        - COMPOSABILITY: Transforms can be added without modifying env code
        - TESTABILITY: Easy to mock and test individual components
        - SCALABILITY: HTTP wrapper enables distributed agent-env setups

    Args:
        transform: Optional Transform to apply to observations after step()
                   If provided, EVERY observation will pass through transform
    """

    def __init__(self, transform: Transform | None = None):
        """
        Initialize the environment with an optional observation transform.

        WHAT: Constructor that stores the transform (if any)
        HOW: Simply saves the transform reference for later use in _apply_transform()
        WHY: Allows environments to be created with or without reward shaping

        Args:
            transform: Optional Transform instance to modify observations
                       Set to None (default) if no transformation needed

        EXAMPLE USAGE:
            # Without transform
            env = MyEnvironment()

            # With transform for reward scaling
            env = MyEnvironment(transform=RewardScalingTransform())
        """
        self.transform = transform

    # ==========================================================================
    # METHOD 1: reset() - START A NEW EPISODE
    # ==========================================================================

    @abstractmethod
    def reset(self) -> Observation:
        """
        Reset the environment and return initial observation.

        ┌──────────────────────────────────────────────────────────────┐
        │                    RESET LIFECYCLE                            │
        ├──────────────────────────────────────────────────────────────┤
        │                                                               │
        │  1. Generate new episode_id (typically uuid4())               │
        │  2. Set step_count = 0                                        │
        │  3. Reset environment-specific state (e.g., game board)       │
        │  4. Create initial Observation                                │
        │  5. Return observation (with done=False, reward=0.0)          │
        │                                                               │
        └──────────────────────────────────────────────────────────────┘

        WHAT: Initializes a new episode of interaction

        HOW:
            - Generates a unique episode_id to track this episode
            - Resets all episode-specific state to initial conditions
            - Returns the initial observation the agent sees

        WHY:
            - Enables multi-episode training/evaluation
            - Provides clean separation between episodes
            - Ensures reproducibility (especially with seeding)

        WHEN CALLED:
            - At the start of agent training/evaluation
            - After an episode terminates (done=True)
            - When the agent wants to start fresh

        OBSERVATION STRUCTURE:
            The returned Observation MUST be a subclass of core.env_server.types.Observation

            Base fields (inherited from Observation):
                - done: bool = False         (always False after reset)
                - reward: float | None = 0.0 (typically 0 for initial state)
                - metadata: dict = {}        (optional context)

            Custom fields (defined by your environment):
                e.g., for EchoEnvironment:
                    - echoed_message: str
                    - message_length: int

        Returns:
            Observation: Initial state observation
                        - done is always False (episode just started)
                        - reward is typically 0.0 (no action taken yet)
                        - Environment-specific fields describe initial state

        EXAMPLE IMPLEMENTATION (Echo Environment):
            def reset(self) -> EchoObservation:
                # 1. Generate new episode ID
                self._state = State(episode_id=str(uuid4()), step_count=0)

                # 2. Create initial observation
                return EchoObservation(
                    echoed_message="Echo environment ready!",
                    message_length=0,
                    done=False,
                    reward=0.0,
                )
        """
        pass

    # ==========================================================================
    # METHOD 2: step(action) - EXECUTE ONE TIMESTEP
    # ==========================================================================

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """
        Take a step in the environment by executing an action.

        ┌──────────────────────────────────────────────────────────────┐
        │                  STEP TRANSITION LOGIC                        │
        ├──────────────────────────────────────────────────────────────┤
        │                                                               │
        │  INPUT:  action (Action subclass)                             │
        │                                                               │
        │  PROCESS:                                                     │
        │    1. Parse action data                                       │
        │    2. Validate action (optional: check if legal)              │
        │    3. Update environment state based on action                │
        │    4. Compute reward for this transition                      │
        │    5. Check termination conditions (done?)                    │
        │    6. Increment step_count                                    │
        │    7. Construct observation                                   │
        │                                                               │
        │  OUTPUT: observation (Observation subclass)                   │
        │          Contains: new state, reward, done flag, metadata     │
        │                                                               │
        └──────────────────────────────────────────────────────────────┘

        WHAT: The core state transition function of the environment

        HOW:
            1. Receives an Action from the agent
            2. Updates internal state according to environment dynamics
            3. Computes reward signal
            4. Checks if episode is done
            5. Returns new Observation

        WHY:
            - This is where the environment logic lives
            - Implements the MDP (Markov Decision Process) transition function
            - Provides feedback (reward) to guide agent learning

        ACTION STRUCTURE:
            The action parameter MUST be a subclass of core.env_server.types.Action

            Base fields (inherited from Action):
                - metadata: dict = {}  (optional context)

            Custom fields (defined by your environment):
                e.g., for EchoEnvironment:
                    - message: str

        OBSERVATION STRUCTURE:
            The returned Observation MUST include:
                - done: bool        → Is episode finished?
                - reward: float     → Reward for taking this action
                - metadata: dict    → Optional extra info
                - [custom fields]   → Environment-specific state

        STATE ISOLATION:
            Each environment instance maintains its OWN state.
            - episode_id tracks which episode this is
            - step_count tracks progress within the episode
            - Environment-specific state (e.g., game board) is instance-local

        Args:
            action: Action to execute (subclass of core.env_server.types.Action)
                   Must match the action schema expected by this environment

        Returns:
            Observation: Result of taking the action
                - Contains new state information
                - reward: Scalar feedback signal
                - done: True if episode should terminate, False otherwise
                - metadata: Optional diagnostic info

        EXAMPLE IMPLEMENTATION (Echo Environment):
            def step(self, action: EchoAction) -> EchoObservation:
                # 1. Increment step count
                self._state.step_count += 1

                # 2. Process action
                message = action.message
                length = len(message)

                # 3. Compute reward
                reward = length * 0.1  # Simple: reward based on message length

                # 4. Create observation
                return EchoObservation(
                    echoed_message=message,
                    message_length=length,
                    done=False,  # Echo never terminates
                    reward=reward,
                    metadata={"step": self._state.step_count},
                )

        TERMINATION CONDITIONS (done=True):
            Set done=True when:
                - Goal is reached (e.g., solved a problem)
                - Failure condition met (e.g., invalid move)
                - Max steps reached (e.g., timeout)
                - Natural episode end (e.g., game over)
        """
        pass

    # ==========================================================================
    # PROPERTY: state - QUERY EPISODE METADATA
    # ==========================================================================

    @property
    @abstractmethod
    def state(self) -> State:
        """
        Get the current environment state metadata.

        WHAT: A read-only property that returns episode tracking information

        HOW: Returns a State object containing:
            - episode_id: str   → Unique identifier for this episode
            - step_count: int   → Number of steps taken in this episode

        WHY:
            - Enables episode tracking and monitoring
            - Useful for logging, debugging, and analysis
            - Does NOT expose full environment state (maintains abstraction)

        STATE vs OBSERVATION:
            - State: Metadata about the episode (episode_id, step_count)
            - Observation: Information given to the agent about the environment

        WHEN USED:
            - Agents can query this to check progress
            - Logging systems track episode_id for data organization
            - Debugging tools inspect step_count

        Returns:
            State: Object with episode_id and step_count
                  - episode_id: Unique UUID string for this episode
                  - step_count: Number of steps taken (0 after reset)

        EXAMPLE IMPLEMENTATION:
            @property
            def state(self) -> State:
                return self._state  # Return the stored State object

        IMPORTANT: This is a READ-ONLY property. It should NOT modify state.
        """
        pass

    # ==========================================================================
    # INTERNAL HELPER: _apply_transform() - OBSERVATION POST-PROCESSING
    # ==========================================================================

    def _apply_transform(self, observation: Observation) -> Observation:
        """
        Apply transform if one is provided.

        WHAT: Internal helper that applies the optional Transform to observations

        HOW:
            1. Check if self.transform is not None
            2. If transform exists, pass observation through it
            3. If no transform, return observation unchanged

        WHY:
            - Enables reward shaping without modifying core environment code
            - Provides a clean separation of concerns
            - Allows composable observation transformations

        WHEN CALLED:
            Typically called at the END of reset() and step() in subclasses:
                def step(self, action):
                    # ... compute raw observation ...
                    return self._apply_transform(raw_observation)

        Args:
            observation: Raw observation from the environment

        Returns:
            Transformed observation (or original if no transform)

        USAGE PATTERN IN SUBCLASS:
            def reset(self) -> Observation:
                obs = self._compute_initial_observation()
                return self._apply_transform(obs)  # Apply transform

            def step(self, action: Action) -> Observation:
                obs = self._compute_next_observation(action)
                return self._apply_transform(obs)  # Apply transform
        """
        if self.transform is not None:
            return self.transform(observation)
        return observation


# ==============================================================================
# KEY TAKEAWAYS
# ==============================================================================
"""
1. GYMNASIUM PATTERN:
   - reset() initializes episodes
   - step(action) executes state transitions
   - state property tracks episode metadata

2. TYPE SAFETY:
   - Action and Observation are base classes
   - Subclass them for environment-specific schemas
   - Type hints enable IDE autocomplete and error checking

3. TRANSFORMS:
   - Optional post-processing of observations
   - Enables reward shaping independently from environment logic
   - Applied via _apply_transform() helper

4. STATEFUL vs STATELESS:
   - Environment instance maintains state (episode_id, step_count)
   - Each reset() creates a NEW episode
   - state property exposes metadata (not full state)

5. EXTENSIBILITY:
   - Subclass Environment
   - Define custom Action and Observation types
   - Implement reset(), step(), and state
   - HTTPEnvServer automatically wraps it with HTTP API
"""

# ==============================================================================
# NEXT STEPS
# ==============================================================================
"""
To implement a custom environment:

1. Define your Action and Observation types:

   @dataclass(kw_only=True)
   class MyAction(Action):
       # Add your custom fields
       command: str

   @dataclass(kw_only=True)
   class MyObservation(Observation):
       # Add your custom fields
       result: str

2. Implement Environment:

   class MyEnvironment(Environment):
       def __init__(self):
           super().__init__()
           self._state = State(episode_id=str(uuid4()), step_count=0)

       def reset(self) -> MyObservation:
           self._state = State(episode_id=str(uuid4()), step_count=0)
           return MyObservation(result="Ready", done=False, reward=0.0)

       def step(self, action: MyAction) -> MyObservation:
           self._state.step_count += 1
           # Your logic here
           return MyObservation(result="...", done=False, reward=1.0)

       @property
       def state(self) -> State:
           return self._state

3. Wrap with HTTPEnvServer:

   from core.env_server.http_server import create_fastapi_app

   env = MyEnvironment()
   app = create_fastapi_app(env, MyAction, MyObservation)

4. Agents connect via HTTPEnvClient:

   client = MyEnvClient.from_docker_image("my-env:latest")
   result = client.reset()
   result = client.step(MyAction(command="do something"))
"""
