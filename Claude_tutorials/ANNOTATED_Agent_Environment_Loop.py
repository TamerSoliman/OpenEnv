#!/usr/bin/env python3
"""
================================================================================
HEAVILY ANNOTATED: Agent-Environment Interaction Loop
================================================================================

This file demonstrates the COMPLETE agent-environment interaction pattern
used in OpenEnv, based on the Gymnasium API.

WHAT: Shows how agents interact with environments through reset() and step()
HOW: Line-by-line walkthrough of a complete episode execution
WHY: Understanding this loop is essential for building RL agents and
     environment consumers

Source: Based on /home/user/OpenEnv/examples/local_echo_env.py
================================================================================
"""

# ==============================================================================
# IMPORTS AND SETUP
# ==============================================================================

import sys
from pathlib import Path

# Add src to path (typical pattern for running examples)
# WHY: Allows importing from the OpenEnv source code directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the environment client and action types
# WHAT: These are the interfaces for talking to the environment
from envs.echo_env import EchoAction, EchoEnv


# ==============================================================================
# MAIN INTERACTION FUNCTION - THE COMPLETE AGENT-ENVIRONMENT LOOP
# ==============================================================================

def main():
    """
    Demonstrates the full agent-environment interaction loop.

    ┌──────────────────────────────────────────────────────────────┐
    │              AGENT-ENVIRONMENT INTERACTION LOOP               │
    ├──────────────────────────────────────────────────────────────┤
    │                                                               │
    │  1. CREATE ENVIRONMENT                                        │
    │     └─→ from_docker_image() or from_hub()                    │
    │                                                               │
    │  2. RESET ENVIRONMENT                                         │
    │     └─→ client.reset() → StepResult[Observation]             │
    │                                                               │
    │  3. LOOP UNTIL DONE:                                          │
    │     ├─→ Agent observes current state                         │
    │     ├─→ Agent decides on action                              │
    │     ├─→ client.step(action) → StepResult[Observation]        │
    │     ├─→ Agent receives: observation, reward, done            │
    │     └─→ Check if done, else continue loop                    │
    │                                                               │
    │  4. CLEANUP                                                   │
    │     └─→ client.close()                                       │
    │                                                               │
    └──────────────────────────────────────────────────────────────┘
    """

    print("=" * 60)
    print("AGENT-ENVIRONMENT INTERACTION LOOP DEMONSTRATION")
    print("=" * 60)
    print()

    # ==========================================================================
    # STEP 1: CREATE ENVIRONMENT CLIENT
    # ==========================================================================
    # WHAT: Initialize connection to the environment
    # HOW: Use from_docker_image() factory method to:
    #      1. Start a Docker container running the environment server
    #      2. Wait for the HTTP server to be ready
    #      3. Create an HTTPEnvClient connected to that server
    # WHY: Abstracts away the complexity of Docker + HTTP setup

    try:
        print("STEP 1: Creating environment client from Docker image")
        print("-" * 60)
        print("  Calling: EchoEnv.from_docker_image('echo-env:latest')")
        print()
        print("  This does:")
        print("    1. docker run -d -p <port>:8000 echo-env:latest")
        print("    2. Wait for HTTP health check at http://localhost:<port>/health")
        print("    3. Create EchoEnv client pointing to http://localhost:<port>")
        print()

        # CRITICAL LINE: Create the environment client
        # TYPE: EchoEnv (which is a subclass of HTTPEnvClient[EchoAction, EchoObservation])
        client = EchoEnv.from_docker_image("echo-env:latest")

        print("✓ Client created successfully!")
        print(f"  Client type: {type(client).__name__}")
        print(f"  Base URL: {client._base}")
        print()

        # ==========================================================================
        # STEP 2: RESET THE ENVIRONMENT - START A NEW EPISODE
        # ==========================================================================
        # WHAT: Initialize a new episode and get the initial observation
        # HOW:
        #   1. Client sends HTTP POST to /reset
        #   2. Server calls env.reset()
        #   3. Server returns JSON: {observation: {...}, reward: 0.0, done: false}
        #   4. Client parses JSON into StepResult[EchoObservation]
        # WHY: Every episode must start with reset() to initialize state

        print("STEP 2: Reset environment - initialize new episode")
        print("-" * 60)
        print("  Calling: client.reset()")
        print()
        print("  HTTP Request:")
        print("    POST http://localhost:<port>/reset")
        print("    Body: {}")
        print()

        # CRITICAL LINE: Reset the environment
        # RETURN TYPE: StepResult[EchoObservation]
        result = client.reset()

        print("  HTTP Response (parsed):")
        print(f"    observation: {result.observation}")
        print(f"    reward: {result.reward}")
        print(f"    done: {result.done}")
        print()

        # UNPACK THE RESULT
        # StepResult contains three key fields:
        #   - observation: EchoObservation (environment-specific data)
        #   - reward: float | None (scalar feedback signal)
        #   - done: bool (is episode finished?)

        initial_observation = result.observation  # Type: EchoObservation
        initial_reward = result.reward           # Type: float | None
        initial_done = result.done               # Type: bool

        print("  Unpacked StepResult:")
        print(f"    observation.echoed_message = '{initial_observation.echoed_message}'")
        print(f"    observation.message_length = {initial_observation.message_length}")
        print(f"    reward = {initial_reward}")
        print(f"    done = {initial_done}")
        print()

        # AGENT STATE INITIALIZATION
        # At this point, the agent knows:
        #   - The episode has started (done=False)
        #   - The initial observation (what the environment looks like)
        #   - No reward yet (typically 0.0 after reset)

        print("✓ Environment reset complete - episode initialized")
        print()

        # ==========================================================================
        # STEP 3: AGENT-ENVIRONMENT INTERACTION LOOP
        # ==========================================================================
        # WHAT: The core RL loop where agent and environment interact
        # HOW:
        #   1. Agent observes current state (observation from previous step/reset)
        #   2. Agent decides on action (policy: observation → action)
        #   3. Agent executes action via client.step(action)
        #   4. Environment returns new observation, reward, done
        #   5. Agent processes reward (learning update, logging, etc.)
        #   6. Check if done; if not, loop back to step 1
        # WHY: This is the fundamental pattern of reinforcement learning

        print("STEP 3: Agent-environment interaction loop")
        print("-" * 60)
        print()

        # Define a sequence of messages for the agent to send
        # AGENT POLICY: In this simple example, the policy is scripted
        #               (pre-defined messages). In a real RL agent, this would be
        #               a learned policy: π(a|s) = probability of action a given state s

        messages = [
            "Hello, World!",
            "Testing echo environment",
            "One more message",
        ]

        print(f"  Agent policy: Send {len(messages)} pre-defined messages")
        print()

        # -------------------------------------------------------------------------
        # LOOP ITERATION: For each action the agent wants to take
        # -------------------------------------------------------------------------

        for i, msg in enumerate(messages, 1):
            print(f"  ┌─ ITERATION {i}/{len(messages)} " + "─" * 40)
            print(f"  │")

            # =====================================================================
            # SUB-STEP 3.1: AGENT DECISION - CONSTRUCT ACTION
            # =====================================================================
            # WHAT: Agent decides what action to take based on observation
            # HOW: Creates an Action object with the chosen action parameters
            # WHY: Actions must be structured according to environment's schema

            print(f"  │ 3.1 Agent Decision: Construct action")
            print(f"  │     Message to send: '{msg}'")
            print(f"  │")

            # CRITICAL LINE: Create the action object
            # TYPE: EchoAction (subclass of Action)
            # SCHEMA:
            #   - message: str  (the message to echo)
            #   - metadata: dict (optional, inherited from Action base class)

            action = EchoAction(message=msg)

            print(f"  │     Action object: EchoAction(message='{action.message}')")
            print(f"  │")

            # =====================================================================
            # SUB-STEP 3.2: EXECUTE ACTION - ENVIRONMENT TRANSITION
            # =====================================================================
            # WHAT: Send action to environment and receive result
            # HOW:
            #   1. Client serializes action to JSON
            #   2. Client sends HTTP POST to /step with action payload
            #   3. Server deserializes action to EchoAction object
            #   4. Server calls env.step(action)
            #   5. Environment computes next state, reward, done
            #   6. Server serializes observation to JSON
            #   7. Client deserializes JSON to StepResult[EchoObservation]
            # WHY: This is the state transition: s_t, a_t → s_{t+1}, r_t, done

            print(f"  │ 3.2 Execute Action: Send to environment")
            print(f"  │     Calling: client.step(action)")
            print(f"  │")
            print(f"  │     HTTP Request:")
            print(f"  │       POST http://localhost:<port>/step")
            print(f"  │       Body: {{")
            print(f"  │         'action': {{'message': '{msg}'}},"
            print(f"  │         'timeout_s': 15")
            print(f"  │       }}")
            print(f"  │")

            # CRITICAL LINE: Execute the step
            # RETURN TYPE: StepResult[EchoObservation]
            #   Contains: (observation, reward, done)

            result = client.step(action)

            # =====================================================================
            # SUB-STEP 3.3: PROCESS RESULT - OBSERVATION, REWARD, DONE
            # =====================================================================
            # WHAT: Extract and process the environment's response
            # HOW: Unpack StepResult into its components
            # WHY: Agent needs this info for learning and decision-making

            print(f"  │     HTTP Response (parsed):")
            print(f"  │       observation: {result.observation}")
            print(f"  │       reward: {result.reward}")
            print(f"  │       done: {result.done}")
            print(f"  │")

            # UNPACK RESULT
            observation = result.observation  # Type: EchoObservation
            reward = result.reward           # Type: float | None
            done = result.done               # Type: bool

            # OBSERVATION FIELDS (specific to EchoObservation)
            echoed_message = observation.echoed_message  # What was echoed back
            message_length = observation.message_length  # Length of message
            obs_metadata = observation.metadata         # Optional extra info

            print(f"  │ 3.3 Process Result:")
            print(f"  │     Observation:")
            print(f"  │       echoed_message = '{echoed_message}'")
            print(f"  │       message_length = {message_length}")
            print(f"  │       metadata = {obs_metadata}")
            print(f"  │")
            print(f"  │     Reward: {reward}")
            print(f"  │       (In Echo env: reward = message_length * 0.1)")
            print(f"  │")
            print(f"  │     Done: {done}")
            print(f"  │       (In Echo env: never terminates, always False)")
            print(f"  │")

            # =====================================================================
            # SUB-STEP 3.4: AGENT UPDATE (if this were a learning agent)
            # =====================================================================
            # WHAT: Update agent's policy based on received reward
            # HOW: Depends on RL algorithm (Q-learning, policy gradient, etc.)
            # WHY: This is where learning happens in RL
            #
            # PSEUDOCODE FOR RL AGENT:
            #   agent.update(
            #       state=previous_observation,
            #       action=action,
            #       reward=reward,
            #       next_state=observation,
            #       done=done
            #   )
            #
            # IN THIS EXAMPLE: We just print the info (no learning)

            print(f"  │ 3.4 Agent Update: (scripted agent - no learning)")
            print(f"  │     A real RL agent would:")
            print(f"  │       - Store transition: (s_t, a_t, r_t, s_{{t+1}}, done)")
            print(f"  │       - Update policy/value function")
            print(f"  │       - Adjust exploration strategy")
            print(f"  │")

            # =====================================================================
            # SUB-STEP 3.5: CHECK TERMINATION CONDITION
            # =====================================================================
            # WHAT: Determine if episode should end
            # HOW: Check the 'done' flag from environment
            # WHY: Episodes have finite length (goal reached, max steps, etc.)

            print(f"  │ 3.5 Check Termination:")
            print(f"  │     done = {done}")
            if done:
                print(f"  │     → Episode finished! Breaking loop.")
                print(f"  │")
                print(f"  └" + "─" * 56)
                print()
                break  # Exit the loop - episode is over
            else:
                print(f"  │     → Episode continues, moving to next iteration")
                print(f"  │")
                print(f"  └" + "─" * 56)
                print()

            # Loop continues with next action (if not done)

        # ==========================================================================
        # STEP 4: QUERY ENVIRONMENT STATE (OPTIONAL)
        # ==========================================================================
        # WHAT: Get episode metadata from environment
        # HOW: Call client.state() which sends HTTP GET to /state
        # WHY: Useful for logging, debugging, monitoring

        print("STEP 4: Query environment state (optional)")
        print("-" * 60)
        print("  Calling: client.state()")
        print()
        print("  HTTP Request:")
        print("    GET http://localhost:<port>/state")
        print()

        # CRITICAL LINE: Get environment state
        # RETURN TYPE: State (with episode_id and step_count)
        state = client.state()

        print("  HTTP Response (parsed):")
        print(f"    episode_id: {state.episode_id}")
        print(f"    step_count: {state.step_count}")
        print()
        print(f"✓ Episode {state.episode_id} completed {state.step_count} steps")
        print()

        # ==========================================================================
        # STEP 5: CLEANUP - STOP ENVIRONMENT
        # ==========================================================================
        # WHAT: Shut down the environment and clean up resources
        # HOW: Call client.close() which stops the Docker container
        # WHY: Free up system resources, prevent container buildup

        print("STEP 5: Cleanup")
        print("-" * 60)
        print("  Calling: client.close()")
        print()
        print("  This does:")
        print("    1. Stop the Docker container")
        print("    2. Remove the container")
        print()

        client.close()

        print("✓ Environment closed, container removed")
        print()

        # ==========================================================================
        # SUMMARY
        # ==========================================================================

        print("=" * 60)
        print("INTERACTION LOOP COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print()
        print("Summary of what happened:")
        print()
        print("  1. Created environment client (Docker + HTTP)")
        print("  2. Reset environment → received initial observation")
        print(f"  3. Executed {len(messages)} actions:")
        print("     - Agent constructed action")
        print("     - Environment transitioned state")
        print("     - Agent received observation + reward + done")
        print("  4. Queried environment state metadata")
        print("  5. Cleaned up resources")
        print()
        print("This is the fundamental pattern for ALL OpenEnv interactions!")
        print()

        return True

    except Exception as e:
        print(f"\n❌ Interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# DATA FLOW DIAGRAM
# ==============================================================================
"""
┌──────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA FLOW                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  AGENT CODE                      HTTP                   ENVIRONMENT       │
│  (this file)                                            (Docker)          │
│                                                                           │
│  ┌─────────────┐                                      ┌──────────────┐   │
│  │ EchoEnv     │◄─────────────────────────────────────┤ FastAPI      │   │
│  │ Client      │  1. from_docker_image()              │ HTTP Server  │   │
│  └─────────────┘  ────────────────────────────────────►              │   │
│        │          Start container, wait for health     └──────┬───────┘   │
│        │                                                      │           │
│        │ 2. reset()                                           │           │
│        ├─────────POST /reset ──────────────────────────►      │           │
│        │          {}                                          │           │
│        │                                                      ▼           │
│        │                                              ┌──────────────┐    │
│        │                                              │ EchoEnv      │    │
│        │                                              │ Environment  │    │
│        │                                              └──────┬───────┘    │
│        │                                                     │            │
│        │                                         env.reset() │            │
│        │                                                     ▼            │
│        │                                         Create State:           │
│        │                                           episode_id = uuid4()  │
│        │                                           step_count = 0        │
│        │                                                     │            │
│        │                                         Return:     │            │
│        │                                           EchoObservation        │
│        │                                             echoed_message       │
│        │                                             message_length       │
│        │                                             done=False           │
│        │                                             reward=0.0           │
│        │                                                     │            │
│        │◄────────────────────────────────────────────────────┘            │
│        │          {observation: {...}, reward: 0.0, done: false}         │
│        │                                                                  │
│        │ StepResult[EchoObservation]                                      │
│        │   observation.echoed_message = "Echo environment ready!"        │
│        │   reward = 0.0                                                   │
│        │   done = False                                                   │
│        │                                                                  │
│        │ 3. step(EchoAction(message="Hello"))                             │
│        ├──────POST /step ──────────────────────────────────►              │
│        │       {action: {message: "Hello"}, timeout_s: 15}               │
│        │                                                      │           │
│        │                                         Deserialize: │           │
│        │                                           EchoAction │           │
│        │                                                      ▼           │
│        │                                         env.step(action)         │
│        │                                                      │           │
│        │                                         Update:      │           │
│        │                                           step_count += 1        │
│        │                                                      │           │
│        │                                         Compute:     │           │
│        │                                           reward = len * 0.1     │
│        │                                                      │           │
│        │                                         Return:      │           │
│        │                                           EchoObservation        │
│        │                                             echoed_message       │
│        │                                             message_length       │
│        │                                             done=False           │
│        │                                             reward=0.5           │
│        │                                                      │           │
│        │◄─────────────────────────────────────────────────────┘           │
│        │       {observation: {...}, reward: 0.5, done: false}            │
│        │                                                                  │
│        │ StepResult[EchoObservation]                                      │
│        │   observation.echoed_message = "Hello"                           │
│        │   observation.message_length = 5                                 │
│        │   reward = 0.5                                                   │
│        │   done = False                                                   │
│        │                                                                  │
│        │ 4. state()                                                       │
│        ├──────GET /state ──────────────────────────────────►              │
│        │                                                      │           │
│        │                                         env.state    │           │
│        │                                                      │           │
│        │◄─────────────────────────────────────────────────────┘           │
│        │       {episode_id: "...", step_count: 1}                         │
│        │                                                                  │
│        │ State                                                            │
│        │   episode_id = "abc-123-..."                                     │
│        │   step_count = 1                                                 │
│        │                                                                  │
│        │ 5. close()                                                       │
│        └────── Stop container ──────────────────────────────►             │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
"""


# ==============================================================================
# KEY CONCEPTS DEMONSTRATED
# ==============================================================================
"""
1. EPISODE LIFECYCLE:
   - Episodes start with reset()
   - Episodes progress with step(action)
   - Episodes end when done=True
   - Each episode has unique episode_id

2. ACTION-OBSERVATION LOOP:
   - Agent observes state → decides action → executes action
   - Environment transitions → computes reward → returns new state
   - This repeats until done=True

3. TYPE SAFETY:
   - Actions are typed (EchoAction)
   - Observations are typed (EchoObservation)
   - StepResult[T] wraps observations with reward and done

4. HTTP ABSTRACTION:
   - Client methods (reset, step, state) hide HTTP details
   - Server endpoints (/reset, /step, /state) wrap environment
   - JSON serialization/deserialization is automatic

5. CONTAINER LIFECYCLE:
   - from_docker_image() starts container
   - Container runs throughout interaction
   - close() stops and removes container

6. STATE TRACKING:
   - episode_id uniquely identifies episodes
   - step_count tracks progress within episode
   - State is separate from Observation
"""


# ==============================================================================
# COMPARISON: SCRIPTED AGENT vs RL AGENT
# ==============================================================================
"""
SCRIPTED AGENT (this example):
┌────────────────────────────────────────────────────────────────┐
│ for msg in ["Hello", "Test", "Goodbye"]:                       │
│     action = EchoAction(message=msg)  # Fixed policy           │
│     result = client.step(action)                               │
│     # No learning, just execution                              │
└────────────────────────────────────────────────────────────────┘

RL AGENT (with learning):
┌────────────────────────────────────────────────────────────────┐
│ observation = client.reset().observation                       │
│ done = False                                                   │
│                                                                │
│ while not done:                                                │
│     # Policy: Choose action based on observation              │
│     action = agent.select_action(observation)                 │
│                                                                │
│     # Execute action                                           │
│     result = client.step(action)                              │
│                                                                │
│     # Learn from transition                                    │
│     agent.update(                                              │
│         state=observation,                                     │
│         action=action,                                         │
│         reward=result.reward,                                  │
│         next_state=result.observation,                         │
│         done=result.done                                       │
│     )                                                          │
│                                                                │
│     # Move to next state                                       │
│     observation = result.observation                           │
│     done = result.done                                         │
└────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
