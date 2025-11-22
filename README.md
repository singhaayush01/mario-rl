🍄 Super Mario Bros. RL Project (DDQN Agent)

This repository documents the development of an Artificial Intelligence (AI) agent trained to complete World 1-1 of the classic Super Mario Bros. using Deep Reinforcement Learning (DRL).

Objective: Train a Deep Double Q-Network (DDQN) agent to navigate and complete Super Mario Bros. Level 1-1 using only raw pixel input.

🗓️ Weekly Progress Summary (Weeks 1 - 6)

Week 1: Baseline and Setup

The initial phase focused on establishing a stable working environment. The gym-super-mario-bros environment was successfully integrated, and a Random Agent was implemented to establish the performance floor.

Week 2: DRL Fundamentals & Reward Engineering

Key accomplishments included:

State Space Preprocessing: Implemented grayscale conversion, downsampling (84x84), and frame stacking (4 frames) to reduce state complexity and capture motion (see environment.py).

Custom Reward Shaping: Developed a custom reward function that aggressively penalizes idleness and rewards forward progress and completion. This was crucial for moving the agent beyond trivial random movement.

Week 3: Deep Q-Network (DQN) Implementation

The core learning algorithm was deployed: a Deep Q-Network (DQN), later adapted to the DDQN structure for stability. Initial training confirmed the agent learned the fundamental skill of Running Right and avoiding immediate death pits, with average reward improving dramatically from $\approx -750$ to $\approx -50$ after 1,000 episodes.

Week 4: DDQN Refinements & Optimized Training

The focus shifted to maximizing learning stability and efficiency.

DDQN Transition: Fully implemented the Deep Dueling Double Q-Network (DDQN) structure to improve the accuracy of Q-value estimation. This is critical for states where Mario has multiple advantageous actions (e.g., when close to a power-up).

Reward Finalization: Introduced the dedicated FastRewardShaper class to refine the reward logic, ensuring new distance records are highly rewarded and eliminating micro-penalties for minor hesitations.

Architecture Optimization: Confirmed all training runs utilize the Mac MPS GPU acceleration (torch.backends.mps) for significantly faster iteration times.

Week 5: Comprehensive Evaluation Pipeline

A robust testing system was introduced to formally measure the agent's performance, moving beyond anecdotal observation.

Formal Testing (test_agent.py): Developed a script to run the trained model over multiple episodes (typically 20-50) to calculate key metrics:

Completion Rate: Percentage of episodes ending in victory (reaching the flagpole).

Average Distance: Mean maximum X-position reached.

Best Distance: The furthest X-position achieved (The current record-holder).

Model Checkpointing: Implemented model saving logic (mario_best.pth) to automatically store the network weights whenever a new Max X distance is achieved.

Week 6: Policy Visualization and Long-Term Run

The project entered its final, performance-focused phase.

Live Demonstration (play_mario.py): Developed a script to load the latest mario_best.pth model and visualize the agent's learned policy in real-time. This provides critical insight into why the agent succeeds or fails at specific obstacles.

Commitment to Completion: Commenced the final, long-duration training run (set for 5,000+ episodes) required to achieve the necessary skill set for full level completion (X=3152).

🛠️ Key Technical Details

1. DDQN Architecture

The model uses a Dueling DQN structure to separate the Value stream (how good a state is) and the Advantage stream (the benefit of taking a specific action). This separation significantly speeds up learning complex state representations.

2. Environment Compatibility

The project utilizes torch.backends.mps for native GPU acceleration on Mac (Apple Silicon), ensuring high-speed training and efficient resource usage.

3. Current Status

The agent is currently in long-term training, aggressively exploring advanced platforming and enemy interactions, with the goal of generating the mario_completed.pth checkpoint file.
