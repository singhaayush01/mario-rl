Action Space (SIMPLE_MOVEMENT):
0: noop
1: right
2: right + A
3: right + B
4: right + A + B
5: A
6: left

Reward structure: positive rewards for moving forward, collecting coins, killing enemies.
Distance calculated from horizontal x position (RAM-based fallback).

# 8-Week RL Mario Project

🗓️ Week 1: Environment setup and random agent baseline.

- Tested SuperMarioBros-1-1-v3 environment
- Random agent ran for 10 episodes
- Baseline metrics saved in baseline_report.txt
- Action space and reward structure documented

🗓️ Week 2: Learning Progress and Updates

During Week 2, I focused on building a deeper understanding of reinforcement learning fundamentals and improving the Mario RL environment. Key takeaways:

State Space & Preprocessing – Learned how each game frame represents the environment’s state and how preprocessing (grayscale, resize, normalization) reduces complexity for the model.

Reward Engineering – Designed custom reward functions that encourage progress, penalize idleness, and reward level completion.

Policy Learning Basics – Studied how agents map observed states to actions using learned policies instead of random behavior.

Baseline Random Agent – Implemented a random-action agent to establish initial performance benchmarks.

Understanding Environment Feedback – Explored how the environment returns observations, rewards, and episode termination signals.

Exploration vs. Exploitation – Learned why balancing these two is critical for efficient RL training.

Importance of Reward Design – Understood that poor reward shaping can mislead learning, while well-engineered rewards accelerate convergence toward optimal behavior.


🗓️ Week 3 – Basic DQN Implementation

This week I implemented the full Deep Q-Network to replace the random policy.

Network architecture

Three convolutional layers (32, 64, 64 filters) followed by two fully connected layers (512 → action space).

ReLU activations and Xavier weight initialization.

Training setup

ε-greedy exploration: start = 1.0, end = 0.01, decay = 0.995.

Replay memory size = 50 000, batch size = 32.

Discount factor γ = 0.99, optimizer = Adam (lr = 1e-4).

Target network updated every two episodes.

Trained on World 1-1 for 1,000 episodes.

Used TensorBoard to track loss, reward, and ε values.

Initial results (≈ first 1,000 episodes)

Average reward improved from about –750 to around –50.

Training loss decreased steadily over time.

Epsilon decayed smoothly to ≈ 0.4 by episode 1,000.

The trained model was saved as mario_dqn.pth.

Observations

Early learning was unstable due to sparse rewards.

Performance improved once ε decreased and the replay buffer filled.

Future goals include longer training (1,000 episodes) and video recording of evaluation runs.