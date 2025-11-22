import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from environment import MarioEnvironment
from model import DDQN
from memory import ReplayMemory
from policy import select_action

# Configuration
# Check for CUDA (Windows/Linux) OR MPS (Mac M1/M2/M3) OR CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

LEARNING_RATE = 0.00025
GAMMA = 0.99           # Discount factor
BATCH_SIZE = 64        # Larger batch size for stability
MEMORY_SIZE = 30000    # Larger memory
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.995
TARGET_UPDATE = 1000   # Update target network every N steps

class RewardHandler:
    """
    Handles reward normalization and shaping.
    Focuses on velocity and survival (No hardcoded cheating).
    """
    def __init__(self):
        self.prev_x = 0
        self.max_x = 0
        self.stuck_counter = 0

    def reset(self, info):
        self.prev_x = info.get('x_pos', 0)
        self.max_x = self.prev_x
        self.stuck_counter = 0

    def get_reward(self, info, reward, done):
        current_x = info.get('x_pos', 0)
        custom_reward = 0.0
        
        # 1. Velocity Reward: Reward actual distance moved
        distance = current_x - self.prev_x
        custom_reward += distance
        
        # 2. Stagnation Penalty: Penalize standing still
        if distance <= 0:
            self.stuck_counter += 1
            custom_reward -= 1.0  # Penalty for not moving
        else:
            self.stuck_counter = 0
            # Small momentum bonus
            if distance > 1: 
                custom_reward += 1.0

        # 3. Death Penalty
        if done and not info.get('flag_get', False):
            custom_reward -= 50.0
            
        # 4. Completion Bonus
        if info.get('flag_get', False):
            custom_reward += 500.0
            
        # 5. Clip rewards to stabilize gradients
        final_reward = custom_reward / 10.0
        
        self.prev_x = current_x
        self.max_x = max(self.max_x, current_x)
        
        return final_reward

def train_agent(episodes=5000, model_name="mario_d3qn"):
    env = MarioEnvironment(level="1-1", render_mode=None)
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    
    # Initialize Double DQN Networks
    policy_net = DDQN(state_shape, n_actions).to(DEVICE)
    target_net = DDQN(state_shape, n_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    reward_handler = RewardHandler()
    
    epsilon = EPSILON_START
    total_steps = 0
    
    print(f"Training started on {DEVICE}")
    print(f"Configuration: LR={LEARNING_RATE}, Batch={BATCH_SIZE}, Memory={MEMORY_SIZE}")
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        reward_handler.reset(info)
        
        total_reward = 0
        
        while True:
            # Select action
            action = select_action(policy_net, state, epsilon, n_actions, DEVICE)
            
            # Execute action
            next_state, _, done, truncated, info = env.step(action)
            
            # Calculate Custom Reward
            reward = reward_handler.get_reward(info, _, done)
            total_reward += reward
            
            # Store in memory
            memory.push(state, action, reward, next_state, done)
            state = next_state
            
            total_steps += 1
            
            # Optimization Step
            if len(memory) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                states = torch.FloatTensor(states).to(DEVICE)
                actions = torch.LongTensor(actions).to(DEVICE)
                rewards = torch.FloatTensor(rewards).to(DEVICE)
                next_states = torch.FloatTensor(next_states).to(DEVICE)
                
                # --- FIX: Use FloatTensor for dones instead of BoolTensor ---
                dones = torch.FloatTensor(dones).to(DEVICE)
                
                # Double DQN Logic
                with torch.no_grad():
                    next_actions = policy_net(next_states).max(1)[1]
                    next_q_values = target_net(next_states)
                    next_q_value = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    expected_q_values = rewards + GAMMA * next_q_value * (1 - dones)
                
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                loss = F.smooth_l1_loss(current_q_values, expected_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            
            # Soft Update Target Network
            if total_steps % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if done or truncated or info.get('flag_get', False):
                break
        
        # Decay Epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        if episode % 10 == 0:
            print(f"Episode {episode} | Max X: {reward_handler.max_x:.0f} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
            
        if info.get('flag_get', False):
            print(f"Level Completed at Episode {episode}!")
            torch.save(policy_net.state_dict(), f"{model_name}_completed.pth")

    env.close()
    torch.save(policy_net.state_dict(), f"{model_name}_final.pth")
    print("Training complete.")

if __name__ == "__main__":
    # Reduced episodes to 200 for a quicker initial test run, increase to 2000-5000 later
    train_agent(episodes=5000)