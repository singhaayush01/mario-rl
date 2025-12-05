import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os

from environment import MarioEnvironment
from model import DDQN
from memory import ReplayMemory
from policy import select_action

# --- Configuration for "The Final Push" ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

LEARNING_RATE = 0.00025
GAMMA = 0.99           
BATCH_SIZE = 64        
MEMORY_SIZE = 40000    # Slightly larger memory to remember old + new tricks
# Start exploration lower because we are loading a smart agent
EPSILON_START = 0.45   # 45% random (allows fixing mistakes without acting drunk)
EPSILON_END = 0.01     # Go very low for precision
EPSILON_DECAY = 0.99   # Slow decay to ensure he practices the hard jumps enough
TARGET_UPDATE = 1000   

class RewardHandler:
    """Pure RL Reward Shaping (No Cheating)"""
    def __init__(self):
        self.reset()
        
    def reset(self, info=None):
        self.prev_x = 0
        self.max_x = 0
        if info:
            self.prev_x = info.get('x_pos', 0)
            self.max_x = self.prev_x
            
    def get_reward(self, info, done):
        current_x = info.get('x_pos', 0)
        custom_reward = 0.0
        
        # 1. Moving Right (Velocity)
        distance = current_x - self.prev_x
        custom_reward += distance
        
        # 2. Exploration Bonus (Pure RL: Intrinsic Motivation)
        # Big reward for discovering NEW territory
        if current_x > self.max_x:
            bonus = (current_x - self.max_x) * 2.0 
            custom_reward += bonus
            self.max_x = current_x
        
        # 3. Penalties (Natural Consequences)
        if done and not info.get('flag_get', False):
            # Death penalty
            custom_reward -= 50.0
        
        # Stagnation penalty (don't stand still)
        if distance <= 0:
            custom_reward -= 1.0

        # 4. Goal Completion
        if info.get('flag_get', False):
            custom_reward += 2000.0 # Huge incentive to touch the flag
            
        self.prev_x = current_x
        return custom_reward

def train_agent(episodes=5000, model_name="mario_d3qn"):
    env = MarioEnvironment(level="1-1", render_mode=None)
    n_actions = env.action_space.n
    state_shape = env.observation_space.shape
    
    policy_net = DDQN(state_shape, n_actions).to(DEVICE)
    target_net = DDQN(state_shape, n_actions).to(DEVICE)
    
    # --- LOAD CHECKPOINT IF EXISTS ---
    # This is the key to finishing in 3 hours
    load_path = f"{model_name}_best.pth"
    if os.path.exists(load_path):
        print(f"🔄 Resuming training from: {load_path}")
        try:
            policy_net.load_state_dict(torch.load(load_path, map_location=DEVICE))
            print("✅ Brain loaded! Tuning existing agent.")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}. Starting fresh.")
    else:
        print("🆕 No checkpoint found. Starting fresh.")
    # ---------------------------------

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)
    reward_handler = RewardHandler()
    
    epsilon = EPSILON_START
    total_steps = 0
    best_max_x = 0 
    
    print(f"🚀 Training started on {DEVICE}")
    
    try: 
        for episode in range(1, episodes + 1):
            state, info = env.reset()
            reward_handler.reset(info)
            total_reward = 0
            
            # Update best_max_x based on loaded model's potential
            # (We don't know the file's max, so we start tracking fresh)
            
            while True:
                action = select_action(policy_net, state, epsilon, n_actions, DEVICE)
                next_state, _, done, truncated, info = env.step(action)
                
                # Custom Reward
                reward = reward_handler.get_reward(info, done)
                total_reward += reward
                
                memory.push(state, action, reward, next_state, done)
                state = next_state
                total_steps += 1
                
                # Optimize
                if len(memory) > BATCH_SIZE:
                    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                    states = torch.FloatTensor(states).to(DEVICE)
                    actions = torch.LongTensor(actions).to(DEVICE)
                    rewards = torch.FloatTensor(rewards).to(DEVICE)
                    next_states = torch.FloatTensor(next_states).to(DEVICE)
                    dones = torch.FloatTensor(dones).to(DEVICE)

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
                
                if total_steps % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
                if done or truncated or info.get('flag_get', False):
                    break
            
            # Decay Epsilon
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
            
            # Save Best
            if reward_handler.max_x > best_max_x and reward_handler.max_x > 500:
                best_max_x = reward_handler.max_x
                torch.save(policy_net.state_dict(), f"{model_name}_best.pth")
                print(f"💾 Improvement! Max X: {best_max_x:.0f} saved.")

            if episode % 10 == 0:
                print(f"Ep {episode} | Max X: {reward_handler.max_x:.0f} | Reward: {total_reward:.1f} | Eps: {epsilon:.3f}")
                
            if info.get('flag_get', False):
                print(f"🎉 LEVEL COMPLETED at Episode {episode}!")
                torch.save(policy_net.state_dict(), f"{model_name}_completed.pth")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Pausing training.")
        torch.save(policy_net.state_dict(), f"{model_name}_interrupted.pth")
        
    env.close()

if __name__ == "__main__":
    # 5 Hours on Mac MPS is roughly 3000-5000 episodes
    train_agent(episodes=5000)