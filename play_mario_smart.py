import torch
import numpy as np
from environment import MarioEnvironment
from model import DDQN
import os
import random

# --- Configuration ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

class SmartPlayer:
    def __init__(self, model_path, env):
        self.env = env
        self.n_actions = env.action_space.n
        self.state_shape = env.observation_space.shape
        
        # Load the Brain
        self.policy_net = DDQN(self.state_shape, self.n_actions).to(DEVICE)
        
        if os.path.exists(model_path):
            try:
                self.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
                print(f"✅ Brain Loaded: {model_path}")
            except Exception as e:
                print(f"❌ Corrupt Model: {e}")
        else:
            print(f"❌ No model found at {model_path}")

        self.policy_net.eval()
        
        # Stuck Detection Variables
        self.last_x = 0
        self.same_x_frames = 0
        self.stuck_threshold = 60  # 1 second of not moving

    def get_ai_action(self, state, epsilon=0.0):
        # Standard Epsilon-Greedy Policy
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(1).item()

    def run_episode(self):
        state, info = self.env.reset()
        done = False
        current_x = 0
        
        while not done:
            # 1. Ask the AI what to do
            # We use a tiny bit of randomness (0.05) to stop it from looping exactly the same way
            action = self.get_ai_action(state, epsilon=0.05)

            # 2. DYNAMIC ACTION REPEATER (The Real Fix)
            # If the AI says "Jump" (Action 2, 4, or 5), we MUST hold it.
            # If the AI says "Run" (Action 1 or 3), we tap it briefly.
            
            # Default: Repeat action 4 times (standard RL frame skip)
            frames_to_repeat = 4 
            
            # COMPLEX MOVEMENT MAP (Gym Super Mario Bros SIMPLE_MOVEMENT)
            # 0: NOOP
            # 1: Right
            # 2: Right + A (Jump)
            # 3: Right + B (Run)
            # 4: Right + A + B (Run Jump)
            # 5: A (Jump)
            # 6: Left
            
            # If Jumping, hold for 12 frames (approx 0.2s) to get height
            if action in [2, 4, 5]: 
                frames_to_repeat = 12
            
            # 3. Execute the Action Loop
            for _ in range(frames_to_repeat):
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()
                
                state = next_state
                current_x = info.get('x_pos', 0)
                done = terminated or truncated

                # Break loop immediately if we die or finish
                if done: 
                    break

            # 4. Stuck Detection (The "Unsticker")
            # If we haven't moved past our last best X for 60 frames, force a random jump
            if current_x <= self.last_x + 2:
                self.same_x_frames += 1
            else:
                self.same_x_frames = 0
                self.last_x = current_x

            # If stuck for 1 second, override AI and FORCE a jump
            if self.same_x_frames > self.stuck_threshold:
                print(f"⚠️ Stuck at {current_x}. Forcing Unstick...")
                # Force "Right + Jump" (Action 2) for 15 frames
                for _ in range(15):
                    state, _, terminated, truncated, info = self.env.step(2) 
                    self.env.render()
                    if terminated or truncated: break
                
                self.same_x_frames = 0 # Reset counter

        return current_x

def main():
    # Setup Environment
    env = MarioEnvironment(level="1-1", render_mode="human")
    
    # Initialize Player
    player = SmartPlayer("mario_d3qn_best.pth", env)
    
    try:
        print("🎮 Starting Demo (Press Ctrl+C to quit)")
        print("---------------------------------------")
        while True:
            final_distance = player.run_episode()
            print(f"🏁 Episode Finished. Distance: {final_distance}")
            
            if final_distance > 3000:
                print("🏆 LEVEL COMPLETE (or close enough!)")
                break
                
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    finally:
        env.close()

if __name__ == "__main__":
    main()