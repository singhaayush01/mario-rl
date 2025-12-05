import torch
from environment import MarioEnvironment
from model import DDQN
from policy import select_action
import os

# --- Configuration ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def play_demo(model_path):
    print(f"🔍 Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found.")
        print("💡 Make sure you have 'mario_d3qn_best.pth' in this folder.")
        return

    # --- 1. Setup Environment for Visualization ---
    # render_mode="human" is REQUIRED to see the window
    env = MarioEnvironment(level="1-1", render_mode="human")
    n_actions = env.action_space.n 
    state_shape = env.observation_space.shape
    
    # --- 2. Load Policy Network ---
    policy_net = DDQN(state_shape, n_actions).to(DEVICE)
    
    try:
        policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ Model loaded successfully from {model_path}!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    policy_net.eval() # Switch to evaluation mode (no learning)
    
    # --- 3. Run Demo Loop ---
    try:
        print("🎮 Starting Demo... Press Ctrl+C to stop.")
        # Run for 10 episodes or until you stop it
        for episode in range(1, 11): 
            state, info = env.reset()
            total_reward = 0
            print(f"\n🎬 Demo Episode {episode} start...")
            
            while True:
                # 5% Randomness (Human-like twitch)
                action = select_action(policy_net, state, epsilon=0.1, n_actions=n_actions, device=DEVICE)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Render the game (shows the window)
                env.render()
                
                state = next_state
                
                if done or info.get('flag_get', False):
                    final_x = info.get('x_pos', 0)
                    print(f"   🏁 Finished. Distance: {final_x}")
                    break
    except KeyboardInterrupt:
        print("👋 Demo stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    # This points to the file saved by your training script
    # Your logs show you have 'mario_d3qn_best.pth' saved!
    play_demo(model_path="mario_d3qn_best.pth")