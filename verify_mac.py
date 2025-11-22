import torch
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import matplotlib.pyplot as plt

def verify():
    print("🔍 MAC SYSTEM DIAGNOSTIC...\n")

    # 1. Check Device
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) Detected")
    elif torch.cuda.is_available():
        print("✅ CUDA Detected (Unlikely on Mac)")
    else:
        print("⚠️  Running on CPU (Standard for Intel Mac)")

    # 2. Check Environment
    try:
        # Mac requires 'render_mode' to be explicit for the window to pop up
        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', apply_api_compatibility=True, render_mode="human")
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        print("✅ Mario Environment Loaded")
        
        env.reset()
        print("✅ Reset Successful - Attempting to render...")
        
        for i in range(20):
            action = env.action_space.sample()
            step_result = env.step(action)
            env.render() # This should open a window
            
        print("✅ Rendering Successful")
        env.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Hint: If the window didn't appear, check your Dock. Python might be bouncing.")

if __name__ == "__main__":
    verify()