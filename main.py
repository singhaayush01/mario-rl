# main.py
import time
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import os

def safe_reset(env):
    """Handle both old and new reset API signatures."""
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
    else:
        obs = result
        info = {}
    return obs, info

def safe_step(env, action):
    """Handle both old and new step API signatures."""
    out = env.step(action)
    if len(out) == 4:
        obs, reward, done, info = out
    else:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    return obs, float(reward), done, info

def get_x_pos(env=None, info=None):
    """Return Mario's horizontal position."""
    # Try info first
    if info:
        for key in ['x_pos', 'x_position', 'scroll_x', 'xscroll', 'screen_x', 'x']:
            if key in info and info[key] is not None:
                return float(info[key])
    # Fallback to reading from RAM (works for older versions)
    if env is not None:
        try:
            page = int(env.ram[0x6d])
            x = int(env.ram[0x86])
            return float(page * 256 + x)
        except Exception:
            return None
    return None

def run_random_baseline(env_name='SuperMarioBros-1-1-v3', episodes=10, max_steps_per_episode=2000, render=True):
    # Create environment
    try:
        env = gym_super_mario_bros.make(env_name, render_mode="human")
    except Exception:
        # fallback to v0
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    results = []
    print("Starting random agent baseline. Episodes:", episodes)
    start_time = time.time()

    for ep in range(1, episodes + 1):
        obs, info = safe_reset(env)
        done = False
        total_reward = 0.0
        steps = 0
        start_x = get_x_pos(env, info)
        last_x = start_x if start_x is not None else 0.0

        while not done and steps < max_steps_per_episode:
            action = env.action_space.sample()
            obs, reward, done, info = safe_step(env, action)
            total_reward += reward
            steps += 1

            # Update last_x
            x = get_x_pos(env, info)
            if x is not None:
                last_x = x

            # Render
            if render:
                try:
                    env.render()
                except Exception:
                    pass

            # Debug print for first episode first step
            if ep == 1 and steps == 1:
                print("DEBUG info keys:", info.keys())
                print("DEBUG info sample:", info)

        distance = last_x - start_x if (start_x is not None and last_x is not None) else None
        results.append({'episode': ep, 'reward': total_reward, 'steps': steps, 'distance': distance})
        print(f"Episode {ep}: Reward={total_reward:.2f}, Steps={steps}, Distance={distance}")

        time.sleep(0.5)

    env.close()
    elapsed = time.time() - start_time

    # Summary
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_distances = [r['distance'] for r in results if r['distance'] is not None]
    avg_distance = np.mean(avg_distances) if avg_distances else None

    print("\n=== Random Baseline Summary ===")
    print(f"Episodes: {episodes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print(f"Average Distance: {avg_distance}")
    print(f"Elapsed time: {elapsed:.1f}s")

    # Save results
    out_file = "baseline_report.txt"
    with open(out_file, "w") as f:
        f.write("Episode,Reward,Steps,Distance\n")
        for r in results:
            f.write(f"{r['episode']},{r['reward']:.2f},{r['steps']},{r['distance']}\n")
        f.write("\n")
        f.write(f"Average Reward,{avg_reward:.2f}\n")
        f.write(f"Average Steps,{avg_steps:.1f}\n")
        f.write(f"Average Distance,{avg_distance}\n")
    print(f"\nSaved baseline results to {os.path.abspath(out_file)}")

    return results

if __name__ == "__main__":
    EPISODES = 5
    RENDER = True
    run_random_baseline(episodes=EPISODES, render=RENDER)
