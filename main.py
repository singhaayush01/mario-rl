import random
import numpy as np
from environment import GameEnv

def run_random_agent(episodes=5):
    env = GameEnv("SuperMarioBros-1-1-v0")

    scores = []
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = random.randint(0, env.env.action_space.n - 1)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            if done:
                print(f"Episode {ep+1}: score={info.get('score', 0)}, "
                      f"x_pos={info.get('x_pos', 0)}, steps={steps}")
                scores.append(total_reward)
                break

    env.close()
    print(f"\nAverage reward over {episodes} episodes: {np.mean(scores):.2f}")

if __name__ == "__main__":
    run_random_agent()
