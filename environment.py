import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

class GameEnv:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

    def reset(self):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
