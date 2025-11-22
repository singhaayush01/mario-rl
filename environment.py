import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
import numpy as np
from collections import deque
import gym

class MarioEnvironment:
    def __init__(self, level="1-1", render_mode=None, stack_frames=4):
        """
        Initializes the Mario environment with frame preprocessing.
        
        Args:
            level (str): Level to play (e.g., "1-1").
            render_mode (str or None): "human" for visualization, None for training.
            stack_frames (int): Number of frames to stack for temporal awareness.
        """
        self.env = gym_super_mario_bros.make(
            f"SuperMarioBros-{level}-v0", 
            apply_api_compatibility=True, 
            render_mode=render_mode
        )
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        
        self.stack_frames = stack_frames
        self.frames = deque(maxlen=stack_frames)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(stack_frames, 84, 84), dtype=np.uint8
        )
        self.action_space = self.env.action_space

    def preprocess_frame(self, frame):
        """Grayscale and resize frame to 84x84."""
        if frame is not None:
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame

    def reset(self):
        obs, info = self.env.reset()
        processed_frame = self.preprocess_frame(obs)
        
        self.frames.clear()
        for _ in range(self.stack_frames):
            self.frames.append(processed_frame)
            
        return np.array(self.frames), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_frame = self.preprocess_frame(obs)
        self.frames.append(processed_frame)
        
        return np.array(self.frames), reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()