import gym
from preprocessing import preprocess_frame, FrameStack

class GameEnv:
    def __init__(self, env_name):
        # Create the environment
        self.env = gym.make(env_name) # creating a game
        self.frame_stack = FrameStack(4)

    def reset(self):
        frame = self.env.reset() # starting a new game

        if isinstance(frame, tuple):
            frame, _ = frame

        frame = preprocess_frame(frame) # Process the first game
        state = self.frame_stack.reset(frame)
        return state # what the model sees
    
    def step(self, action): # A function named step inside environment wrapper classes
        # Take (one step/runs one frame) in the game
        out = self.env.step(action)

        if len(out) == 4:
            next_frame, reward, done, info = out
        else:
            next_frame, reward, terminated, truncated, info = out
            done = terminated or truncated

        next_frame = preprocess_frame(next_frame)
        next_state = self.frame_stack.step(next_frame)
        # Return everything the agent needs
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

