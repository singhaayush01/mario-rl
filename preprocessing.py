import cv2 #for preprocessing image
import numpy as np # for arrays and numerical operations
from collections import deque # for frame stacking(4)

#changing RGB color to gray to save memory
def preprocess_frame(frame):
    #convert from RGB scale to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
    #reduce the size of photo
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    #Normalized
    normalized = resized/255.0

    return normalized.astype(np.float32)

class FrameStack:
    def __init__(self, stack_size = 4):
        # number of frames to keep
        self.stack_size = stack_size
        self.frames = deque(maxlen = stack_size)

    def reset(self, frame):
        #clear old frames
        self.frames.clear()
        #fill the stack with the first frame 4 times
        for _ in range(self.stack_size):
            self.frames.append(frame)
        #convert to numpy array of shape (4,84,84)
        return np.stack(self.frames, axis = 0)
    
    def step(self, frame):
        #add new frame to stack, old one is dropped automatically
        self.frames.append(frame)
        return np.stack(self.frames, axis = 0)
        



    

