import torch
import torch.nn as nn
import torch.nn.functional as F

class DDQN(nn.Module):
    """
    Dueling Deep Q-Network (DDQN) architecture.
    Separates Value and Advantage streams for better training stability.
    """
    def __init__(self, input_shape, num_actions):
        super(DDQN, self).__init__()
        
        # Convolutional Feature Extractor
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the flattened size after convolutions
        # For 84x84 input, this results in 64 * 7 * 7 = 3136
        self.fc_input_dim = 64 * 7 * 7
        
        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        
        # Dueling Streams
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

    def forward(self, x):
        # Normalize pixel values
        x = x.float() / 255.0
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + advantage - advantage.mean(1, keepdim=True)
        
        return q_values