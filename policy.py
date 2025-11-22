import numpy as np
import torch

def select_action(policy_net, state, epsilon, n_actions, device):
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            return q_values.argmax(1).item()