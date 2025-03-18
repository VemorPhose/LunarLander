import numpy as np
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, output_dim=4):
        super(PolicyNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared_out = self.shared(x)
        action_logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_logits, value

class Policy:
    def __init__(self):
        self.model = PolicyNetwork()
        params = np.load('best_policy_2202.npy', allow_pickle=True).item()
        state_dict = {key: torch.tensor(value) for key, value in params.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, obs):
        obs = np.array(obs)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action_logits, _ = self.model(obs_tensor)
        action = torch.argmax(action_logits).item()
        return action, None