import numpy as np
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=512, output_dim=4):
        super(PolicyNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU()
        )
        # Separate actor and critic networks for better specialization
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, output_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )
        
        # Adjusted initialization for deeper network
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, x):
        shared_out = self.shared(x)
        action_logits = self.actor(shared_out)
        value = self.critic(shared_out)
        return action_logits, value

# class Policy:
#     def __init__(self):
#         self.model = PolicyNetwork()
#         params = np.load('best_policy_2202.npy', allow_pickle=True).item()
#         state_dict = {key: torch.tensor(value) for key, value in params.items()}
#         self.model.load_state_dict(state_dict)
#         self.model.eval()

#     def act(self, obs):
#         obs = np.array(obs)
#         obs_tensor = torch.tensor(obs, dtype=torch.float32)
#         with torch.no_grad():
#             action_logits, _ = self.model(obs_tensor)
#         action = torch.argmax(action_logits).item()
#         return action, None

def policy_action(policy, observation):
    """
    Function that takes the policy parameters and observation as input and returns an action.
    Args:
        policy: Dictionary containing the neural network parameters
        observation: Current environment observation
    Returns:
        action: Integer representing the action to take
    """
    # Create model instance and load state dict
    model = PolicyNetwork()
    state_dict = {key: torch.tensor(value) for key, value in policy.item().items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Convert observation to tensor and get action
    obs_tensor = torch.tensor(observation, dtype=torch.float32)
    with torch.no_grad():
        action_logits, _ = model(obs_tensor)
    action = torch.argmax(action_logits).item()
    return action