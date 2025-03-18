import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=256, output_dim=4):
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

def compute_advantages(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    batch_size = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    for t in reversed(range(batch_size)):
        if t == batch_size - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value = 0.0
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t+1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        advantages[t] = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        last_advantage = advantages[t]
    return advantages

def train_agent():
    env = gym.make('LunarLander-v3')
    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    ppo_epochs = 4
    batch_size = 4096
    mini_batch_size = 64
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    max_timesteps = 1e6

    total_timesteps = 0
    episode = 0

    while total_timesteps < max_timesteps:
        obs_batch, acts_batch, log_probs_batch, rews_batch, vals_batch, dones_batch = [], [], [], [], [], []
        timesteps = 0

        while timesteps < batch_size:
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_rew = 0

            while not (done or truncated):
                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs).float()
                    logits, val = policy(obs_tensor)
                    dist = Categorical(logits=logits)
                    act = dist.sample()
                    log_prob = dist.log_prob(act)

                next_obs, rew, done, truncated, _ = env.step(act.item())

                obs_batch.append(obs)
                acts_batch.append(act.item())
                log_probs_batch.append(log_prob.item())
                rews_batch.append(rew)
                vals_batch.append(val.item())
                dones_batch.append(done)

                obs = next_obs
                episode_rew += rew
                timesteps += 1
                total_timesteps += 1

                if done or truncated:
                    print(f"Episode: {episode}, Reward: {episode_rew}")
                    episode += 1

        obs_tensor = torch.from_numpy(np.array(obs_batch)).float()
        acts_tensor = torch.tensor(acts_batch, dtype=torch.long)
        log_probs_tensor = torch.tensor(log_probs_batch, dtype=torch.float32)
        rews_tensor = torch.tensor(rews_batch, dtype=torch.float32)
        vals_tensor = torch.tensor(vals_batch, dtype=torch.float32)
        dones_tensor = torch.tensor(dones_batch, dtype=torch.float32)

        advantages = compute_advantages(rews_tensor, vals_tensor, dones_tensor, gamma, gae_lambda)
        returns = advantages + vals_tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(ppo_epochs):
            indices = np.arange(len(obs_batch))
            np.random.shuffle(indices)
            for start in range(0, len(indices), mini_batch_size):
                end = start + mini_batch_size
                idx = indices[start:end]

                obs_mb = obs_tensor[idx]
                acts_mb = acts_tensor[idx]
                old_log_probs_mb = log_probs_tensor[idx]
                adv_mb = advantages[idx]
                returns_mb = returns[idx]

                logits, vals = policy(obs_mb)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(acts_mb)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_log_probs_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = 0.5 * (returns_mb - vals.squeeze()).pow(2).mean()
                loss = actor_loss + vf_coef * critic_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

        state_dict = policy.state_dict()
        state_dict_np = {key: value.numpy() for key, value in state_dict.items()}
        np.save('best_policy_2202.npy', state_dict_np)

if __name__ == '__main__':
    train_agent()