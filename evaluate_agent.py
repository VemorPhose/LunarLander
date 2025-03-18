import gymnasium as gym
import numpy as np
import argparse
import importlib

def evaluate_policy(policy_instance, total_episodes=100, render_first=5):
    total_reward = 0.0
    for episode in range(total_episodes):
        render_mode = "human" if episode < render_first else "rgb_array"
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        observation, info = env.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action, _ = policy_instance.act(observation)  # Assuming Policy class has act method
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        env.close()
        total_reward += episode_reward
    return total_reward / total_episodes

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an AI agent for LunarLander-v3"
    )
    parser.add_argument(
        "--policy_module", type=str, required=True,
        help="The name of the Python module that defines the Policy class"
    )
    args = parser.parse_args()

    try:
        policy_module = importlib.import_module(args.policy_module)
        policy = policy_module.Policy()
    except ImportError as e:
        print(f"Error importing module {args.policy_module}: {e}")
        return
    except AttributeError:
        print(f"Module {args.policy_module} must define a Policy class")
        return

    average_reward = evaluate_policy(policy, total_episodes=100, render_first=5)
    print(f"Average reward over 100 episodes: {average_reward:.2f}")

if __name__ == "__main__":
    main()
