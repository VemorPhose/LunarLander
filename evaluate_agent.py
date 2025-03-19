import gymnasium as gym
import numpy as np
import argparse
import importlib
import time
from tqdm import tqdm

def evaluate_policy(policy, policy_action, total_episodes=100, render_first=5):
    total_reward = 0.0
    start_time = time.time()
    
    # Create progress bar
    pbar = tqdm(range(total_episodes), desc="Evaluating episodes", 
                unit="episode", ncols=80)
    
    for episode in pbar:
        # Render the first few episodes
        render_mode = "human" if episode < render_first else "rgb_array"
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        observation, info = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action = policy_action(policy, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
        env.close()
        total_reward += episode_reward
        
        # Update progress bar with current reward
        pbar.set_postfix({"Last Reward": f"{episode_reward:.1f}", 
                         "Avg Reward": f"{total_reward/(episode+1):.1f}"})
    
    # Calculate timing and final metrics
    total_time = time.time() - start_time
    avg_reward = total_reward / total_episodes
    
    # Print final statistics
    print("\nEvaluation Complete:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per episode: {total_time/total_episodes:.2f} seconds")
    
    return avg_reward

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an AI agent for LunarLander-v3 using a provided policy and policy_action function."
    )
    parser.add_argument(
        "--filename", type=str, required=True,
        help="Path to the .npy file containing the policy parameters."
    )
    parser.add_argument(
        "--policy_module", type=str, required=True,
        help="The name of the Python module that defines the policy_action function."
    )
    args = parser.parse_args()

    # Load the policy parameters from the file.
    policy = np.load(args.filename, allow_pickle=True)
    
    # Dynamically import the module that defines policy_action.
    try:
        policy_module = importlib.import_module(args.policy_module)
    except ImportError as e:
        print(f"Error importing module {args.policy_module}: {e}")
        return

    # Verify that the module has a callable policy_action function.
    if not hasattr(policy_module, "policy_action") or not callable(policy_module.policy_action):
        print(f"Module {args.policy_module} must define a callable 'policy_action(policy, observation)' function.")
        return
    policy_action_func = policy_module.policy_action

    # Evaluate the policy over 100 episodes (first 5 are rendered).
    average_reward = evaluate_policy(policy, policy_action_func, total_episodes=100, render_first=5)
    print(f"Average reward over 100 episodes: {average_reward:.2f}")

if __name__ == "__main__":
    main()
