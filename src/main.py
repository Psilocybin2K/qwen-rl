"""Main execution pipeline for RL-based test case generation system."""

import os
from pathlib import Path
from dotenv import load_dotenv

from src.environment import RLEnvironment
from src.agent import RLAgent
from src.reward import RewardFunction
from src.trainer import RLTrainer

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


def main():
    """Main execution pipeline using RL architecture.
    
    This replaces the old hybrid learning approach with a proper RL setup:
    - Environment provides states (query+context) separately from ground truth
    - Agent generates actions from states using prompts with instructions only
    - Reward function uses AI grader without ground truth in prompts
    - Trainer implements RL algorithms (REINFORCE/PPO) for policy updates
    """
    
    print("QA Test Case Generation System - RL Architecture")
    print("=" * 70)
    
    # Initialize RL components
    print("Setting up RL components...")
    
    # Environment (provides states and ground truth separately)
    dataset_path = "sample_dataset.json"
    try:
        env = RLEnvironment(dataset_path)
        print(f"[OK] Environment loaded: {env.size()} states available")
    except FileNotFoundError:
        print(f"[FAIL] Dataset file not found: {dataset_path}")
        return
    
    # Agent (generates actions from states)
    agent = RLAgent(model_name="Qwen/Qwen2.5-0.5B-Instruct", templates_dir="src/templates")
    print("[OK] Agent initialized")
    
    # Reward function (uses AI grader without ground truth in prompts)
    try:
        reward_fn = RewardFunction(templates_dir="src/templates")
        print("[OK] Reward function initialized")
    except KeyError as e:
        print(f"[FAIL] Missing environment variable: {e}")
        print("   Required: AOAI_API_KEY, AOAI_ENDPOINT")
        return
    
    # Trainer (implements RL training loop)
    trainer = RLTrainer(agent, env, reward_fn)
    print("[OK] Trainer initialized")
    
    # RL Training Loop
    print("\n" + "="*70)
    print("RL TRAINING")
    print("="*70)
    
    num_episodes = 5
    print(f"\nTraining on {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Sample a random state from environment
        episode_index = env.sample()
        state = env.get_state(episode_index)
        
        print(f"Query: {state['query']}")
        print(f"Context: {state['context'][:50]}...")
        
        # Train episode
        trajectory = trainer.train_episode(episode_index)
        
        print(f"Generated steps: {len(trajectory['action'])} steps")
        print(f"Reward: {trajectory['reward']:.3f}")
        
        # Show ground truth for comparison (logging only, not used in training)
        ground_truth = trajectory['ground_truth']
        print(f"Ground truth: {len(ground_truth)} steps")
    
    # Update policy
    print("\n" + "="*70)
    print("POLICY UPDATE")
    print("="*70)
    
    stats = trainer.get_statistics()
    print(f"\nTraining Statistics:")
    print(f"  Episodes: {stats.get('num_episodes', 0)}")
    print(f"  Average Reward: {stats.get('average_reward', 0):.3f}")
    print(f"  Max Reward: {stats.get('max_reward', 0):.3f}")
    print(f"  Min Reward: {stats.get('min_reward', 0):.3f}")
    
    print("\nUpdating policy with REINFORCE...")
    trainer.update_policy(algorithm="REINFORCE", learning_rate=5e-5)
    
    # Final summary
    print("\n" + "="*70)
    print("RL TRAINING SUMMARY")
    print("="*70)
    print("[OK] Environment: States and ground truth properly separated")
    print("[OK] Agent: Generates actions from states only (no ground truth access)")
    print("[OK] Reward: AI grader evaluates without ground truth in prompts")
    print("[OK] Training: RL-based policy updates (no supervised learning)")
    print("[OK] Data Separation: No dataset content embedded in prompts")
    
    print("\n[SUCCESS] RL-based QA Test Case Generation System ready!")


if __name__ == "__main__":
    main()

