"""RL Trainer implementing policy-based reinforcement learning."""

import torch
import torch.nn.functional as F
from typing import Dict, List, Any
from transformers import Trainer, TrainingArguments
from datasets import Dataset

from src.environment import RLEnvironment
from src.agent import RLAgent
from src.reward import RewardFunction


class RLTrainer:
    """RL Trainer that implements policy-based RL algorithms (REINFORCE/PPO).
    
    Trains the agent using rewards from the reward function, without supervised
    learning or embedding correct answers in training sequences.
    """
    
    def __init__(self, agent: RLAgent, environment: RLEnvironment, reward_fn: RewardFunction):
        """Initialize RL trainer.
        
        Args:
            agent (RLAgent): The RL agent to train
            environment (RLEnvironment): The RL environment
            reward_fn (RewardFunction): The reward function
        """
        self.agent = agent
        self.env = environment
        self.reward_fn = reward_fn
        self.trajectories: List[Dict] = []
    
    def train_episode(self, episode_index: int) -> Dict:
        """Run a single RL training episode.
        
        Args:
            episode_index (int): Index of the episode (dataset entry)
            
        Returns:
            Dict: Trajectory containing state, action, reward, and ground truth (for logging)
        """
        # 1. Get state (NO answer)
        state = self.env.reset(episode_index)
        
        # 2. Agent generates action
        action = self.agent.act(state)
        
        # 3. Get ground truth from environment (for logging/validation only, not for reward)
        ground_truth = self.env.get_ground_truth(episode_index)
        
        # 4. Compute reward using AI grader (NO ground truth in prompt)
        generated_answer_str = self._format_action(action)
        reward = self.reward_fn.compute_reward(
            generated_answer=generated_answer_str,
            context=state["context"],
            query=state["query"]
            # Ground truth NOT passed to reward function
        )
        
        # 5. Store trajectory
        trajectory = {
            "state": state,
            "action": action,
            "reward": reward,
            "ground_truth": ground_truth,  # For logging/validation only
            "episode_index": episode_index
        }
        self.trajectories.append(trajectory)
        
        return trajectory
    
    def train_batch(self, indices: List[int]) -> List[Dict]:
        """Train on a batch of episodes.
        
        Args:
            indices (List[int]): List of episode indices to train on
            
        Returns:
            List[Dict]: List of trajectories from the batch
        """
        batch_trajectories = []
        for index in indices:
            trajectory = self.train_episode(index)
            batch_trajectories.append(trajectory)
        return batch_trajectories
    
    def update_policy(self, algorithm: str = "REINFORCE", learning_rate: float = 5e-5):
        """Update agent policy based on collected trajectories.
        
        Args:
            algorithm (str): RL algorithm to use ("REINFORCE" or "PPO")
            learning_rate (float): Learning rate for policy updates
        """
        if not self.trajectories:
            print("No trajectories collected. Run train_episode() first.")
            return
        
        if algorithm == "REINFORCE":
            self._reinforce_update(learning_rate)
        elif algorithm == "PPO":
            self._ppo_update(learning_rate)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Clear trajectories after update
        self.trajectories = []
    
    def _reinforce_update(self, learning_rate: float):
        """Update policy using REINFORCE algorithm.
        
        REINFORCE is a policy gradient method that updates the policy
        based on the gradient of expected return.
        
        Args:
            learning_rate (float): Learning rate for updates
        """
        # For now, implement a simplified version
        # Full REINFORCE would require:
        # 1. Computing log probabilities of actions
        # 2. Computing returns (discounted rewards)
        # 3. Gradient ascent on log_prob * return
        
        # This is a placeholder - full implementation would require
        # tracking log probabilities during generation and computing
        # policy gradients
        
        print(f"REINFORCE update with {len(self.trajectories)} trajectories")
        print(f"Average reward: {sum(t['reward'] for t in self.trajectories) / len(self.trajectories):.3f}")
        
        # TODO: Implement full REINFORCE with policy gradients
        # This would require modifying the agent to return log probabilities
        # and then computing gradients: grad = E[grad(log_prob) * return]
    
    def _ppo_update(self, learning_rate: float):
        """Update policy using PPO (Proximal Policy Optimization) algorithm.
        
        PPO is a more stable policy gradient method that clips the
        policy update to prevent large changes.
        
        Args:
            learning_rate (float): Learning rate for updates
        """
        # Placeholder for PPO implementation
        print(f"PPO update with {len(self.trajectories)} trajectories")
        print(f"Average reward: {sum(t['reward'] for t in self.trajectories) / len(self.trajectories):.3f}")
        
        # TODO: Implement full PPO with clipped objective
        # This would require:
        # 1. Computing old and new policy probabilities
        # 2. Computing advantage estimates
        # 3. Clipped objective: min(r * A, clip(r, 1-eps, 1+eps) * A)
    
    def _format_action(self, action: List[str]) -> str:
        """Format action list as JSON string for reward function.
        
        Args:
            action (List[str]): List of test steps
            
        Returns:
            str: JSON-formatted string
        """
        import json
        return json.dumps(action)
    
    def clear_trajectories(self):
        """Clear collected trajectories."""
        self.trajectories = []
    
    def get_statistics(self) -> Dict[str, float]:
        """Get training statistics from collected trajectories.
        
        Returns:
            Dict[str, float]: Statistics including average reward, etc.
        """
        if not self.trajectories:
            return {}
        
        rewards = [t["reward"] for t in self.trajectories]
        return {
            "num_episodes": len(self.trajectories),
            "average_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
        }

