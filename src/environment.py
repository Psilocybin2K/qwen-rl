"""RL Environment that provides states and ground truth separately from dataset."""

import json
import random
from typing import Dict, List
from pathlib import Path

# Import DatasetManager from parent directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwenrl import DatasetManager


class RLEnvironment:
    """RL Environment that wraps the dataset.
    
    The environment provides states (query + context) separately from ground truth.
    This ensures that the agent never sees the correct answer during generation,
    and the reward function can access ground truth independently.
    """
    
    def __init__(self, dataset_path: str):
        """Initialize RL environment with dataset.
        
        Args:
            dataset_path (str): Path to the dataset JSON file
        """
        self.dataset_manager = DatasetManager(dataset_path)
        self.dataset = self.dataset_manager.get_all_queries()
    
    def get_state(self, index: int) -> Dict[str, str]:
        """Get state (query, context) - NO answer included.
        
        Returns only the information the agent should see during generation.
        Ground truth is kept separate in the environment.
        
        Args:
            index (int): Index of the dataset entry
            
        Returns:
            Dict[str, str]: State containing 'query' and 'context' only
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        
        entry = self.dataset[index]
        return {
            "query": entry["query"],
            "context": entry["context"]
            # Explicitly NO answer here - it's in environment, not state
        }
    
    def get_ground_truth(self, index: int) -> List[str]:
        """Get ground truth answer for reward calculation.
        
        This is kept separate from state to ensure the agent never sees
        the correct answer during generation. Only the reward function
        should access this.
        
        Args:
            index (int): Index of the dataset entry
            
        Returns:
            List[str]: Ground truth answer as list of steps
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        
        answer_str = self.dataset[index]["answer"]
        try:
            return json.loads(answer_str)
        except (json.JSONDecodeError, TypeError):
            # Fallback if answer is not valid JSON
            return [answer_str]
    
    def reset(self, index: int) -> Dict[str, str]:
        """Reset environment to a specific state.
        
        Args:
            index (int): Index of the dataset entry to reset to
            
        Returns:
            Dict[str, str]: State (query, context) without answer
        """
        return self.get_state(index)
    
    def sample(self) -> int:
        """Sample a random state index from the environment.
        
        Returns:
            int: Random index within dataset bounds
        """
        if not self.dataset:
            raise ValueError("Dataset is empty. Cannot sample.")
        return random.randint(0, len(self.dataset) - 1)
    
    def size(self) -> int:
        """Get the size of the environment (number of states).
        
        Returns:
            int: Number of entries in the dataset
        """
        return len(self.dataset)

