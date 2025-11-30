"""Reward function using Azure OpenAI grader - evaluates without ground truth in prompt."""

import os
import json
import re
from typing import Dict, List
from smolagents import ToolCallingAgent, AzureOpenAIModel

from src.template_loader import TemplateLoader


class RewardFunction:
    """Reward function that uses Azure OpenAI grader to evaluate response quality.
    
    Key difference from old system: Grader receives generated_answer, context, and query,
    but NOT ground truth. The grader evaluates quality based on context requirements
    and query intent, not by comparing to a correct answer.
    
    This maintains similar functionality to SmoLAgentsGrader but removes data
    (ground truth) from the grading prompt.
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """Initialize reward function with Azure OpenAI grader.
        
        Sets up the Azure OpenAI model, creates a tool-calling agent, and initializes
        the template loader for prompt management.
        
        Environment Variables Required:
            AOAI_API_KEY: Azure OpenAI API key
            AOAI_ENDPOINT: Azure OpenAI endpoint URL
        """
        self.model = AzureOpenAIModel(
            api_key=os.environ["AOAI_API_KEY"],
            model_id="gpt-4.1",
            api_version="2024-06-01",
            azure_endpoint=os.environ["AOAI_ENDPOINT"],
        )
        
        self.agent = ToolCallingAgent(
            model=self.model,
            tools=[]
        )
        
        self.template_loader = TemplateLoader(templates_dir)
    
    def compute_reward(self, generated_answer: str, context: str, query: str) -> float:
        """Compute reward using AI grader - NO ground truth in prompt.
        
        The grader evaluates the generated answer based on:
        - Context requirements (what the business process needs)
        - Query intent (what the user is asking for)
        - Quality criteria (completeness, order, clarity)
        
        It does NOT receive the correct answer, so it must evaluate quality
        independently based on the context and query.
        
        Args:
            generated_answer (str): The generated test case steps to be graded
            context (str): Business context for the query
            query (str): The original query
            
        Returns:
            float: Reward score between 0.0 and 1.0, where 1.0 is perfect quality
        """
        # Build grading prompt with generated_answer, context, query
        # BUT NOT with correct_answer or ground_truth
        prompt = self.template_loader.format_template(
            "grading_prompt_no_ground_truth",
            generated_answer=generated_answer,
            context=context,
            query=query
            # Explicitly NO correct_answer parameter
        )
        
        try:
            result = self.agent.run(prompt)
            # Debug: uncomment to see what agent.run() returns
            # print(f"DEBUG: result type = {type(result)}, value = {result}")
            return self._extract_accuracy_score(result)
        except Exception as e:
            print(f"Grading error: {e}")
            return 0.0
    
    def _extract_accuracy_score(self, result) -> float:
        """Extract accuracy score from grader response.
        
        Handles both dict and string results from SmoLAgents ToolCallingAgent.
        
        Args:
            result: Response from grader agent (can be dict or str)
            
        Returns:
            float: Normalized accuracy score between 0.0 and 1.0
        """
        # Handle dict result (most common with SmoLAgents ToolCallingAgent)
        if isinstance(result, dict):
            # Direct dict with accuracy key
            if "accuracy" in result:
                return max(0.0, min(1.0, result["accuracy"]))
            # Nested in 'answer' key (from final_answer tool call)
            if "answer" in result and isinstance(result["answer"], dict):
                if "accuracy" in result["answer"]:
                    return max(0.0, min(1.0, result["answer"]["accuracy"]))
            # Check for nested structures
            for key in ["answer", "result", "data", "observations"]:
                if key in result and isinstance(result[key], dict):
                    if "accuracy" in result[key]:
                        return max(0.0, min(1.0, result[key]["accuracy"]))
        
        # Handle string result (fallback)
        if isinstance(result, str):
            # Try to extract JSON from the response
            try:
                result_object = json.loads(result)
                return max(0.0, min(1.0, result_object["accuracy"]))
            except json.JSONDecodeError:
                # If direct JSON parsing fails, try to extract from the response text
                json_match = re.search(r'final_answer\(\s*(\{.*?\})\s*\)', result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result_object = json.loads(json_str)
                    return max(0.0, min(1.0, result_object["accuracy"]))
                else:
                    print(f"Could not extract JSON from response: {result[:200]}...")
                    return 0.0
        
        # Debug: unexpected type
        print(f"Unexpected result type: {type(result)}, value: {result}")
        return 0.0

