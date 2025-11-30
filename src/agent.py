"""RL Agent that generates actions from states using prompts with instructions only."""

import json
import re
import torch
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.template_loader import TemplateLoader


class RLAgent:
    """RL Agent that generates actions (test steps) from states (query + context).
    
    The agent uses prompts containing ONLY instructions - no examples, no correct answers,
    no dataset content. This ensures proper separation between data and prompts.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", templates_dir: str = "templates"):
        """Initialize RL agent.
        
        Args:
            model_name (str): The name or path of the Qwen model to use
            templates_dir (str): Directory containing template files
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.template_loader = TemplateLoader(templates_dir)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model for generation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32
        )
        
        # Resize embeddings if we added tokens
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def act(self, state: Dict[str, str], max_new_tokens: int = 128) -> List[str]:
        """Generate action (test steps) from state - NO ground truth access.
        
        Args:
            state (Dict[str, str]): State containing 'query' and 'context'
            max_new_tokens (int): Maximum number of new tokens to generate
            
        Returns:
            List[str]: Generated test steps as a list of strings
        """
        query = state.get("query", "")
        context = state.get("context", "")
        
        # Build prompt with ONLY instructions - NO examples, NO answers
        prompt = self._build_prompt(query, context)
        
        # Generate response
        response = self._generate(prompt, max_new_tokens)
        
        # Parse action from response
        return self._parse_action(response)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt with ONLY instructions - NO examples, NO answers.
        
        Args:
            query (str): User query
            context (str): Business context
            
        Returns:
            str: Formatted prompt ready for model input
        """
        # Load system prompt (instructions only)
        system_content = self.template_loader.load_template("system_prompt_generic")
        
        # Format user query template (query and context passed, but not embedded in template structure)
        user_content = self.template_loader.format_template(
            "user_query_template",
            query=query,
            context=context
        )
        
        # Use Qwen's chat template format
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
    
    def _generate(self, prompt: str, max_new_tokens: int = 128) -> str:
        """Generate response from prompt.
        
        Args:
            prompt (str): Formatted prompt
            max_new_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Generated text response
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=3,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response - keep special tokens for proper extraction
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=False
        )
        
        # Extract only the assistant's response using ChatML format
        if "<|im_start|>assistant" in generated_text:
            assistant_start = generated_text.find("<|im_start|>assistant")
            assistant_content = generated_text[assistant_start + len("<|im_start|>assistant"):]
            if "<|im_end|>" in assistant_content:
                response_text = assistant_content.split("<|im_end|>")[0].strip()
            else:
                response_text = assistant_content.strip()
            
            # Clean up markdown formatting and newlines
            response_text = response_text.replace("\\n", "\n")
            response_text = response_text.replace("```json", "").replace("```", "")
            response_text = response_text.strip()
        else:
            # Fallback: try to extract after the prompt
            response_text = generated_text.split(prompt)[-1].strip()
        
        return response_text
    
    def _parse_action(self, text: str) -> List[str]:
        """Parse generated text into a structured list of test steps.
        
        Args:
            text (str): The generated text to parse
            
        Returns:
            List[str]: A list of parsed test steps
        """
        try:
            # Try JSON parsing first
            if '[' in text and ']' in text:
                start = text.find('[')
                end = text.rfind(']') + 1
                json_text = text[start:end]
                return json.loads(json_text)
        except Exception:
            pass
        
        # Fallback parsing
        steps = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Query:', 'Generate', 'Test')):
                # Remove numbering, bullets, etc.
                clean_step = re.sub(r'^[\d\.\-\*\+]\s*', '', line)
                clean_step = re.sub(r'^Step\s*\d+[:.]?\s*', '', clean_step, flags=re.IGNORECASE)
                if clean_step and len(clean_step) > 3:
                    steps.append(clean_step)
        
        # If no structured steps found, try to extract from natural language
        if not steps and text.strip():
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in ['enter', 'click', 'select', 'choose', 'type', 'input', 'press']):
                    steps.append(sentence)
        
        return steps[:10]  # Limit to 10 steps max

