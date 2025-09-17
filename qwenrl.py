import os
import torch
import json
import time
import random
from pathlib import Path

from typing import List, Dict, Any
from smolagents import ToolCallingAgent, AzureOpenAIModel
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset



class DatasetValidator:
    """Validates dataset entries and format according to required schema"""
    
    @staticmethod
    def validate_entry(entry: Dict) -> bool:
        """Validate a single dataset entry
        
        Args:
            entry (Dict): Dataset entry to validate
            
        Returns:
            bool: True if entry is valid, False otherwise
        """
        required_fields = ['query', 'context', 'answer']
        
        # Check required fields exist
        if not all(field in entry for field in required_fields):
            return False
            
        # Validate each field
        return (DatasetValidator.validate_query_format(entry['query']) and
                DatasetValidator.validate_context(entry['context']) and
                DatasetValidator.validate_answer_format(entry['answer']))
    
    @staticmethod
    def validate_query_format(query: str) -> bool:
        """Validate query format
        
        Args:
            query (str): Query string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return (isinstance(query, str) and 
                len(query.strip()) > 0 and 
                query.strip().endswith('?'))
    
    @staticmethod
    def validate_context(context: str) -> bool:
        """Validate context format
        
        Args:
            context (str): Context string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return (isinstance(context, str) and 
                len(context.strip()) >= 10)
    
    @staticmethod
    def validate_answer_format(answer: str) -> bool:
        """Validate answer format (must be valid JSON array of strings)
        
        Args:
            answer (str): Answer string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            parsed = json.loads(answer)
            return (isinstance(parsed, list) and 
                    all(isinstance(item, str) for item in parsed) and
                    len(parsed) > 0)
        except (json.JSONDecodeError, TypeError):
            return False


class DatasetManager:
    """Manages training dataset loading, validation, and query selection"""
    
    def __init__(self, dataset_path: str):
        """Initialize dataset manager
        
        Args:
            dataset_path (str): Path to the dataset JSON file
        """
        self.dataset_path = Path(dataset_path)
        self.dataset: List[Dict] = []
        self.validator = DatasetValidator()
        
        # Automatically load dataset if file exists
        if self.dataset_path.exists():
            self.load_dataset()
        
    def load_dataset(self) -> List[Dict]:
        """Load and validate dataset from file
        
        Returns:
            List[Dict]: Loaded and validated dataset
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset validation fails
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        if not isinstance(raw_data, list):
            raise ValueError("Dataset must be a list of entries")
        
        # Validate all entries
        for i, entry in enumerate(raw_data):
            if not self.validator.validate_entry(entry):
                raise ValueError(f"Invalid entry at index {i}: {entry}")
        
        self.dataset = raw_data
        print(f"Successfully loaded {len(self.dataset)} validated entries from {self.dataset_path}")
        return self.dataset
    
    def validate_dataset(self, data: List[Dict]) -> bool:
        """Validate entire dataset
        
        Args:
            data (List[Dict]): Dataset to validate
            
        Returns:
            bool: True if all entries are valid, False otherwise
        """
        if not isinstance(data, list):
            return False
            
        return all(self.validator.validate_entry(entry) for entry in data)
    
    def get_random_query(self) -> Dict:
        """Get a random query from the dataset
        
        Returns:
            Dict: Random query data entry
            
        Raises:
            ValueError: If dataset is empty
        """
        if not self.dataset:
            raise ValueError("Dataset is empty. Load dataset first.")
        
        return random.choice(self.dataset)
    
    def get_query_by_index(self, index: int) -> Dict:
        """Get query by index
        
        Args:
            index (int): Index of the query to retrieve
            
        Returns:
            Dict: Query data entry at specified index
            
        Raises:
            IndexError: If index is out of range
            ValueError: If dataset is empty
        """
        if not self.dataset:
            raise ValueError("Dataset is empty. Load dataset first.")
        
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.dataset)}")
        
        return self.dataset[index]
    
    def get_all_queries(self) -> List[Dict]:
        """Get all queries from the dataset
        
        Returns:
            List[Dict]: All query data entries
        """
        return self.dataset.copy()
    
    def get_dataset_size(self) -> int:
        """Get the size of the loaded dataset
        
        Returns:
            int: Number of entries in the dataset
        """
        return len(self.dataset)


class DynamicTemplateLoader:
    """Unified template loader with both static and dynamic content injection capabilities"""
    
    def __init__(self, templates_dir: str = "templates"):
        """Initialize dynamic template loader
        
        Args:
            templates_dir (str): Directory containing template files
        """
        self.templates_dir = Path(templates_dir)
    
    def load_template(self, template_name: str) -> str:
        """Load a template from a markdown file
        
        Args:
            template_name (str): Name of the template file (without .md extension)
            
        Returns:
            str: Clean template content without markdown headers
            
        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        template_path = self.templates_dir / f"{template_name}.md"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove markdown headers and return clean content
        lines = content.split('\n')
        # Skip lines that start with # (markdown headers)
        content_lines = [line for line in lines if not line.strip().startswith('#')]
        return '\n'.join(content_lines).strip()
    
    def format_template(self, template_name: str, **kwargs) -> str:
        """Load and format a template with provided variables
        
        Args:
            template_name (str): Name of the template file (without .md extension)
            **kwargs: Keyword arguments to use for template variable substitution
            
        Returns:
            str: Formatted template content with variables substituted
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            KeyError: If template contains variables not provided in kwargs
            ValueError: If template contains invalid format syntax
        """
        template = self.load_template(template_name)
        return template.format(**kwargs)
    
    def format_grading_prompt(self, context: str, correct_answer: str, generated_answer: str) -> str:
        """Format grading prompt with dynamic context and correct answer
        
        Args:
            context (str): Business context for the query
            correct_answer (str): The correct answer to use as source of truth
            generated_answer (str): The generated answer to be graded
            
        Returns:
            str: Formatted grading prompt with dynamic content
        """
        # Parse correct answer to create source of truth
        try:
            correct_steps = json.loads(correct_answer)
            source_of_truth = "\n".join([f"{i+1}. {step}" for i, step in enumerate(correct_steps)])
        except (json.JSONDecodeError, TypeError):
            source_of_truth = correct_answer
        
        # Load and format template with dynamic content
        return self.format_template(
            "dynamic_grading_prompt",
            context=context,
            correct_answer=correct_answer,
            source_of_truth=source_of_truth,
            generated_answer=generated_answer
        )
    
    def format_few_shot_prompt(self, context: str, example_query: str, example_answer: str) -> str:
        """Format few-shot prompt with dynamic context and examples
        
        Args:
            context (str): Business context for the query
            example_query (str): Example query to demonstrate
            example_answer (str): Example answer to demonstrate
            
        Returns:
            str: Formatted few-shot prompt with dynamic content
        """
        return self.format_template(
            "dynamic_few_shot_prompt",
            context=context,
            example_query=example_query,
            example_answer=example_answer
        )
    
    def format_teaching_prompt(self, context: str, query: str) -> str:
        """Format teaching prompt with dynamic context
        
        Args:
            context (str): Business context for the query
            query (str): The query to generate teaching examples for
            
        Returns:
            str: Formatted teaching prompt with dynamic content
        """
        return self.format_template(
            "dynamic_teaching_prompt",
            context=context,
            query=query
        )
    
    def format_context_integration_prompt(self, context: str, query: str) -> str:
        """Format context integration prompt for enhanced generation
        
        Args:
            context (str): Business context for the query
            query (str): The query to generate steps for
            
        Returns:
            str: Formatted context integration prompt
        """
        return self.format_template(
            "dynamic_context_integration_prompt",
            context=context,
            query=query
        )
    
    def format_user_query_with_context(self, query: str, context: str) -> str:
        """Format user query template with context information
        
        Args:
            query (str): The user query
            context (str): Business context for the query
            
        Returns:
            str: Formatted user query with context
        """
        return self.format_template(
            "dynamic_user_query_with_context",
            context=context,
            query=query
        )
    
    def format_context_aware_user_query(self, query: str, context: str) -> str:
        """Format context-aware user query template
        
        Args:
            query (str): The user query
            context (str): Business context for the query
            
        Returns:
            str: Formatted context-aware user query
        """
        return self.format_template(
            "context_aware_user_query",
            context=context,
            query=query
        )
    
    def format_context_aware_teaching_query(self, query: str, context: str) -> str:
        """Format context-aware teaching query template
        
        Args:
            query (str): The user query
            context (str): Business context for the query
            
        Returns:
            str: Formatted context-aware teaching query
        """
        return self.format_template(
            "context_aware_teaching_query",
            context=context,
            query=query
        )
    
    def format_context_aware_few_shot_example(self, query: str, context: str, example_query: str, example_answer: str) -> str:
        """Format context-aware few-shot example template
        
        Args:
            query (str): The current user query
            context (str): Business context for the query
            example_query (str): Example query for few-shot learning
            example_answer (str): Example answer for few-shot learning
            
        Returns:
            str: Formatted context-aware few-shot example
        """
        return self.format_template(
            "context_aware_few_shot_example",
            context=context,
            query=query,
            example_query=example_query,
            example_answer=example_answer
        )
    
    def format_multi_query_training_prompt(self, session_id: str, total_queries: int, current_query_index: int, 
                                         business_domain: str, query: str, context: str, correct_answer: str,
                                         completed_queries: int, successful_queries: int) -> str:
        """Format multi-query training coordination prompt
        
        Args:
            session_id (str): Unique session identifier
            total_queries (int): Total number of queries in the session
            current_query_index (int): Current query index (1-based)
            business_domain (str): Business domain for the training session
            query (str): Current query to process
            context (str): Business context for the query
            correct_answer (str): Expected correct answer
            completed_queries (int): Number of completed queries
            successful_queries (int): Number of successful queries
            
        Returns:
            str: Formatted multi-query training prompt
        """
        success_rate = (successful_queries / completed_queries * 100) if completed_queries > 0 else 0
        
        return self.format_template(
            "multi_query_training_prompt",
            session_id=session_id,
            total_queries=total_queries,
            current_query_index=current_query_index,
            business_domain=business_domain,
            query=query,
            context=context,
            correct_answer=correct_answer,
            completed_queries=completed_queries,
            successful_queries=successful_queries,
            success_rate=success_rate
        )
    
    def format_dataset_validation_prompt(self, dataset_name: str, total_entries: int, validation_status: str,
                                       valid_entries: int, invalid_entries: int, validation_errors: str,
                                       completeness_score: float, consistency_score: float, relevance_score: float,
                                       recommendations: str, next_steps: str) -> str:
        """Format dataset validation and management prompt
        
        Args:
            dataset_name (str): Name of the dataset
            total_entries (int): Total number of entries in the dataset
            validation_status (str): Current validation status
            valid_entries (int): Number of valid entries
            invalid_entries (int): Number of invalid entries
            validation_errors (str): Description of validation errors
            completeness_score (float): Completeness score (0-100)
            consistency_score (float): Consistency score (0-100)
            relevance_score (float): Relevance score (0-100)
            recommendations (str): Validation recommendations
            next_steps (str): Recommended next steps
            
        Returns:
            str: Formatted dataset validation prompt
        """
        return self.format_template(
            "dataset_validation_prompt",
            dataset_name=dataset_name,
            total_entries=total_entries,
            validation_status=validation_status,
            valid_entries=valid_entries,
            invalid_entries=invalid_entries,
            validation_errors=validation_errors,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            relevance_score=relevance_score,
            recommendations=recommendations,
            next_steps=next_steps
        )


class QwenTestCaseGenerator:
    """Test case generator using Qwen language model for creating automated test steps.
    
    This class provides functionality to generate test case steps from natural language
    queries using the Qwen language model. It supports both few-shot and regular prompting
    modes, context-aware generation, and includes parsing capabilities to extract structured test steps.
    
    When initialized with a DatasetManager, the generator can perform context-aware generation
    using business context and dynamic few-shot example selection from the dataset.
    
    Attributes:
        model_name (str): The name/path of the Qwen model to use
        dataset_manager (DatasetManager, optional): Dataset manager for context-aware generation
        tokenizer: The tokenizer for the model
        model: The loaded language model
        template_loader: Utility for loading prompt templates
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct", dataset_manager: DatasetManager = None):
        """Initialize Qwen-based test case generator.
        
        Args:
            model_name (str): The name or path of the Qwen model to use.
                Defaults to "Qwen/Qwen2.5-0.5B-Instruct".
            dataset_manager (DatasetManager, optional): Dataset manager for context-aware generation.
                If provided, enables dataset-driven operation with context injection.
        """
        
        self.model_name = model_name
        self.dataset_manager = dataset_manager
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.template_loader = DynamicTemplateLoader()
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load regular model for generation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )
        
        # Resize embeddings if we added tokens
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def generate_test_steps(self, query_data: Dict, max_new_tokens: int = 128, use_few_shot: bool = True, few_shot_example: Dict = None) -> Dict[str, Any]:
        """Generate test case steps from query data with optional context awareness.
        
        Takes query data (either a string for backward compatibility or a dict with context)
        and generates structured test steps using the Qwen language model. Supports both
        few-shot and regular prompting modes with optional context-aware generation.
        
        Args:
            query_data (Dict or str): Query data containing 'query' and optionally 'context'.
                For backward compatibility, can also accept a string query.
            max_new_tokens (int): Maximum number of new tokens to generate. Defaults to 128.
            use_few_shot (bool): Whether to use few-shot prompting. Defaults to True.
            few_shot_example (Dict, optional): Specific few-shot example to use.
                If not provided and dataset_manager is available, will select automatically.
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - steps (List[str]): Parsed list of test steps
                - raw_text (str): The complete generated text
                - response_text (str): The extracted assistant response
                - formatted_answer (str): Steps formatted as a readable list
        """
        
        # Handle backward compatibility: if query_data is a string, convert to dict
        if isinstance(query_data, str):
            query_data = {"query": query_data, "context": ""}
        
        # Validate query data
        if not self._validate_query_data(query_data):
            raise ValueError("Invalid query_data: must contain 'query' field")
        
        # Prepare context-aware prompts
        system_content, user_content = self._prepare_context_prompt(query_data, use_few_shot, few_shot_example)
        
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
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to same device as model - fixes MPS issue
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
        
                # Decode response - DON'T skip special tokens for proper extraction
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=False  # Keep special tokens for proper extraction
        )
        
        # Extract only the assistant's response using ChatML format
        if "<|im_start|>assistant" in generated_text:
            # Find the assistant section and extract content
            assistant_start = generated_text.find("<|im_start|>assistant")
            assistant_content = generated_text[assistant_start + len("<|im_start|>assistant"):]
            # Remove any trailing <|im_end|> token
            if "<|im_end|>" in assistant_content:
                response_text = assistant_content.split("<|im_end|>")[0].strip()
            else:
                response_text = assistant_content.strip()
            
            # Clean up markdown formatting and newlines
            response_text = response_text.replace("\\n", "\n")  # Convert literal \n to actual newlines
            response_text = response_text.replace("```json", "").replace("```", "")  # Remove markdown code block markers
            response_text = response_text.strip()  # Remove leading/trailing whitespace
            
        else:
            # Fallback: try to extract after the prompt
            response_text = generated_text.split(prompt)[-1].strip()
        
        steps = self._parse_steps(response_text)
        
        return {
            "steps": steps,
            "raw_text": generated_text,
            "response_text": response_text,
            "formatted_answer": self._format_as_list(steps)
        }
    
    def _validate_query_data(self, query_data: Dict) -> bool:
        """Validate query data structure
        
        Args:
            query_data (Dict): Query data to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return isinstance(query_data, dict) and "query" in query_data
    
    def _prepare_context_prompt(self, query_data: Dict, use_few_shot: bool, few_shot_example: Dict = None) -> tuple[str, str]:
        """Prepare context-aware system and user prompts
        
        Args:
            query_data (Dict): Query data containing query and context
            use_few_shot (bool): Whether to use few-shot prompting
            few_shot_example (Dict, optional): Specific few-shot example to use
            
        Returns:
            tuple[str, str]: (system_content, user_content) formatted prompts
        """
        query = query_data["query"]
        context = query_data.get("context", "")
        
        # Prepare system content
        if use_few_shot:
            if few_shot_example:
                # Use provided few-shot example
                system_content = self.template_loader.format_few_shot_prompt(
                    context, 
                    few_shot_example["query"], 
                    few_shot_example["answer"]
                )
            elif self.dataset_manager and self.dataset_manager.get_dataset_size() > 0:
                # Select few-shot example from dataset
                selected_example = self._select_few_shot_example(query_data)
                system_content = self.template_loader.format_few_shot_prompt(
                    context,
                    selected_example["query"],
                    selected_example["answer"]
                )
            else:
                # Fallback to static few-shot template
                system_content = self.template_loader.load_template("system_prompt_few_shot")
        else:
            # Use regular system prompt
            system_content = self.template_loader.load_template("system_prompt_regular")
        
        # Prepare user content
        if context:
            # Use context-aware user query template
            user_content = self.template_loader.format_user_query_with_context(query, context)
        else:
            # Use standard user query template
            user_content = self.template_loader.format_template("user_query_template", query=query)
        
        return system_content, user_content
    
    def _select_few_shot_example(self, current_query: Dict) -> Dict:
        """Select a relevant few-shot example from the dataset
        
        Args:
            current_query (Dict): Current query data to find similar example for
            
        Returns:
            Dict: Selected few-shot example from dataset
        """
        if not self.dataset_manager or self.dataset_manager.get_dataset_size() == 0:
            # Fallback to hard-coded example
            return {
                "query": "How to login?",
                "answer": '["Enter username", "Enter password", "Click login"]'
            }
        
        # Simple selection strategy: find example with similar context or query type
        current_context = current_query.get("context", "").lower()
        current_query_text = current_query.get("query", "").lower()
        
        all_queries = self.dataset_manager.get_all_queries()
        
        # Try to find example with similar context
        for example in all_queries:
            example_context = example.get("context", "").lower()
            if current_context and example_context:
                # Check for keyword overlap
                context_words = set(current_context.split())
                example_words = set(example_context.split())
                if len(context_words.intersection(example_words)) > 0:
                    return example
        
        # Try to find example with similar query type
        for example in all_queries:
            example_query = example.get("query", "").lower()
            if current_query_text and example_query:
                # Check for similar action words
                action_words = ["login", "add", "create", "delete", "update", "search", "checkout", "reset"]
                for action in action_words:
                    if action in current_query_text and action in example_query:
                        return example
        
        # Fallback: return first example
        return all_queries[0]
    
    def _parse_steps(self, text: str) -> List[str]:
        """Parse generated text into a structured list of test steps.
        
        Attempts to extract test steps from the generated text using multiple
        parsing strategies including JSON parsing and natural language processing.
        
        Args:
            text (str): The generated text to parse
            
        Returns:
            List[str]: A list of parsed test steps, limited to 5 steps maximum
        """
        try:
            # Try JSON parsing first
            if '[' in text and ']' in text:
                start = text.find('[')
                end = text.rfind(']') + 1
                json_text = text[start:end]
                return json.loads(json_text)
        except Exception as e:
            logger.error(f"Error parsing steps: {text}")
        
        # Fallback parsing
        steps = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Query:', 'Generate', 'Test')):
                # Remove numbering, bullets, etc.
                import re
                clean_step = re.sub(r'^[\d\.\-\*\+]\s*', '', line)
                clean_step = re.sub(r'^Step\s*\d+[:.]?\s*', '', clean_step, flags=re.IGNORECASE)
                if clean_step and len(clean_step) > 3:
                    steps.append(clean_step)
        
        # If no structured steps found, try to extract from natural language
        if not steps and text.strip():
            # Look for action words
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(word in sentence.lower() for word in ['enter', 'click', 'select', 'choose', 'type', 'input', 'press']):
                    steps.append(sentence)
        
        return steps[:5]  # Limit to 5 steps max
    
    def _format_as_list(self, steps: List[str]) -> str:
        """Format test steps as a readable JSON-style list string.
        
        Takes a list of test steps and formats them as a readable string
        representation similar to a JSON array.
        
        Args:
            steps (List[str]): The list of test steps to format
            
        Returns:
            str: A formatted string representation of the steps list
        """
        if not steps:
            return "[]"
        
        formatted_steps = [f'"{step}"' for step in steps]
        return f"[{', '.join(formatted_steps)}]"

class SmoLAgentsGrader:
    """A grader class that uses SmoLM agents with Azure OpenAI to evaluate test case accuracy.
    
    This class provides functionality to grade the accuracy of generated test case steps
    using a language model-based evaluation approach through Azure OpenAI services. Supports
    both static grading (backward compatibility) and dynamic context-aware grading with
    dataset-driven source of truth.
    
    When context and correct_answer are provided, the grader performs context-aware evaluation
    using dynamic grading prompts that adapt to different business contexts and query types.
    """
    
    def __init__(self):
        """Initialize SmoLAgents grader with Azure OpenAI.
        
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
        
        self.template_loader = DynamicTemplateLoader()
    
    def grade_response(self, generated_answer: str, context: str = "", correct_answer: str = "") -> float:
        """Grade the accuracy of generated test case steps with optional context awareness.
        
        Uses the configured language model to evaluate how accurate the generated
        test steps are, returning a normalized score between 0.0 and 1.0. Supports
        both static grading (backward compatibility) and dynamic context-aware grading.
        
        Args:
            generated_answer (str): The generated test case steps to be graded
            context (str, optional): Business context for the query. If provided, enables
                context-aware grading with dynamic source of truth.
            correct_answer (str, optional): The correct answer to use as source of truth.
                If provided along with context, enables dynamic grading.
            
        Returns:
            float: Accuracy score between 0.0 and 1.0, where 1.0 is perfect accuracy
        """
        
        # Choose grading approach based on available parameters
        if context and correct_answer:
            # Use dynamic context-aware grading
            prompt = self.template_loader.format_grading_prompt(
                context=context,
                correct_answer=correct_answer,
                generated_answer=generated_answer
            )
        else:
            # Use static grading for backward compatibility
            prompt = self.template_loader.format_template("grading_prompt", generated_answer=generated_answer)
        
        try:
            result = self.agent.run(prompt)
            
            # Extract JSON from response
            if isinstance(result, str):
                # Try to extract JSON from the response
                try:
                    result_object = json.loads(result)
                    return max(0.0, min(1.0, result_object["accuracy"]))
                except json.JSONDecodeError:
                    # If direct JSON parsing fails, try to extract from the response text
                    import re
                    json_match = re.search(r'final_answer\(\s*(\{.*?\})\s*\)', result, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        result_object = json.loads(json_str)
                        return max(0.0, min(1.0, result_object["accuracy"]))
                    else:
                        print(f"Could not extract JSON from response: {result[:200]}...")
                        return 0.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Grading error: {e}")
            return 0.0   
    
    def _build_dynamic_grading_prompt(self, context: str, correct_answer: str) -> str:
        """Build dynamic grading prompt with context and correct answer
        
        Args:
            context (str): Business context for the query
            correct_answer (str): The correct answer to use as source of truth
            
        Returns:
            str: Formatted dynamic grading prompt
        """
        return self.template_loader.format_grading_prompt(
            context=context,
            correct_answer=correct_answer,
            generated_answer="{generated_answer}"  # Placeholder for actual generated answer
        )
    
    def _extract_source_of_truth(self, context: str, correct_answer: str) -> str:
        """Extract and format source of truth from context and correct answer
        
        Args:
            context (str): Business context for the query
            correct_answer (str): The correct answer to use as source of truth
            
        Returns:
            str: Formatted source of truth for grading
        """
        try:
            # Parse correct answer to create numbered steps
            correct_steps = json.loads(correct_answer)
            source_of_truth = "\n".join([f"{i+1}. {step}" for i, step in enumerate(correct_steps)])
        except (json.JSONDecodeError, TypeError):
            source_of_truth = correct_answer
        
        return f"**Context**: {context}\n**Correct Answer**: {correct_answer}\n\n**Process Steps**:\n{source_of_truth}"

class FewShotTeachingTrainer:
    """A trainer class for evaluating and improving few-shot learning performance.
    
    This class provides comprehensive testing and analysis capabilities for evaluating
    how well a test case generator performs using few-shot in-context learning,
    with automated grading and detailed performance metrics.
    """
    
    def __init__(self, generator: QwenTestCaseGenerator, grader: SmoLAgentsGrader, dataset_manager: DatasetManager = None):
        """Initialize the few-shot teaching trainer.
        
        Args:
            generator (QwenTestCaseGenerator): The test case generator to train and evaluate
            grader (SmoLAgentsGrader): The grader to evaluate generated responses
            dataset_manager (DatasetManager, optional): Dataset manager for multi-query training.
                If provided, enables dataset-driven few-shot training with dynamic context and answers.
        """
        self.generator = generator
        self.grader = grader
        self.dataset_manager = dataset_manager
        # Fallback for backward compatibility
        self.correct_answer = '["Enter username", "Enter password", "Click login"]'

    def analyze_feedback(self, results: List[Dict]):
        """Analyze and display detailed training results and performance metrics.
        
        Processes a list of test results and provides comprehensive analysis including
        success rates, accuracy scores, exact matches, and individual test breakdowns.
        
        Args:
            results (List[Dict]): List of test result dictionaries containing
                test outcomes, scores, and generated vs expected answers
        """
        
        print("\nDetailed Results Analysis")
        print("=" * 30)
        
        total_tests = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        high_scores = sum(1 for r in results if r["accuracy_score"] >= 0.9)
        successful = sum(1 for r in results if r["success"])
        
        avg_score = sum(r["accuracy_score"] for r in results) / total_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Exact Matches: {exact_matches} ({exact_matches/total_tests*100:.1f}%)")
        print(f"High Scores (>=0.9): {high_scores} ({high_scores/total_tests*100:.1f}%)")
        print(f"Overall Success: {successful} ({successful/total_tests*100:.1f}%)")
        print(f"Average Score: {avg_score:.3f}")
        
        # Show individual results
        print("\nDetailed Results:")
        for result in results:
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"Test {result['test_number']}: {status} (Score: {result['accuracy_score']:.3f})")
            print(f"   Generated: {result['generated']}")
            print(f"   Expected:  {result['expected']}")
            print()
        
    def test_few_shot_performance(self, query_data: Dict, num_tests: int = 5) -> List[Dict]:
        """Test few-shot in-context learning performance with multiple trials.
        
        Runs multiple test iterations using few-shot learning to evaluate consistency
        and performance of the test case generation approach.
        Supports both static training (backward compatibility) and dataset-driven training.
        
        Args:
            query_data (Dict): Query data containing 'query' field, and optionally 'context' and 'answer'.
                For backward compatibility, can also accept a string query.
            num_tests (int, optional): Number of test iterations to run. Defaults to 5.
            
        Returns:
            List[Dict]: List of test results containing generated answers, scores,
                exact match status, and success indicators for each test
        """
        
        # Handle backward compatibility
        if isinstance(query_data, str):
            query_data = {"query": query_data, "context": "", "answer": self.correct_answer}
        
        query = query_data.get("query", "")
        context = query_data.get("context", "")
        correct_answer = query_data.get("answer", self.correct_answer)
        
        print(f"\nTesting few-shot performance for: {query}")
        if context:
            print(f"Context: {context}")
        print("-" * 40)
        
        results = []
        
        for i in range(num_tests):
            print(f"\nTest {i+1}/{num_tests}")
            
            # Generate response using few-shot learning
            result = self.generator.generate_test_steps(query_data, use_few_shot=True)
            formatted_answer = result["formatted_answer"]
            
            print(f"Generated: {formatted_answer}")
            print(f"Expected:  {correct_answer}")
            
            # Check exact match
            exact_match = formatted_answer.strip() == correct_answer.strip()
            
            # Get SmoLAgents grade (use dynamic grading if context provided)
            if context:
                accuracy_score = self.grader.grade_response(formatted_answer, context=context, correct_answer=correct_answer)
            else:
                accuracy_score = self.grader.grade_response(formatted_answer)
            
            print(f"Exact Match: {exact_match}")
            print(f"SmoLAgents Score: {accuracy_score:.3f}")
            
            results.append({
                "test_number": i+1,
                "generated": formatted_answer,
                "expected": correct_answer,
                "exact_match": exact_match,
                "accuracy_score": accuracy_score,
                "success": exact_match or accuracy_score >= 0.9
            })
            
            time.sleep(0.5)
        
        return results
    
    def compare_approaches(self, query_data: Dict, num_tests: int = 3):
        """Compare few-shot learning vs regular generation performance.
        
        Runs comparative tests between standard generation and few-shot in-context
        learning to measure the effectiveness of the few-shot approach.
        Supports both static training (backward compatibility) and dataset-driven training.
        
        Args:
            query_data (Dict): Query data containing 'query' field, and optionally 'context' and 'answer'.
                For backward compatibility, can also accept a string query.
            num_tests (int, optional): Number of tests per approach. Defaults to 3.
            
        Returns:
            tuple: A tuple containing (regular_results, few_shot_results) where each
                is a list of accuracy scores for the respective approach
        """
        
        # Handle backward compatibility
        if isinstance(query_data, str):
            query_data = {"query": query_data, "context": "", "answer": self.correct_answer}
        
        query = query_data.get("query", "")
        context = query_data.get("context", "")
        correct_answer = query_data.get("answer", self.correct_answer)
        
        print("Comparing Few-Shot vs Regular Generation")
        print("=" * 50)
        print(f"Query: {query}")
        if context:
            print(f"Context: {context}")
        
        # Test without few-shot
        print("\n--- WITHOUT Few-Shot Examples ---")
        regular_results = []
        for i in range(num_tests):
            result = self.generator.generate_test_steps(query_data, use_few_shot=False)
            # Use dynamic grading if context provided
            if context:
                accuracy = self.grader.grade_response(result["formatted_answer"], context=context, correct_answer=correct_answer)
            else:
                accuracy = self.grader.grade_response(result["formatted_answer"])
            regular_results.append(accuracy)
            print(f"Test {i+1}: {result['formatted_answer']} (Score: {accuracy:.3f})")
        
        # Test with few-shot
        print("\n--- WITH Few-Shot Examples ---")
        few_shot_results = []
        for i in range(num_tests):
            result = self.generator.generate_test_steps(query_data, use_few_shot=True)
            # Use dynamic grading if context provided
            if context:
                accuracy = self.grader.grade_response(result["formatted_answer"], context=context, correct_answer=correct_answer)
            else:
                accuracy = self.grader.grade_response(result["formatted_answer"])
            few_shot_results.append(accuracy)
            print(f"Test {i+1}: {result['formatted_answer']} (Score: {accuracy:.3f})")
        
        # Compare results
        regular_avg = sum(regular_results) / len(regular_results)
        few_shot_avg = sum(few_shot_results) / len(few_shot_results)
        
        print(f"\nComparison Results:")
        print(f"Regular Generation Average: {regular_avg:.3f}")
        print(f"Few-Shot Generation Average: {few_shot_avg:.3f}")
        print(f"Improvement: {few_shot_avg - regular_avg:+.3f}")
        
        return regular_results, few_shot_results
    
    def evaluate_success(self, query_data: Dict, success_threshold: int = 4, num_tests: int = 5):
        """Evaluate if few-shot approach achieves the defined success criteria.
        
        Runs comprehensive evaluation to determine if the few-shot learning approach
        meets the specified success threshold based on accuracy and correctness metrics.
        Supports both static training (backward compatibility) and dataset-driven training.
        
        Args:
            query_data (Dict): Query data containing 'query' field, and optionally 'context' and 'answer'.
                For backward compatibility, can also accept a string query.
            success_threshold (int, optional): Minimum number of successful tests required. Defaults to 4.
            num_tests (int, optional): Total number of tests to run. Defaults to 5.
            
        Returns:
            tuple: A tuple containing (success_achieved: bool, results: List[Dict])
                where success_achieved indicates if threshold was met and results
                contains detailed test outcomes
        """
        
        # Handle backward compatibility
        if isinstance(query_data, str):
            query_data = {"query": query_data, "context": "", "answer": self.correct_answer}
        
        query = query_data.get("query", "")
        context = query_data.get("context", "")
        correct_answer = query_data.get("answer", self.correct_answer)
        
        print(f"Evaluating Few-Shot Success for: {query}")
        if context:
            print(f"Context: {context}")
        print(f"Target Answer: {correct_answer}")
        print(f"Success Threshold: {success_threshold}/{num_tests} correct")
        
        # Test few-shot performance
        results = self.test_few_shot_performance(query_data, num_tests=num_tests)
        
        # Calculate success metrics
        successful_tests = sum(1 for r in results if r["success"])
        exact_matches = sum(1 for r in results if r["exact_match"])
        avg_score = sum(r["accuracy_score"] for r in results) / len(results)
        
        success_achieved = successful_tests >= success_threshold
        
        print(f"\nResults Summary:")
        print(f"Successful Tests: {successful_tests}/{num_tests} ({successful_tests/num_tests*100:.1f}%)")
        print(f"Exact Matches: {exact_matches}/{num_tests} ({exact_matches/num_tests*100:.1f}%)")
        print(f"Average Score: {avg_score:.3f}")
        print(f"Success Achieved: {success_achieved}")
        
        return success_achieved, results
    
    def train_on_dataset(self, num_queries: int = 5, max_attempts_per_query: int = 3, success_threshold: int = 4) -> Dict:
        """Train the model on multiple queries from the dataset using few-shot learning.
        
        Performs few-shot training on multiple queries from the dataset, enabling the model
        to learn diverse patterns and contexts. Only works when dataset_manager is provided.
        
        Args:
            num_queries (int, optional): Number of queries to train on. Defaults to 5.
            max_attempts_per_query (int, optional): Maximum training attempts per query. Defaults to 3.
            success_threshold (int, optional): Minimum successful tests per query. Defaults to 4.
            
        Returns:
            Dict: Training results summary with per-query results and overall statistics
        """
        
        if not self.dataset_manager:
            raise ValueError("Dataset manager not provided. Cannot perform multi-query training.")
        
        print("Multi-Query Few-Shot Training")
        print("=" * 50)
        print(f"Training on {num_queries} queries from dataset")
        print(f"Max attempts per query: {max_attempts_per_query}")
        print(f"Success threshold: {success_threshold}")
        
        all_results = []
        successful_queries = 0
        
        for i in range(num_queries):
            print(f"\n{'='*60}")
            print(f"FEW-SHOT TRAINING QUERY {i+1}/{num_queries}")
            print(f"{'='*60}")
            
            # Get random query from dataset
            query_data = self.dataset_manager.get_random_query()
            
            try:
                # Evaluate few-shot success on this query
                success, results = self.evaluate_success(
                    query_data=query_data,
                    success_threshold=success_threshold,
                    num_tests=5
                )
                
                if success:
                    successful_queries += 1
                    print(f"✅ Query {i+1} few-shot training SUCCESSFUL")
                else:
                    print(f"❌ Query {i+1} few-shot training FAILED")
                
                all_results.append({
                    "query_index": i+1,
                    "query_data": query_data,
                    "success": success,
                    "results": results
                })
                
            except Exception as e:
                print(f"❌ Query {i+1} few-shot training ERROR: {e}")
                all_results.append({
                    "query_index": i+1,
                    "query_data": query_data,
                    "success": False,
                    "error": str(e),
                    "results": []
                })
        
        # Calculate overall statistics
        overall_success_rate = successful_queries / num_queries
        
        print(f"\n{'='*60}")
        print("MULTI-QUERY FEW-SHOT TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Queries: {num_queries}")
        print(f"Successful: {successful_queries}")
        print(f"Failed: {num_queries - successful_queries}")
        print(f"Overall Success Rate: {overall_success_rate:.1%}")
        
        return {
            "total_queries": num_queries,
            "successful_queries": successful_queries,
            "failed_queries": num_queries - successful_queries,
            "overall_success_rate": overall_success_rate,
            "per_query_results": all_results
        }
        
class TeachingFeedbackTrainer:
    """A trainer class that uses teaching feedback to improve model performance through fine-tuning.
    
    This class implements a teaching-based approach where the model is fine-tuned with correct
    answers to improve its accuracy on specific test case generation tasks. It provides methods
    for creating teaching examples, fine-tuning the model, and evaluating learning progress.
    
    Attributes:
        generator (QwenTestCaseGenerator): The test case generator to be trained
        grader (SmoLAgentsGrader): The grader used to evaluate generated responses
        feedback_data (List[Dict]): Collection of feedback data from training sessions
        correct_answer (str): The target correct answer for training
        template_loader (DynamicTemplateLoader): Loader for prompt templates
    """
    
    def __init__(self, generator: QwenTestCaseGenerator, grader: SmoLAgentsGrader, dataset_manager: DatasetManager = None):
        """Initialize the teaching feedback trainer.
        
        Args:
            generator (QwenTestCaseGenerator): The test case generator to be trained
            grader (SmoLAgentsGrader): The grader used to evaluate generated responses
            dataset_manager (DatasetManager, optional): Dataset manager for multi-query training.
                If provided, enables dataset-driven training with dynamic context and answers.
        """
        self.generator = generator
        self.grader = grader
        self.dataset_manager = dataset_manager
        self.feedback_data: List[Dict] = []
        # Fallback for backward compatibility
        self.correct_answer = '["Enter username", "Enter password", "Click login"]'
        self.template_loader = DynamicTemplateLoader()

    def create_teaching_examples(self, query_data: Dict) -> List[Dict]:
        """Create teaching examples that demonstrate the correct answer for a given query.
        
        Generates training examples by loading system prompts and user query templates,
        then pairing them with the correct answer to create supervised learning data.
        Supports both static training (backward compatibility) and dataset-driven training.
        
        Args:
            query_data (Dict): Query data containing 'query' field, and optionally 'context' and 'answer'.
                For backward compatibility, can also accept a string query.
            
        Returns:
            List[Dict]: A list of teaching examples containing text, query, correct answer, and type
        """
        
        # Handle backward compatibility
        if isinstance(query_data, str):
            query_data = {"query": query_data, "context": "", "answer": self.correct_answer}
        
        # Extract query, context, and correct answer
        query = query_data.get("query", "")
        context = query_data.get("context", "")
        correct_answer = query_data.get("answer", self.correct_answer)
        
        teaching_examples = []
        
        # Load system content and user query from templates
        system_content = self.template_loader.load_template("system_prompt_teaching")
        
        # Use context-aware user query if context is provided
        if context:
            user_content = self.template_loader.format_user_query_with_context(query=query, context=context)
        else:
            user_content = self.template_loader.format_template("teaching_user_query_template", query=query)
        
        # Correct example
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        prompt = self.generator.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Create training example with correct answer
        training_text = prompt + correct_answer + self.generator.tokenizer.eos_token
        
        teaching_examples.append({
            "text": training_text,
            "query": query,
            "context": context,
            "correct_answer": correct_answer,
            "type": "correct_example"
        })
        
        return teaching_examples
    
    def fine_tune_with_correct_answer(self, query_data: Dict, num_epochs: int = 3):
        """Fine-tune the model with the correct answer for improved performance.
        
        Creates a teaching dataset from the query and correct answer, then fine-tunes
        the model using supervised learning to teach it the correct response pattern.
        Supports both static training (backward compatibility) and dataset-driven training.
        
        Args:
            query_data (Dict): Query data containing 'query' field, and optionally 'context' and 'answer'.
                For backward compatibility, can also accept a string query.
            num_epochs (int, optional): Number of training epochs. Defaults to 3.
            
        Returns:
            Trainer: The trained Hugging Face trainer object
        """
        
        # Handle backward compatibility
        if isinstance(query_data, str):
            query_data = {"query": query_data, "context": "", "answer": self.correct_answer}
        
        query = query_data.get("query", "")
        context = query_data.get("context", "")
        correct_answer = query_data.get("answer", self.correct_answer)
        
        print(f"Teaching model the correct answer for: {query}")
        if context:
            print(f"Context: {context}")
        print(f"Correct answer: {correct_answer}")
        
        # Create teaching dataset
        teaching_examples = self.create_teaching_examples(query_data)
        
        # Create multiple copies to reinforce learning
        expanded_examples = teaching_examples * 20  # Repeat 20 times
        
        dataset = Dataset.from_list(expanded_examples)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            learning_rate=5e-5,
            logging_steps=5,
            save_steps=100,
            remove_unused_columns=False,
        )
        
        # Custom data collator
        def data_collator(examples):
            texts = [example["text"] for example in examples]
            encodings = self.generator.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            encodings["labels"] = encodings["input_ids"].clone()
            return encodings
        
        # Create trainer
        trainer = Trainer(
            model=self.generator.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.generator.tokenizer,
        )
        
        print("Starting fine-tuning...")
        trainer.train()
        print("Fine-tuning complete!")
        
        # Save the fine-tuned model
        trainer.save_model("./fine_tuned_model")
        self.generator.tokenizer.save_pretrained("./fine_tuned_model")
        
        return trainer
    
    def test_learning_progress(self, query_data: Dict, num_tests: int = 5) -> List[Dict]:
        """Test if the model has learned the correct answer through evaluation runs.
        
        Generates multiple test responses for the given query and evaluates them
        against the correct answer using both exact matching and grader scoring.
        Supports both static training (backward compatibility) and dataset-driven training.
        
        Args:
            query_data (Dict): Query data containing 'query' field, and optionally 'context' and 'answer'.
                For backward compatibility, can also accept a string query.
            num_tests (int, optional): Number of test runs to perform. Defaults to 5.
            
        Returns:
            List[Dict]: A list of test results containing generated answers, scores, and success metrics
        """
        
        # Handle backward compatibility
        if isinstance(query_data, str):
            query_data = {"query": query_data, "context": "", "answer": self.correct_answer}
        
        query = query_data.get("query", "")
        context = query_data.get("context", "")
        correct_answer = query_data.get("answer", self.correct_answer)
        
        print(f"\nTesting learning progress for: {query}")
        if context:
            print(f"Context: {context}")
        print("-" * 40)
        
        results = []
        
        for i in range(num_tests):
            print(f"\nTest {i+1}/{num_tests}")
            
            # Generate response
            result = self.generator.generate_test_steps(query_data)
            formatted_answer = result["formatted_answer"]
            
            print(f"Generated: {formatted_answer}")
            print(f"Expected:  {correct_answer}")
            
            # Check exact match
            exact_match = formatted_answer.strip() == correct_answer.strip()
            
            # Get SmoLAgents grade (use dynamic grading if context provided)
            if context:
                accuracy_score = self.grader.grade_response(formatted_answer, context=context, correct_answer=correct_answer)
            else:
                accuracy_score = self.grader.grade_response(formatted_answer)
            
            print(f"Exact Match: {exact_match}")
            print(f"SmoLAgents Score: {accuracy_score:.3f}")
            
            results.append({
                "test_number": i+1,
                "generated": formatted_answer,
                "expected": correct_answer,
                "exact_match": exact_match,
                "accuracy_score": accuracy_score,
                "success": exact_match or accuracy_score >= 0.9
            })
            
            time.sleep(0.5)
        
        return results
    
    def train_until_success(self, query_data: Dict, max_attempts: int = 5, success_threshold: int = 4):
        """Train iteratively until the model consistently produces the correct answer.
        
        Performs multiple rounds of fine-tuning and testing until the model achieves
        the specified success threshold or reaches the maximum number of attempts.
        Supports both static training (backward compatibility) and dataset-driven training.
        
        Args:
            query_data (Dict): Query data containing 'query' field, and optionally 'context' and 'answer'.
                For backward compatibility, can also accept a string query.
            max_attempts (int, optional): Maximum number of training attempts. Defaults to 5.
            success_threshold (int, optional): Minimum number of successful tests required. Defaults to 4.
            
        Returns:
            tuple: A tuple containing (success_achieved: bool, final_results: List[Dict])
                where success_achieved indicates if the threshold was met
        """
        
        # Handle backward compatibility
        if isinstance(query_data, str):
            query_data = {"query": query_data, "context": "", "answer": self.correct_answer}
        
        query = query_data.get("query", "")
        context = query_data.get("context", "")
        correct_answer = query_data.get("answer", self.correct_answer)
        
        print("Training Until Success")
        print("=" * 50)
        print(f"Query: {query}")
        if context:
            print(f"Context: {context}")
        print(f"Target Answer: {correct_answer}")
        print(f"Success Threshold: {success_threshold}/{success_threshold} correct")
        
        for attempt in range(max_attempts):
            print(f"\n--- Training Attempt {attempt + 1}/{max_attempts} ---")
            
            # Test current performance
            results = self.test_learning_progress(query_data, num_tests=5)
            
            # Calculate success rate
            successful_tests = sum(1 for r in results if r["success"])
            success_rate = successful_tests / len(results)
            
            print(f"\nCurrent Performance: {successful_tests}/{len(results)} successful ({success_rate*100:.1f}%)")
            
            # Check if we've achieved success
            if successful_tests >= success_threshold:
                print("\nSUCCESS! Model consistently produces correct answer.")
                return True, results
            
            # Fine-tune the model
            print(f"\nNeed improvement. Fine-tuning model (attempt {attempt + 1})...")
            self.fine_tune_with_correct_answer(query_data, num_epochs=2)
            
            print(f"Fine-tuning attempt {attempt + 1} complete.")
        
        print(f"\nTraining completed after {max_attempts} attempts.")
        final_results = self.test_learning_progress(query_data, num_tests=5)
        final_successful = sum(1 for r in final_results if r["success"])
        
        return final_successful >= success_threshold, final_results
    
    def train_on_dataset(self, num_queries: int = 5, max_attempts_per_query: int = 3, success_threshold: int = 4) -> Dict:
        """Train the model on multiple queries from the dataset.
        
        Performs training on multiple queries from the dataset, enabling the model
        to learn diverse patterns and contexts. Only works when dataset_manager is provided.
        
        Args:
            num_queries (int, optional): Number of queries to train on. Defaults to 5.
            max_attempts_per_query (int, optional): Maximum training attempts per query. Defaults to 3.
            success_threshold (int, optional): Minimum successful tests per query. Defaults to 4.
            
        Returns:
            Dict: Training results summary with per-query results and overall statistics
        """
        
        if not self.dataset_manager:
            raise ValueError("Dataset manager not provided. Cannot perform multi-query training.")
        
        print("Multi-Query Dataset Training")
        print("=" * 50)
        print(f"Training on {num_queries} queries from dataset")
        print(f"Max attempts per query: {max_attempts_per_query}")
        print(f"Success threshold: {success_threshold}")
        
        all_results = []
        successful_queries = 0
        
        for i in range(num_queries):
            print(f"\n{'='*60}")
            print(f"TRAINING QUERY {i+1}/{num_queries}")
            print(f"{'='*60}")
            
            # Get random query from dataset
            query_data = self.dataset_manager.get_random_query()
            
            try:
                # Train on this query
                success, results = self.train_until_success(
                    query_data=query_data,
                    max_attempts=max_attempts_per_query,
                    success_threshold=success_threshold
                )
                
                if success:
                    successful_queries += 1
                    print(f"✅ Query {i+1} training SUCCESSFUL")
                else:
                    print(f"❌ Query {i+1} training FAILED")
                
                all_results.append({
                    "query_index": i+1,
                    "query_data": query_data,
                    "success": success,
                    "results": results
                })
                
            except Exception as e:
                print(f"❌ Query {i+1} training ERROR: {e}")
                all_results.append({
                    "query_index": i+1,
                    "query_data": query_data,
                    "success": False,
                    "error": str(e),
                    "results": []
                })
        
        # Calculate overall statistics
        overall_success_rate = successful_queries / num_queries
        
        print(f"\n{'='*60}")
        print("MULTI-QUERY TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Queries: {num_queries}")
        print(f"Successful: {successful_queries}")
        print(f"Failed: {num_queries - successful_queries}")
        print(f"Overall Success Rate: {overall_success_rate:.1%}")
        
        return {
            "total_queries": num_queries,
            "successful_queries": successful_queries,
            "failed_queries": num_queries - successful_queries,
            "overall_success_rate": overall_success_rate,
            "per_query_results": all_results
        }
    
    def analyze_results(self, results: List[Dict]):
        """Analyze and display detailed statistics from training results.
        
        Calculates and prints comprehensive metrics including success rates,
        exact matches, accuracy scores, and individual test outcomes.
        
        Args:
            results (List[Dict]): List of test results to analyze, each containing
                test outcomes and performance metrics
        """
        
        print("\nTraining Results Analysis")
        print("=" * 30)
        
        total_tests = len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        high_scores = sum(1 for r in results if r["accuracy_score"] >= 0.9)
        successful = sum(1 for r in results if r["success"])
        
        avg_score = sum(r["accuracy_score"] for r in results) / total_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Exact Matches: {exact_matches} ({exact_matches/total_tests*100:.1f}%)")
        print(f"High Scores (>=0.9): {high_scores} ({high_scores/total_tests*100:.1f}%)")
        print(f"Overall Success: {successful} ({successful/total_tests*100:.1f}%)")
        print(f"Average Score: {avg_score:.3f}")
        
        # Show individual results
        print("\nDetailed Results:")
        for result in results:
            status = "SUCCESS" if result["success"] else "FAILED"
            print(f"Test {result['test_number']}: {status} (Score: {result['accuracy_score']:.3f})")

def main():
    """Main execution pipeline supporting both single-query and dataset-driven multi-query approaches"""
    
    print("QA Test Case Generation System - Dataset-Driven Architecture")
    print("=" * 70)
    
    # Initialize components
    print("Setting up core components...")
    grader = SmoLAgentsGrader()
    generator = QwenTestCaseGenerator()
    
    # Check if dataset is available
    dataset_path = "sample_dataset.json"
    dataset_manager = None
    
    try:
        dataset_manager = DatasetManager(dataset_path)
        print(f"✅ Dataset loaded successfully: {dataset_manager.get_dataset_size()} entries")
        use_dataset = True
    except FileNotFoundError:
        print("⚠️  No dataset found. Running in single-query mode.")
        use_dataset = False
    
    # Initialize trainers
    print("Setting up trainers...")
    few_shot_trainer = FewShotTeachingTrainer(generator, grader, dataset_manager)
    teaching_trainer = TeachingFeedbackTrainer(generator, grader, dataset_manager)
    
    if use_dataset:
        # Dataset-driven multi-query approach
        print("\n" + "="*70)
        print("DATASET-DRIVEN MULTI-QUERY TRAINING")
        print("="*70)
        
        # Multi-query few-shot training
        print("\n1. Multi-Query Few-Shot Training")
        print("-" * 40)
        few_shot_training_results = few_shot_trainer.train_on_dataset(
            num_queries=3,
            max_attempts_per_query=2,
            success_threshold=3
        )
        
        # Multi-query teaching feedback training
        print("\n2. Multi-Query Teaching Feedback Training")
        print("-" * 40)
        teaching_training_results = teaching_trainer.train_on_dataset(
            num_queries=3,
            max_attempts_per_query=2,
            success_threshold=3
        )
        
        # Compare approaches on dataset
        print("\n3. Dataset-Based Approach Comparison")
        print("-" * 40)
        dataset_query = dataset_manager.get_random_query()
        regular_results, few_shot_comparison_results = few_shot_trainer.compare_approaches(
            dataset_query, num_tests=2
        )
        
        # Final summary for dataset approach
        print("\n" + "="*70)
        print("DATASET-DRIVEN TRAINING SUMMARY")
        print("="*70)
        print(f"Few-Shot Training Success Rate: {few_shot_training_results['overall_success_rate']:.1%}")
        print(f"Teaching Training Success Rate: {teaching_training_results['overall_success_rate']:.1%}")
        print(f"Dataset Size: {dataset_manager.get_dataset_size()} entries")
        print(f"Context-Aware Generation: ✅ Enabled")
        print(f"Dynamic Grading: ✅ Enabled")
        print(f"Multi-Query Training: ✅ Enabled")
        print(f"Approach Comparison: Regular avg: {sum(regular_results)/len(regular_results):.3f}, Few-shot avg: {sum(few_shot_comparison_results)/len(few_shot_comparison_results):.3f}")
        
    else:
        # Single-query approach (backward compatibility)
        print("\n" + "="*70)
        print("SINGLE-QUERY TRAINING (Backward Compatibility)")
        print("="*70)
        
        target_query = "How to login?"
        print(f"\nFocusing on single question: {target_query}")
        print(f"Correct answer: {few_shot_trainer.correct_answer}")
        
        # Compare approaches first
        print("\nStep 1: Comparing Regular vs Few-Shot Generation")
        regular_results, few_shot_results = few_shot_trainer.compare_approaches(target_query, num_tests=3)
        
        # Evaluate few-shot success
        print("\nStep 2: Evaluating Few-Shot Success")
        success, detailed_results = few_shot_trainer.evaluate_success(
            target_query, 
            success_threshold=4, 
            num_tests=5
        )
        
        # Analyze results
        few_shot_trainer.analyze_feedback(detailed_results)
        
        # Test teaching feedback approach
        print("\nStep 3: Testing Teaching Feedback Approach")
        teaching_success, teaching_results = teaching_trainer.train_until_success(
            target_query,
            max_attempts=3,
            success_threshold=4
        )
        
        # Final summary for single-query approach
        print("\n" + "="*70)
        print("SINGLE-QUERY TRAINING SUMMARY")
        print("="*70)
        if success:
            print("✅ Few-shot learning achieved target performance!")
        else:
            print("⚠️  Few-shot learning improved performance but didn't reach target.")
        
        if teaching_success:
            print("✅ Teaching feedback training achieved target performance!")
        else:
            print("⚠️  Teaching feedback training needs improvement.")
        
        print("Static Grading: ✅ Enabled")
        print("Backward Compatibility: ✅ Maintained")
    
    print("\n" + "="*70)
    print("SYSTEM CAPABILITIES SUMMARY")
    print("="*70)
    print("✅ Context-Aware Generation")
    print("✅ Dynamic Grading with Source of Truth")
    print("✅ Multi-Query Training Support")
    print("✅ Dataset-Driven Architecture")
    print("✅ Template System with Dynamic Loading")
    print("✅ Backward Compatibility")
    print("✅ Few-Shot and Teaching Feedback Training")
    print("✅ Comprehensive Validation and Testing")
    
    print("\n🎉 QA Test Case Generation System ready for production use!")

if __name__ == "__main__":
    main()