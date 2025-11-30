"""Simplified template loader for RL agent - no data embedding."""

from pathlib import Path
from typing import Dict


class TemplateLoader:
    """Simplified template loader that only loads templates without embedding data.
    
    Unlike DynamicTemplateLoader, this loader does NOT embed dataset content
    (examples, correct answers, etc.) into prompts. It only provides basic
    template loading and simple variable substitution for instructions.
    """
    
    def __init__(self, templates_dir: str = "templates"):
        """Initialize template loader.
        
        Args:
            templates_dir (str): Directory containing template files
        """
        self.templates_dir = Path(templates_dir)
    
    def load_template(self, template_name: str) -> str:
        """Load a template from a markdown file.
        
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
        """Load and format a template with provided variables.
        
        This is for simple variable substitution only (like {query}, {context}).
        It does NOT embed dataset examples or correct answers.
        
        Args:
            template_name (str): Name of the template file (without .md extension)
            **kwargs: Keyword arguments to use for template variable substitution
            
        Returns:
            str: Formatted template content with variables substituted
            
        Raises:
            FileNotFoundError: If template file doesn't exist
            KeyError: If template contains variables not provided in kwargs
        """
        template = self.load_template(template_name)
        return template.format(**kwargs)

