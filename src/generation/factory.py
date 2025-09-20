from typing import Dict, Type, Optional
import logging

from src.generation.base import BaseGenerator
from src.generation.openai_generator import OpenAIGenerator
from src.generation.local_generator import LocalLLMGenerator
from src.generation.gemini_generator import GeminiGenerator

logger = logging.getLogger(__name__)


class GeneratorFactory:
    """Factory for creating generator instances."""
    
    def __init__(self, config=None):
        """
        Initialize generator factory.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._generators: Dict[str, Type[BaseGenerator]] = {}
        
        # Register default generators
        self.register_generator("openai", OpenAIGenerator)
        self.register_generator("local", LocalLLMGenerator)
        self.register_generator("gemini", GeminiGenerator)
    
    def register_generator(self, name: str, generator_class: Type[BaseGenerator]) -> None:
        """Register a generator class."""
        self._generators[name.lower()] = generator_class
        logger.debug(f"Registered generator {generator_class.__name__} as '{name}'")
    
    def get_generator(self, model_type: str) -> Optional[BaseGenerator]:
        """
        Get a generator instance.
        
        Args:
            model_type: Type of generator to create
            
        Returns:
            Configured generator instance or None
        """
        model_type = model_type.lower()
        generator_class = self._generators.get(model_type)
        
        if not generator_class:
            logger.error(f"No generator found for type '{model_type}'")
            return None
        
        # Get configuration for this generator type
        generator_config = {}
        if self.config and "models" in self.config:
            # Find matching model configuration
            for model_config in self.config["models"]:
                if model_config.get("type") == model_type:
                    generator_config = {
                        k: v for k, v in model_config.items() 
                        if k not in ["name", "type"]
                    }
                    break
        
        try:
            return generator_class(**generator_config)
        except Exception as e:
            logger.error(f"Error creating generator '{model_type}': {str(e)}")
            return None
    
    def get_default_generator(self) -> BaseGenerator:
        """Get the default generator as specified in config."""
        default_model = "gpt-3.5-turbo"
        default_type = "openai"
        
        if self.config:
            default_model = self.config.get("default_model", default_model)
            default_type = self.config.get("default_type", default_type)
        
        # Find the generator type for this model
        generator_type = default_type
        if self.config and "models" in self.config:
            for model_config in self.config["models"]:
                if model_config.get("name") == default_model:
                    generator_type = model_config.get("type", generator_type)
                    break
        
        generator = self.get_generator(generator_type)
        if not generator:
            logger.warning(
                f"Default generator '{generator_type}' failed. "
                f"Trying 'openai'."
            )
            generator = self.get_generator("openai")
            
            if not generator:
                raise RuntimeError("Could not create any generator")
        
        return generator

