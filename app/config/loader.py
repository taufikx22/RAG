import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file and merge with environment variables.
    
    Returns:
        Configuration dictionary
    """
    config_path = Path(__file__).parent / "config.yaml"
    
    # Load base config from YAML
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
    
    # Apply environment variable overrides
    config = _apply_env_overrides(config)
    
    return config

def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.
    
    Args:
        config: Base configuration dictionary
        
    Returns:
        Updated configuration with environment overrides
    """
    # Embedding configuration
    embedding = config.setdefault("embedding", {})
    if os.getenv("EMBEDDING_MODEL"):
        embedding["model_name"] = os.getenv("EMBEDDING_MODEL")
    if os.getenv("EMBEDDING_DEFAULT_MODEL"):
        embedding["default_model"] = os.getenv("EMBEDDING_DEFAULT_MODEL")
    
    # Generation configuration
    generation = config.setdefault("generation", {})
    if os.getenv("OPENAI_API_KEY"):
        generation["openai_api_key"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("GOOGLE_API_KEY"):
        generation["google_api_key"] = os.getenv("GOOGLE_API_KEY")
    if os.getenv("GENERATION_DEFAULT_MODEL"):
        generation["default_model"] = os.getenv("GENERATION_DEFAULT_MODEL")
    if os.getenv("GENERATION_DEFAULT_TYPE"):
        generation["default_type"] = os.getenv("GENERATION_DEFAULT_TYPE")
    
    # Vector store configuration
    vector_store = config.setdefault("vector_store", {})
    if os.getenv("VECTOR_STORE_DEFAULT"):
        vector_store["default"] = os.getenv("VECTOR_STORE_DEFAULT")
    
    options = vector_store.setdefault("options", {})
    
    # Chroma configuration
    chroma = options.setdefault("chroma", {})
    if os.getenv("CHROMA_PERSIST_DIR"):
        chroma["persist_directory"] = os.getenv("CHROMA_PERSIST_DIR")
    if os.getenv("CHROMA_COLLECTION_NAME"):
        chroma["collection_name"] = os.getenv("CHROMA_COLLECTION_NAME")
    
    # Qdrant configuration
    qdrant = options.setdefault("qdrant", {})
    if os.getenv("QDRANT_URL"):
        qdrant["url"] = os.getenv("QDRANT_URL")
    if os.getenv("QDRANT_COLLECTION_NAME"):
        qdrant["collection_name"] = os.getenv("QDRANT_COLLECTION_NAME")
    if os.getenv("QDRANT_VECTOR_SIZE"):
        qdrant["vector_size"] = int(os.getenv("QDRANT_VECTOR_SIZE"))
    
    # Retrieval configuration
    retrieval = config.setdefault("retrieval", {})
    if os.getenv("RETRIEVAL_DEFAULT_STRATEGY"):
        retrieval["default_strategy"] = os.getenv("RETRIEVAL_DEFAULT_STRATEGY")
    
    # Evaluation configuration
    evaluation = config.setdefault("evaluation", {})
    if os.getenv("EVALUATION_ENABLED"):
        evaluation["enabled"] = os.getenv("EVALUATION_ENABLED").lower() == "true"
    
    return config

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "vector_store.options.chroma.persist_directory")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current
