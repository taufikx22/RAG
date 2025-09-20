import pytest
from unittest.mock import Mock, patch, MagicMock
from src.generation.gemini_generator import GeminiGenerator
from src.generation.base import GenerationResult


class TestGeminiGenerator:
    """Test cases for GeminiGenerator."""
    
    @patch('src.generation.gemini_generator.genai')
    def test_init_success(self, mock_genai):
        """Test successful initialization."""
        mock_genai.GenerativeModel.return_value = Mock()
        
        generator = GeminiGenerator(
            model="gemini-pro",
            api_key="test_key",
            temperature=0.5,
            max_tokens=512
        )
        
        assert generator.model == "gemini-pro"
        assert generator.temperature == 0.5
        assert generator.max_tokens == 512
        mock_genai.configure.assert_called_once_with(api_key="test_key")
    
    @patch('src.generation.gemini_generator.genai')
    def test_init_with_env_key(self, mock_genai):
        """Test initialization with environment variable."""
        mock_genai.GenerativeModel.return_value = Mock()
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'env_key'}):
            generator = GeminiGenerator()
            
        assert generator.model == "gemini-pro"
        mock_genai.configure.assert_called_once_with(api_key="env_key")
    
    def test_init_no_api_key(self):
        """Test initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="Google API key not provided"):
                GeminiGenerator()
    
    @patch('src.generation.gemini_generator.genai')
    def test_generate_success(self, mock_genai):
        """Test successful generation."""
        # Mock the model instance
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Mock the response
        mock_response = Mock()
        mock_response.text = "Generated response text"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = "STOP"
        
        mock_model.generate_content.return_value = mock_response
        
        # Mock generation config
        mock_config = Mock()
        mock_genai.types.GenerationConfig.return_value = mock_config
        
        generator = GeminiGenerator(api_key="test_key")
        result = generator.generate("Test question", "Test context")
        
        assert isinstance(result, GenerationResult)
        assert result.query == "Test question"
        assert result.context == "Test context"
        assert result.response == "Generated response text"
        assert result.metadata["model"] == "gemini-pro"
        assert result.metadata["provider"] == "Google Gemini"
    
    @patch('src.generation.gemini_generator.genai')
    def test_generate_with_custom_params(self, mock_genai):
        """Test generation with custom parameters."""
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        mock_response = Mock()
        mock_response.text = "Custom response"
        mock_response.usage_metadata = Mock()
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 15
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = "STOP"
        
        mock_model.generate_content.return_value = mock_response
        mock_genai.types.GenerationConfig.return_value = Mock()
        
        generator = GeminiGenerator(api_key="test_key")
        result = generator.generate(
            "Test question", 
            "Test context",
            temperature=0.8,
            max_tokens=256
        )
        
        assert result.metadata["temperature"] == 0.8
        assert result.metadata["max_tokens"] == 256
    
    @patch('src.generation.gemini_generator.genai')
    def test_get_model_info(self, mock_genai):
        """Test getting model information."""
        mock_genai.GenerativeModel.return_value = Mock()
        
        generator = GeminiGenerator(
            model="gemini-pro",
            api_key="test_key",
            temperature=0.3,
            max_tokens=1024
        )
        
        info = generator.get_model_info()
        
        assert info["provider"] == "Google"
        assert info["model"] == "gemini-pro"
        assert info["temperature"] == 0.3
        assert info["max_tokens"] == 1024
        assert info["type"] == "api"
