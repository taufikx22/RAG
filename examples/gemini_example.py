#!/usr/bin/env python3
"""
Example usage of the Gemini Generator.
This script demonstrates how to use the Gemini generator for text generation.
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

from generation.gemini_generator import GeminiGenerator


def main():
    """Example usage of Gemini generator."""
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not set!")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY=your_api_key_here")
        print("Or on Windows:")
        print("set GOOGLE_API_KEY=your_api_key_here")
        return
    
    try:
        # Initialize Gemini generator
        print("🚀 Initializing Gemini generator...")
        generator = GeminiGenerator(
            model="gemini-pro",
            api_key=api_key,
            temperature=0.2,
            max_tokens=1024
        )
        print("✅ Gemini generator initialized successfully!")
        
        # Example context and query
        context = """
        The RAG (Retrieval-Augmented Generation) system is an AI architecture that combines 
        information retrieval with text generation. It works by first retrieving relevant 
        documents or information from a knowledge base, then using that retrieved context 
        to generate accurate and informative responses to user queries.
        
        Key benefits of RAG systems include:
        1. Up-to-date information through real-time retrieval
        2. Reduced hallucination by grounding responses in retrieved facts
        3. Ability to cite sources and provide evidence
        4. Scalability to large knowledge bases
        """
        
        query = "What are the main benefits of RAG systems and how do they work?"
        
        print(f"\n📝 Query: {query}")
        print(f"📚 Context length: {len(context)} characters")
        
        # Generate response
        print("\n🤖 Generating response with Gemini...")
        result = generator.generate(query, context)
        
        # Display results
        print("\n" + "="*50)
        print("🎯 GENERATION RESULT")
        print("="*50)
        print(f"📝 Query: {result.query}")
        print(f"🤖 Response: {result.response}")
        print(f"📊 Response length: {len(result.response)} characters")
        print(f"⏱️  Generation time: {result.metadata.get('generation_time', 'N/A'):.3f}s")
        print(f"🔧 Model: {result.metadata.get('model', 'N/A')}")
        print(f"🌡️  Temperature: {result.metadata.get('temperature', 'N/A')}")
        print(f"🔢 Max tokens: {result.metadata.get('max_tokens', 'N/A')}")
        
        # Display usage information if available
        usage = result.metadata.get('usage', {})
        if usage:
            print(f"📊 Token usage:")
            print(f"   - Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   - Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"   - Total tokens: {usage.get('total_tokens', 'N/A')}")
        
        print("="*50)
        
        # Test with different parameters
        print("\n🧪 Testing with different parameters...")
        result2 = generator.generate(
            "Explain RAG systems in simple terms",
            context,
            temperature=0.8,
            max_tokens=500
        )
        
        print(f"\n📝 Simple explanation: {result2.response}")
        print(f"🌡️  Used temperature: {result2.metadata.get('temperature')}")
        print(f"🔢 Used max tokens: {result2.metadata.get('max_tokens')}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure GOOGLE_API_KEY is set correctly")
        print("2. Check your internet connection")
        print("3. Verify your Google API key has access to Gemini")
        print("4. Ensure you have the required packages installed")


if __name__ == "__main__":
    main()
