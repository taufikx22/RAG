#!/bin/bash

# RAG System Deployment Script
# This script helps set up and deploy the RAG system

set -e

echo "🚀 RAG System Deployment Script"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "⚠️  Please edit .env file with your API keys and configuration"
    echo "   Key variables to set:"
    echo "   - OPENAI_API_KEY (if using OpenAI)"
    echo "   - GOOGLE_API_KEY (if using Gemini)"
    echo "   - VECTOR_STORE_DEFAULT (chroma or qdrant)"
    echo ""
    read -p "Press Enter after editing .env file..."
else
    echo "✅ .env file already exists"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/chroma_db
mkdir -p data/qdrant
mkdir -p data/redis
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/embeddings
mkdir -p data/evaluations

echo "✅ Data directories created"

# Build and start services
echo "🐳 Building and starting Docker services..."
docker-compose build
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Test API health
echo "🏥 Testing API health..."
if curl -f http://localhost:8000/healthz > /dev/null 2>&1; then
    echo "✅ API is healthy"
    echo ""
    echo "🎉 Deployment successful!"
    echo ""
    echo "📚 Access your RAG system:"
    echo "   - API Documentation: http://localhost:8000/docs"
    echo "   - Health Check: http://localhost:8000/healthz"
    echo "   - Qdrant Dashboard: http://localhost:6333"
    echo ""
    echo "📖 Next steps:"
    echo "   1. Upload documents via /ingest endpoint"
    echo "   2. Query the system via /query endpoint"
    echo "   3. Monitor logs: docker-compose logs -f rag-app"
    echo ""
else
    echo "❌ API health check failed"
    echo "📋 Checking logs..."
    docker-compose logs rag-app
    exit 1
fi
