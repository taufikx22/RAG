@echo off
REM RAG System Deployment Script for Windows
REM This script helps set up and deploy the RAG system

echo 🚀 RAG System Deployment Script
echo ================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are available

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file from template...
    copy env.example .env
    echo ⚠️  Please edit .env file with your API keys and configuration
    echo    Key variables to set:
    echo    - OPENAI_API_KEY (if using OpenAI)
    echo    - GOOGLE_API_KEY (if using Gemini)
    echo    - VECTOR_STORE_DEFAULT (chroma or qdrant)
    echo.
    pause
) else (
    echo ✅ .env file already exists
)

REM Create data directories
echo 📁 Creating data directories...
if not exist data\chroma_db mkdir data\chroma_db
if not exist data\qdrant mkdir data\qdrant
if not exist data\redis mkdir data\redis
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist data\embeddings mkdir data\embeddings
if not exist data\evaluations mkdir data\evaluations

echo ✅ Data directories created

REM Build and start services
echo 🐳 Building and starting Docker services...
docker-compose build
docker-compose up -d

echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...
docker-compose ps

REM Test API health
echo 🏥 Testing API health...
curl -f http://localhost:8000/healthz >nul 2>&1
if errorlevel 1 (
    echo ❌ API health check failed
    echo 📋 Checking logs...
    docker-compose logs rag-app
    pause
    exit /b 1
) else (
    echo ✅ API is healthy
    echo.
    echo 🎉 Deployment successful!
    echo.
    echo 📚 Access your RAG system:
    echo    - API Documentation: http://localhost:8000/docs
    echo    - Health Check: http://localhost:8000/healthz
    echo    - Qdrant Dashboard: http://localhost:6333
    echo.
    echo 📖 Next steps:
    echo    1. Upload documents via /ingest endpoint
    echo    2. Query the system via /query endpoint
    echo    3. Monitor logs: docker-compose logs -f rag-app
    echo.
)

pause
