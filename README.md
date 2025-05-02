# Retrieval-Augmented Generation (RAG) System from Scratch

A fully modular, production-ready Retrieval-Augmented Generation (RAG) system built from the ground up. This project demonstrates end-to-end capabilities from document parsing, chunking, vector search, context-aware generation, evaluation, and API deployment.

---

## Features

- Multi-format document ingestion (PDF, DOCX, HTML)
- Modular chunking (fixed, semantic, recursive)
- Pluggable embedding generators (OpenAI, SentenceTransformers)
- Vector DB support (ChromaDB, Qdrant)
- Retrieval strategies (semantic, hybrid, reranking)
- Prompt-based LLM generation (OpenAI or local LLaMA)
- Evaluation framework (accuracy, latency, hallucination)
- FastAPI backend + Gradio frontend for demo

---

## System Architecture

```
Document → Chunking → Embedding → Vector Store
                       ↓             ↑
User Query → Embed → Retrieve → Build Context → Generate Answer
```

---

## Directory Structure

```
├── app
│   ├── api             # FastAPI backend
│   ├── frontend        # Gradio UI
│   └── config          # YAML-based configuration
├── data                # Raw, processed data, embeddings, evaluations
├── src
│   ├── document_processing
│   ├── chunking
│   ├── embedding
│   ├── vector_store
│   ├── retrieval
│   ├── generation
│   └── evaluation
├── tests               # Unit tests
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quickstart

```bash
# Clone repository
$ git clone https://github.com/yourusername/rag-from-scratch.git
$ cd rag-from-scratch

# Install dependencies
$ pip install -r requirements.txt

# Start the backend
$ uvicorn app.api.main:app --reload

# (Optional) Run Gradio UI
$ python app/frontend/demo_ui.py
```

---

## Docker Deployment

```bash
# Build and run services
$ docker-compose up --build

# Upload a document (e.g., via Postman or Swagger)
POST http://localhost:8000/upload

# Ask a question
POST http://localhost:8000/query
```

---

## Evaluation Sample

```json
{
  "accuracy": 0.85,
  "average_latency": 2.1,
  "average_hallucination": 0.14
}
```

---

## Example Use Cases

- Custom document-based QA assistants
- Enterprise knowledge base chatbots
- Legal or medical document analysis

---

## License

MIT License

---

## Acknowledgments

Inspired by open-source RAG frameworks, including LangChain, Haystack, and llama-cpp. Special thanks to the community for powerful tooling.
