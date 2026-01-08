# RAG-Powered Q&A System

A local Retrieval-Augmented Generation (RAG) system for question answering using documents. This system allows you to upload documents and ask questions about their content using a local LLM (Llama 3).

## 🚀 Features

- **Local Processing**: All processing happens locally - no external APIs required
- **Document Ingestion**: Upload and process multiple document formats (TXT, PDF, Markdown, etc.)
- **Semantic Search**: Find relevant information using vector similarity
- **Structured Responses**: Get answers in a structured JSON format
- **REST API**: Full-featured API for integration with other applications
- **Customizable**: Easy to modify and extend for different use cases

## 🛠️ Prerequisites

- Python 3.8+
- Ollama (for running local LLMs)
- pip (Python package manager)

## 🚀 Quick Start

1. **Clone and set up the project**:
   ```bash
   git clone <repository-url>
   cd rag-system
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download and run the Llama 3 model**:
   ```bash
   ollama pull llama3
   ```

3. **Start the RAG system**:
   ```bash
   python app.py
   ```

4. **Access the API**:
   - Interactive API docs: http://localhost:8000/docs
   - API base URL: http://localhost:8000

## 📁 Project Structure

```
rag-system/
├── config/                  # Configuration settings
│   └── settings.py          # Application configuration
├── data/                    # Data storage
│   └── documents/           # Uploaded documents
├── src/
│   ├── embedding/           # Text embedding models
│   │   └── nomic_embedder.py
│   ├── llm/                 # Language model clients
│   │   └── llama3_client.py
│   ├── utils/               # Utility functions
│   │   └── document_processor.py
│   ├── vectorstore/         # Vector database implementation
│   │   └── faiss_store.py
│   └── rag_system.py        # Main RAG system implementation
├── app.py                   # FastAPI application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔍 How It Works

The RAG system works in three main steps:

1. **Document Processing**:
   - Documents are loaded and split into chunks
   - Each chunk is converted to a vector embedding using Nomic's embedding model
   - Vectors are stored in a FAISS index for efficient similarity search

2. **Query Processing**:
   - User question is converted to a vector using the same embedding model
   - System finds the most relevant document chunks using vector similarity

3. **Response Generation**:
   - Relevant chunks are combined with the question in a prompt
   - Local LLM (Llama 3) generates an answer based on the context
   - Response is formatted and returned to the user

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Get API Information
- **URL**: `GET /`
- **Description**: Get basic information about the API
- **Response**:
  ```json
  {
    "name": "RAG-Powered Q&A System",
    "version": "1.0.0",
    "status": "running",
    "model": "llama3",
    "documents_dir": "/path/to/documents"
  }
  ```

#### 2. Upload Document
- **URL**: `POST /upload`
- **Description**: Upload a document for processing
- **Request**: `multipart/form-data` with file
- **Response**:
  ```json
  {
    "status": "success",
    "message": "File uploaded successfully",
    "filename": "document.pdf"
  }
  ```

#### 3. List Documents
- **URL**: `GET /documents`
- **Description**: List all uploaded documents
- **Response**:
  ```json
  {
    "status": "success",
    "documents": [
      {
        "name": "document.pdf",
        "size": 1024000,
        "modified": 1640995200
      }
    ],
    "count": 1
  }
  ```

#### 4. Ingest Documents
- **URL**: `POST /ingest`
- **Description**: Process all uploaded documents into the vector store
- **Response**:
  ```json
  {
    "status": "success",
    "chunks_processed": 42,
    "total_vectors": 42
  }
  ```

#### 5. Query (Simple)
- **URL**: `GET /query?question=your+question&top_k=3`
- **Description**: Ask a question and get an answer
- **Parameters**:
  - `question` (required): The question to ask
  - `top_k`: Number of relevant chunks to retrieve (default: 3)
- **Response**:
  ```json
  {
    "question": "What is the company policy on remote work?",
    "answer": "The company allows employees to work remotely for up to 3 days per week...",
    "sources": [
      {
        "document": "employee_handbook.pdf",
        "score": 0.87,
        "text": "Remote Work Policy: Employees may work remotely for up to 3 days..."
      }
    ],
    "timestamp": "2026-01-08T03:08:00.000000"
  }
  ```

#### 6. Query (Structured)
- **URL**: `POST /query_structured`
- **Description**: Get a structured response in a specific format
- **Request**:
  - `question`: The question to ask
  - `response_format`: JSON schema for the response format
  - `top_k`: Number of relevant chunks to retrieve (default: 3)
- **Example Request**:
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/query_structured' \
    -H 'Content-Type: application/x-www-form-urlencoded' \
    -d 'question=What+is+the+company+policy+on+remote+work%3F&response_format={"type":"object","properties":{"answer":{"type":"string"},"policy_name":{"type":"string"},"applies_to":{"type":"array","items":{"type":"string"}}},"required":["answer"]}'
  ```
- **Response**:
  ```json
  {
    "question": "What is the company policy on remote work?",
    "answer": {
      "answer": "Employees may work remotely for up to 3 days per week...",
      "policy_name": "Flexible Work Arrangement Policy",
      "applies_to": ["All full-time employees"],
      "requires_approval": true
    },
    "sources": [...],
    "timestamp": "2026-01-08T03:10:00.000000"
  }
  ```

#### 7. Clear Index
- **URL**: `POST /clear`
- **Description**: Clear the vector store index
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Vector store index cleared"
  }
  ```

## ⚙️ Configuration

Edit the `.env` file to configure the system:

```env
# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Model settings
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
LLM_MODEL=llama3

# Storage settings
VECTOR_STORE_PATH=./data/vector_store.faiss
DOCUMENTS_DIR=./data/documents

# Chunking settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RESULTS=3
```

## 🔄 Workflow

1. **Setup**:
   - Install dependencies and set up the environment
   - Start the Ollama server with `ollama serve`

2. **Add Documents**:
   - Upload documents using the `/upload` endpoint
   - Process them with the `/ingest` endpoint

3. **Query**:
   - Use the `/query` or `/query_structured` endpoints to ask questions
   - Get answers based on the document content

4. **Maintenance**:
   - Use `/documents` to manage uploaded files
   - Clear the index with `/clear` when needed

## 🛠️ Development

### Adding New Features

1. **Add a new document type**:
   - Modify `document_processor.py` to handle the new format
   - Add any required dependencies to `requirements.txt`

2. **Change the embedding model**:
   - Update `EMBEDDING_MODEL` in `.env`
   - Modify `nomic_embedder.py` if using a different embedding service

3. **Use a different LLM**:
   - Create a new client in the `llm/` directory
   - Update `LLM_MODEL` in `.env`

### Testing

```bash
# Run the test suite
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for providing easy access to local LLMs
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Nomic](https://nomic.ai/) for the embedding model

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
