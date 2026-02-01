# Summerizer 1.1 - Factory Performance Analytics & Knowledge Generation System

A Python-based intelligent system for analyzing factory production logs, generating performance insights, and creating AI-driven reports using embeddings and semantic similarity.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Design Patterns](#design-patterns)
- [Testing](#testing)
- [Contributing](#contributing)

## Overview

Summerizer is an advanced factory analytics platform that:

1. **Ingests production logs** from multiple factories containing employee and machine event data
2. **Generates intelligent summaries** by analyzing patterns and creating semantic representations
3. **Performs semantic search** using vector embeddings to find similar production events
4. **Generates AI reports** based on semantic relevance and LLM-powered text generation

The system uses a modular, plugin-based architecture with clear separation of concerns through abstraction layers and polymorphism, allowing easy extension and testing.

### Use Cases

- **Production Analysis**: Identify patterns in machine breakdowns and employee productivity
- **Performance Reporting**: Generate comparative reports showing employee/machine performance against factory averages
- **Issue Investigation**: Use semantic search to find similar historical issues and their resolutions
- **Trend Analysis**: Understand how different metrics (breakdowns, output quality, downtime) correlate across the factory

## Key Features

### 1. **Knowledge Generation Pipeline**

- Processes factory logs (production events)
- Extracts employee and machine performance metrics
- Generates contextual statements about performance
- Creates embeddings (768-dimensional vectors) for semantic search
- Stores knowledge base for future querying

### 2. **Intelligent Report Generation**

- Accepts natural language queries (e.g., "which employee has the most breakdowns?")
- Uses semantic similarity to find relevant production records
- Leverages LLM (Large Language Model) to generate structured audit reports
- Returns ranked results based on relevance

### 3. **Vector Embeddings**

- Uses SentenceTransformer for text encoding (768-dimensional embeddings)
- Implements cosine similarity for semantic matching
- Handles query-specific prompting for optimized search
- Stores embeddings in vector database

### 4. **Multi-Factory Support**

- Processes multiple factories concurrently
- Maintains separate knowledge bases per factory
- Compares individual performance against factory averages

## Project Structure

```
summerizer_1.1/
├── app.py                          # Flask application entry point
├── interfaces/                     # Abstract interfaces (polymorphic contracts)
│   ├── Storage.py                 # Storage interface - abstract base class
│   └── EmbeddingModel.py          # EmbeddingModel interface - abstract base class
├── model/                         # Concrete implementations
│   └── Qwen3.py                   # Qwen3 embedding & LLM model implementation
├── storage/                       # Data persistence implementations
│   └── SupabaseStorage.py         # Supabase database implementation
├── services/                      # Business logic layer
│   ├── knowledge_generator.py     # Generates knowledge base from logs
│   ├── report_generator.py        # Generates AI-powered reports
│   └── data_encoder.py            # Encodes data into embeddings
├── utils/                         # Utility functions
│   ├── helpers.py                 # Common helper functions
│   ├── employee.py                # Employee analytics
│   └── machine.py                 # Machine analytics
├── integration/                   # External service integrations
│   └── supabase_client.py         # Supabase client configuration
├── tests/                         # Comprehensive test suite (48 tests)
│   ├── test_helper.py             # 12 unit tests
│   ├── test_employee.py           # 12 unit tests
│   ├── test_knowledge_generator.py # 5 unit tests
│   ├── test_report_generator.py    # 8 unit tests
│   ├── test_data_encoder.py       # 3 unit tests
│   └── test_scenarios.py          # 10 integration tests with mocks
└── TESTING.md                     # Testing documentation
```

## Architecture

### High-Level Data Flow

```
Production Logs (DB)
    ↓
[KnowledgeGenerator]
    ├─→ Extract metrics (breakdowns, pauses, output quality)
    ├─→ Generate statements (narratives describing performance)
    ├─→ [DataEncoder] encodes statements into 768-dim vectors
    ├─→ [Storage] stores embeddings + statements
    └─→ Knowledge Base (vector DB)
         ↓
[ReportGenerator]
    ├─→ Accepts natural language query
    ├─→ Encodes query into 768-dim vector
    ├─→ Calculates cosine similarity against knowledge base
    ├─→ Ranks top-k most relevant records
    ├─→ Uses LLM to generate structured reports
    └─→ Returns formatted report
```

### Key Architectural Principles

#### 1. **Abstraction Layers** (via Abstract Base Classes)

The system uses two main abstract interfaces to define contracts:

**Storage Interface** (`interfaces/Storage.py`):

```python
class Storage(ABC):
    @abstractmethod
    def get_logs(self, factory_id: str)

    @abstractmethod
    def push_embedding(self, vector: list, factory_id: str, type: str, statement: str, code: str) -> bool

    @abstractmethod
    def get_knowledge(self, factory_id: str, type: str)
    # ... more methods
```

**Benefits**:

- Services don't depend on specific database implementations
- Easy to swap implementations (Supabase → PostgreSQL → MongoDB)
- Testable with mock implementations
- Clear contract for what storage layer must provide

**EmbeddingModel Interface** (`interfaces/EmbeddingModel.py`):

```python
class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, documents)

    @abstractmethod
    def similarity(self, query_vec, doc_vecs)

    @abstractmethod
    def getResponse(self, messages)
```

**Benefits**:

- Services don't care about which embedding model is used
- Can upgrade models without changing service code
- Support multiple embedding strategies simultaneously

#### 2. **Polymorphism** (via Concrete Implementations)

**Qwen3 Model** (`model/Qwen3.py`):

- Implements both `EmbeddingModel` interface
- Provides text encoding (via SentenceTransformer)
- Computes semantic similarity (via torch)
- Generates responses (via Qwen3 LLM)
- Can be replaced with other models (GPT, BERT, etc.) as long as they implement the interface

**SupabaseStorage** (`storage/SupabaseStorage.py`):

- Implements `Storage` interface
- Connects to Supabase PostgreSQL database
- Handles all persistence operations
- Can be replaced with other storage backends without changing services

#### 3. **Dependency Injection**

Services receive dependencies through constructors:

```python
class KnowledgeGenerator:
    def __init__(self, storage: Storage, embed_model: EmbeddingModel, encoder: DataEncoder = None):
        self.storage = storage
        self.embed_model = embed_model
        self.encoder = encoder or DataEncoder(storage, embed_model)
```

**Benefits**:

- Loose coupling between components
- Easy to inject mock implementations for testing
- Configuration centralized in `app.py`

### Component Responsibilities

| Component                  | Responsibility                                            | Type      |
| -------------------------- | --------------------------------------------------------- | --------- |
| **KnowledgeGenerator**     | Orchestrates pipeline to create knowledge base from logs  | Service   |
| **ReportGenerator**        | Handles semantic search and LLM-powered report generation | Service   |
| **DataEncoder**            | Converts text statements to vector embeddings             | Service   |
| **Qwen3**                  | Provides embedding and LLM capabilities                   | Model     |
| **SupabaseStorage**        | Manages all database operations                           | Storage   |
| **Employee/Machine Utils** | Extracts metrics and generates performance narratives     | Utilities |

## Installation

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- Virtual environment (recommended)
- Supabase account (for database)

### Steps

1. **Clone or download the project**:

```bash
cd summerizer_1.1
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install flask flask-cors transformers sentence-transformers torch numpy supabase
```

4. **Set up environment variables** (create `.env` file):

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_api_key
```

5. **Verify installation**:

```bash
python -c "import flask, transformers, sentence_transformers; print('All dependencies installed!')"
```

## Configuration

### Supabase Setup

The system requires these tables in Supabase:

**production_logs**:

```
- factory_id (UUID)
- event_type (text): 'breakdown', 'pause', 'good_piece', 'bad_piece'
- event_duration (integer): minutes
- employee_id (UUID, foreign key)
- machine_id (UUID, foreign key)
- created_at (timestamp)
```

**factory_knowledge_base**:

```
- id (UUID, primary key)
- factory_id (UUID)
- type (text): 'employee' or 'machine'
- emp_code (text, nullable)
- machine_code (text, nullable)
- statement (text): Generated narrative
- embedding (float array, 768 dimensions)
- created_at (timestamp)
```

**factory**, **employee**, **machine** tables for reference data.

### Model Configuration

Edit `model/Qwen3.py` to customize:

```python
model_id = "Qwen/Qwen3-Embedding-0.6B"      # LLM model
embed_model_id = "sentence-transformers/all-mpnet-base-v2"  # Embedding model
device = "cpu"                               # "cuda" for GPU
```

## Quick Start

### 1. Generate Knowledge Base

```python
from model.Qwen3 import Qwen3
from storage.SupabaseStorage import SupabaseStorage
from services.knowledge_generator import KnowledgeGenerator

model = Qwen3()
storage = SupabaseStorage()
generator = KnowledgeGenerator(storage, model)

# Process all factories and create embeddings
success = generator.generate_knowledge()
```

### 2. Generate a Report

```python
from services.report_generator import ReportGenerator

report_gen = ReportGenerator(storage, model)

# Query the knowledge base
report = report_gen.generate_report(
    type="employee",
    factory_id="97e90fd2-469a-471b-a824-1e6ac0d5ec93",
    query="which employee has the most breakdowns?",
    top_k=5
)
print(report)
```

### 3. Run Flask API

```bash
python app.py
```

**Endpoints**:

- `POST /summarize` - Generates knowledge base
- `POST /report` - Generates report (see app.py for parameters)

## API Endpoints

### POST /summarize

Generates knowledge base from factory logs.

**Request**:

```json
{}
```

**Response**:

```json
{
  "message": "Summarization process completed successfully."
}
```

**What it does**:

1. Fetches all factories
2. For each factory, retrieves production logs
3. Generates performance statements for each employee-machine pair
4. Creates embeddings for each statement
5. Stores in knowledge base

### POST /report

Generates AI-powered report based on semantic search.

**Request Parameters** (modify in app.py):

- `type`: "employee" or "machine"
- `factory_id`: UUID of factory to analyze
- `query`: Natural language query (e.g., "which employee has the most breakdowns?")
- `top_k`: Number of top results to include (default: 5)

**Response**:

```json
{
  "message": "Report generated successfully.",
  "report": "structured audit report text..."
}
```

## Design Patterns

### 1. **Abstract Factory Pattern** (Interfaces)

The `Storage` and `EmbeddingModel` interfaces act as abstract factories defining contracts that concrete implementations must fulfill. This allows:

```python
# Services don't care about concrete types
def __init__(self, storage: Storage, embed_model: EmbeddingModel):
    # Works with ANY implementation
```

### 2. **Dependency Injection Pattern**

Dependencies are provided via constructors, enabling:

- Loose coupling
- Easy testing with mocks
- Configuration flexibility

### 3. **Strategy Pattern** (Vector Operations)

Different similarity algorithms can be used interchangeably:

```python
sim = embed_model.similarity(vectors, query)  # Implementation doesn't matter
```

### 4. **Decorator Pattern** (DataEncoder)

`DataEncoder` wraps the embedding process:

```python
class DataEncoder:
    def __init__(self, storage: Storage, embed_model: EmbeddingModel):
        # Decorates/wraps the encoding and storage

    def encode_data(self, documents, type_label, factory_id, code):
        # Encodes and stores in one operation
```

### 5. **Pipeline Pattern** (KnowledgeGenerator)

Orchestrates a multi-step workflow:

```
Logs → Metrics → Statements → Encoding → Storage
```

## Testing

The project includes **48 comprehensive tests** covering unit tests and integration scenarios.

### Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -p "test_*.py"

# Run specific test file
python -m unittest tests.test_scenarios -v

# Run with coverage
coverage run -m unittest discover -s tests && coverage report
```

### Test Structure

| File                        | Tests | Focus                            |
| --------------------------- | ----- | -------------------------------- |
| test_helper.py              | 12    | Utility functions                |
| test_employee.py            | 12    | Employee analytics               |
| test_knowledge_generator.py | 5     | Knowledge generation workflow    |
| test_report_generator.py    | 8     | Report generation and similarity |
| test_data_encoder.py        | 3     | Vector encoding                  |
| test_scenarios.py           | 10    | End-to-end integration scenarios |

### Mock Implementations

Tests use realistic mock implementations:

- **MockStorage**: Full in-memory implementation of Storage interface
- **MockEmbeddingModel**: Deterministic embeddings and similarity calculations

See [TESTING.md](TESTING.md) for detailed testing guide.

## Contributing

### Adding New Features

1. **If adding new storage backend**:
   - Inherit from `Storage` interface
   - Implement all abstract methods
   - Update `app.py` to instantiate new backend
   - Add tests using mock implementations

2. **If adding new embedding model**:
   - Inherit from `EmbeddingModel` interface
   - Implement `encode()`, `similarity()`, `getResponse()`
   - Update `app.py` to use new model
   - Verify with existing tests

3. **If modifying services**:
   - Update corresponding test file
   - Ensure all 48 tests still pass
   - Document changes in docstrings

### Code Style

- Follow PEP 8
- Use type hints
- Include docstrings for functions
- Keep methods focused and single-responsibility

## Troubleshooting

### Common Issues

**"No module named 'interfaces'"**:

- Ensure you're running from project root
- Check Python path includes project directory

**"No valid vectors found"**:

- Knowledge base may be empty
- Run `/summarize` endpoint first
- Check Supabase has production logs

**Embedding dimension mismatch**:

- Ensure all encode() calls return 768-dimensional vectors
- Check SentenceTransformer model configuration

**"DATABASE ERROR"**:

- Verify Supabase credentials in `.env`
- Check database tables exist and have correct schema
- Verify query parameters (factory_id) are valid UUIDs

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) - Deep dive into design patterns and architecture
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API and class reference
- [DEVELOPMENT.md](DEVELOPMENT.md) - Developer guide for extending the system
- [TESTING.md](TESTING.md) - Comprehensive testing documentation

## License

[Add your license here]

## Support

For issues or questions, please refer to the documentation files or create an issue in the repository.
