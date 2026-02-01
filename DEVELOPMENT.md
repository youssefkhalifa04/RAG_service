# Developer Guide - Summerizer 1.1

A comprehensive guide for developers who want to extend, modify, and contribute to the Summerizer project.

## Table of Contents
1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Working with Abstractions](#working-with-abstractions)
4. [Adding Features](#adding-features)
5. [Testing Your Changes](#testing-your-changes)
6. [Code Style Guide](#code-style-guide)
7. [Common Development Tasks](#common-development-tasks)
8. [Troubleshooting](#troubleshooting)
9. [Contributing Best Practices](#contributing-best-practices)

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- Virtual environment tool (venv or conda)
- Text editor or IDE (VS Code recommended)

### Initial Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd summerizer_1.1
```

2. **Create virtual environment**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Or using conda
conda create -n summerizer python=3.10
conda activate summerizer
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install development dependencies**:
```bash
pip install pytest pytest-cov black flake8 mypy
```

5. **Verify installation**:
```bash
python -m unittest discover -s tests -p "test_*.py"
# Should show: Ran 48 tests ... OK
```

### Setting up IDE

**VS Code**:
1. Install Python extension
2. Select interpreter: `Ctrl+Shift+P` → Python: Select Interpreter
3. Choose venv environment
4. Install Pylance for IntelliSense

---

## Project Structure

### Directory Layout

```
summerizer_1.1/
│
├── interfaces/                    # Abstract interfaces
│   ├── Storage.py                # Storage contract
│   └── EmbeddingModel.py         # EmbeddingModel contract
│
├── model/                         # ML model implementations
│   └── Qwen3.py                  # Qwen3 LLM + embeddings
│
├── storage/                       # Data persistence
│   └── SupabaseStorage.py        # Supabase implementation
│
├── services/                      # Business logic
│   ├── knowledge_generator.py
│   ├── report_generator.py
│   └── data_encoder.py
│
├── utils/                         # Utility functions
│   ├── helpers.py
│   ├── employee.py
│   └── machine.py
│
├── integration/                   # External services
│   └── supabase_client.py
│
├── tests/                         # Test suite (48 tests)
│   ├── test_helper.py
│   ├── test_employee.py
│   ├── test_knowledge_generator.py
│   ├── test_report_generator.py
│   ├── test_data_encoder.py
│   └── test_scenarios.py
│
├── app.py                         # Flask entry point
├── README.md                      # Main documentation
├── ARCHITECTURE.md                # Architecture overview
├── API_REFERENCE.md              # API documentation
├── DEVELOPMENT.md                 # This file
└── requirements.txt               # Python dependencies
```

### Adding New Modules

When adding a new module, place it in the appropriate directory:

- **Core logic**: `services/`
- **Database operations**: `storage/`
- **ML/LLM operations**: `model/`
- **Reusable utilities**: `utils/`
- **External integrations**: `integration/`
- **Contracts/interfaces**: `interfaces/`

---

## Working with Abstractions

### Understanding Abstraction

The project uses **Abstract Base Classes (ABC)** to define interfaces:

```python
from abc import ABC, abstractmethod

class MyInterface(ABC):
    @abstractmethod
    def required_method(self, param: str) -> str:
        """All implementations MUST provide this method"""
        pass
```

### Creating a New Interface

1. **Create abstract class**:

```python
# interfaces/CacheBackend.py
from abc import ABC, abstractmethod

class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str):
        """Retrieve value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value, ttl: int = 3600) -> bool:
        """Store value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove value from cache"""
        pass
```

2. **Document the interface**:

```python
# At the top of the file
"""
Cache Backend Interface

Defines contract for caching implementations.
Implementations should handle:
- Key-value storage
- TTL (time-to-live)
- Cache eviction
"""
```

3. **Create concrete implementation**:

```python
# storage/RedisCache.py
from interfaces.CacheBackend import CacheBackend
import redis

class RedisCache(CacheBackend):
    def __init__(self, host='localhost', port=6379):
        self.client = redis.Redis(host=host, port=port)
    
    def get(self, key: str):
        value = self.client.get(key)
        return value.decode() if value else None
    
    def set(self, key: str, value, ttl: int = 3600) -> bool:
        self.client.setex(key, ttl, value)
        return True
    
    def delete(self, key: str) -> bool:
        self.client.delete(key)
        return True
```

### Implementing an Existing Interface

**Example**: Adding MongoDB storage

```python
# storage/MongoStorage.py
from interfaces.Storage import Storage
from pymongo import MongoClient

class MongoStorage(Storage):
    def __init__(self, connection_string: str):
        self.client = MongoClient(connection_string)
        self.db = self.client.factory_db
    
    def get_logs(self, factory_id: str):
        """Implement Storage.get_logs()"""
        return list(self.db.logs.find({"factory_id": factory_id}))
    
    def push_embedding(self, vector, factory_id, type, statement, code) -> bool:
        """Implement Storage.push_embedding()"""
        try:
            self.db.knowledge_base.insert_one({
                "factory_id": factory_id,
                "type": type,
                "embedding": vector,
                "statement": statement,
                "code": code
            })
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    # ... implement remaining abstract methods
```

### Key Rules for Abstractions

1. **Implement all abstract methods**: Python will raise `TypeError` if you forget
2. **Match method signatures**: Parameter types and return types must match
3. **Update services to accept interface**: Use type hints: `storage: Storage`
4. **Test with mock implementations**: Create mock for testing before real implementation

---

## Adding Features

### Feature 1: Add a Cache Layer

**Goal**: Cache frequently accessed knowledge base queries.

**Step 1**: Define cache interface
```python
# interfaces/CacheBackend.py
class CacheBackend(ABC):
    @abstractmethod
    def get(self, key: str): pass
    
    @abstractmethod
    def set(self, key: str, value, ttl: int): pass
```

**Step 2**: Implement in-memory cache
```python
# storage/MemoryCache.py
class MemoryCache(CacheBackend):
    def __init__(self):
        self.cache = {}
    
    def get(self, key: str):
        return self.cache.get(key)
    
    def set(self, key: str, value, ttl: int = 3600) -> bool:
        self.cache[key] = value
        return True
```

**Step 3**: Integrate into ReportGenerator
```python
# services/report_generator.py
class ReportGenerator:
    def __init__(self, storage: Storage, embed_model: EmbeddingModel, cache: CacheBackend = None):
        self.storage = storage
        self.embed_model = embed_model
        self.cache = cache
    
    def generate_report(self, type, factory_id, query, top_k=5):
        # Check cache first
        cache_key = f"{factory_id}:{type}:{query}"
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        
        # Generate normally
        records = self.storage.get_knowledge(factory_id, type)
        # ... rest of generation ...
        
        # Store in cache
        if self.cache:
            self.cache.set(cache_key, final_report)
        
        return final_report
```

**Step 4**: Update app.py
```python
from storage.MemoryCache import MemoryCache

cache = MemoryCache()
# Works with any Storage and EmbeddingModel implementation
report_generator = ReportGenerator(storage, embed_model, cache)
```

**Step 5**: Add tests
```python
# tests/test_cache.py
import unittest
from storage.MemoryCache import MemoryCache

class TestMemoryCache(unittest.TestCase):
    def test_get_set(self):
        cache = MemoryCache()
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")
    
    def test_get_nonexistent(self):
        cache = MemoryCache()
        self.assertIsNone(cache.get("missing"))
```

### Feature 2: Add Custom Metrics

**Goal**: Track custom metrics alongside standard ones.

**Step 1**: Extend the Employee class with new metrics

```python
# utils/employee.py
def get_single_employee_deeds(employee_code: str, logs: list):
    employee_logs = [log for log in logs if log['employee']['code'] == employee_code]
    
    deeds = {
        # ... existing metrics ...
        "breakdown_to_output_ratio": calculate_breakdown_ratio(employee_logs),
        "error_rate": calculate_error_rate(employee_logs),
        "efficiency_score": calculate_efficiency(employee_logs)
    }
    return deeds

def calculate_breakdown_ratio(logs):
    """Custom metric: breakdowns per piece produced"""
    breakdowns = len([l for l in logs if l['event_type'] == 'breakdown'])
    pieces = len([l for l in logs if l['event_type'] in ['good_piece', 'bad_piece']])
    return breakdowns / pieces if pieces > 0 else 0

def calculate_error_rate(logs):
    """Custom metric: percentage of bad pieces"""
    bad = len([l for l in logs if l['event_type'] == 'bad_piece'])
    total = len([l for l in logs if l['event_type'] in ['good_piece', 'bad_piece']])
    return (bad / total * 100) if total > 0 else 0

def calculate_efficiency(logs):
    """Custom metric: weighted efficiency score"""
    # Your custom algorithm here
    pass
```

**Step 2**: Update statement generation to use new metrics

```python
def generate_employee_statement(employee_code, machine_code, deeds, overall, avg_comp):
    # Include custom metrics
    stmt = (
        f"Employee {employee_code}: "
        f"... existing data ... "
        f"Efficiency score: {deeds['efficiency_score']:.2f}. "
        f"Error rate: {deeds['error_rate']:.2f}%. "
    )
    return stmt
```

**Step 3**: Add tests

```python
# tests/test_employee_metrics.py
def test_breakdown_ratio(self):
    logs = [
        {'employee': {'code': 'E1'}, 'event_type': 'breakdown'},
        {'employee': {'code': 'E1'}, 'event_type': 'good_piece'},
        # ...
    ]
    ratio = calculate_breakdown_ratio([l for l in logs if l['employee']['code'] == 'E1'])
    self.assertEqual(ratio, 1.0)  # 1 breakdown, 1 piece = 1.0
```

---

## Testing Your Changes

### Running Tests

**Note**: Tests use mock implementations and don't require specific storage or model backends. You can run tests regardless of which storage/model you're using in production.

```bash
# Run all tests (works with any storage/model choice)
python -m unittest discover -s tests -p "test_*.py"

# Run specific test file
python -m unittest tests.test_employee -v

# Run specific test
python -m unittest tests.test_employee.TestEmployee.test_get_comparison -v

# Run with coverage
coverage run -m unittest discover -s tests
coverage report
coverage html  # Generate HTML report
```

### Writing Tests

**Template for new test file**:

```python
# tests/test_new_feature.py
import unittest
from unittest.mock import Mock, patch
from interfaces.Storage import Storage
from services.new_service import NewService

class MockStorage(Storage):
    """Mock storage for testing"""
    def __init__(self):
        self.data = {}
    
    def get_logs(self, factory_id):
        return self.data.get(factory_id, [])
    
    # Implement other abstract methods...

class TestNewService(unittest.TestCase):
    def setUp(self):
        """Called before each test"""
        self.storage = MockStorage()
        self.service = NewService(self.storage)
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = self.service.do_something()
        self.assertIsNotNone(result)
    
    def test_error_handling(self):
        """Test error handling"""
        with self.assertRaises(ValueError):
            self.service.do_something_with_error()
    
    def test_with_mock_data(self):
        """Test with mock data"""
        self.storage.data["factory1"] = [
            {"event_type": "breakdown", "employee": {"code": "E1"}}
        ]
        result = self.service.process()
        self.assertEqual(len(result), 1)

if __name__ == '__main__':
    unittest.main()
```

### Test Organization

```
tests/
├── test_interfaces.py       # Interface tests
├── test_services/
│   ├── test_knowledge_gen.py
│   ├── test_report_gen.py
│   └── test_encoder.py
├── test_storage/
│   └── test_supabase.py
├── test_utils/
│   ├── test_helpers.py
│   ├── test_employee.py
│   └── test_machine.py
└── test_scenarios.py        # Integration tests
```

### Debugging Tests

```python
# Add print statements
def test_something(self):
    result = service.method()
    print(f"Result: {result}")  # Will show with -v flag
    self.assertEqual(result, expected)

# Use pdb
def test_something(self):
    import pdb; pdb.set_trace()
    result = service.method()
    # Debugger will pause here

# Use unittest assertions
self.assertEqual(a, b)              # Equal
self.assertTrue(condition)           # True
self.assertIsNone(value)            # None
self.assertRaises(Exception, func)  # Raises exception
self.assertIn(item, container)      # Item in container
```

---

## Code Style Guide

### Naming Conventions

```python
# Classes: PascalCase
class KnowledgeGenerator:
    pass

# Functions/variables: snake_case
def generate_knowledge():
    pass

# Constants: UPPER_SNAKE_CASE
EMBEDDING_DIM = 768
DEFAULT_TIMEOUT = 30

# Private methods: _leading_underscore
def _internal_helper():
    pass

# Private attributes: _leading_underscore
self._cache = {}
```

### Type Hints

Always use type hints:

```python
# Good
def get_logs(self, factory_id: str) -> list:
    """Get logs for a factory"""
    pass

def calculate(value: int, multiplier: float = 2.0) -> float:
    return value * multiplier

# Avoid
def get_logs(factory_id):
    pass
```

### Docstrings

Use docstrings for all public functions:

```python
def generate_employee_statement(
    employee_code: str,
    machine_code: str,
    deeds: dict,
    overall: dict,
    avg_comp: dict
) -> str:
    """
    Generate a narrative statement about employee performance.
    
    Args:
        employee_code: The employee's code identifier
        machine_code: The machine code they operated
        deeds: Dictionary of employee performance metrics
        overall: Factory-wide statistics
        avg_comp: Average comparisons
    
    Returns:
        A formatted narrative string describing the employee's performance
    
    Example:
        >>> statement = generate_employee_statement("EMP001", "MACH01", {...}, {...}, {...})
        >>> print(statement)
        Employee EMP001 (Machine MACH01) Performance Report: ...
    """
    # Implementation
    pass
```

### Code Format

```python
# Use Black formatter
pip install black
black .  # Format all files

# Check with flake8
pip install flake8
flake8 .

# Type checking with mypy
pip install mypy
mypy services/
```

### Import Organization

```python
# 1. Standard library
import json
import os
from abc import ABC, abstractmethod

# 2. Third-party libraries
import numpy as np
from flask import Flask

# 3. Local imports
from interfaces.Storage import Storage
from services.knowledge_generator import KnowledgeGenerator
```

---

## Common Development Tasks

### Task 1: Add a New API Endpoint

```python
# In app.py
from flask import request, jsonify

@app.route("/new-endpoint", methods=["POST"])
def new_endpoint():
    """Handle new endpoint request"""
    try:
        # Get request data
        data = request.get_json()
        factory_id = data.get("factory_id")
        
        # Validate
        if not factory_id:
            return jsonify({"error": "Missing factory_id"}), 400
        
        # Process
        result = knowledge_generator.generate_knowledge()
        
        # Return response
        return jsonify({
            "message": "Success",
            "data": result
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### Task 2: Add Configuration Management

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    DEBUG = False
    TESTING = False
    EMBEDDING_DIM = 768
    DEFAULT_TOP_K = 5
    
    # Storage configuration (supports multiple backends)
    STORAGE_TYPE = os.getenv("STORAGE_TYPE", "supabase")  # supabase, mongodb, postgres
    
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    # MongoDB
    MONGODB_URI = os.getenv("MONGODB_URI")
    
    # PostgreSQL
    POSTGRES_URI = os.getenv("POSTGRES_URI")
    
    # Model configuration (supports multiple models)
    MODEL_TYPE = os.getenv("MODEL_TYPE", "qwen3")  # qwen3, openai, huggingface
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL")

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

# In app.py - Factory pattern for creating instances
import os
from config import DevelopmentConfig, ProductionConfig

config = ProductionConfig if os.getenv("ENV") == "prod" else DevelopmentConfig

# Storage Factory - Choose your storage backend
def create_storage(config):
    """Create storage instance based on configuration"""
    if config.STORAGE_TYPE == "supabase":
        from storage.SupabaseStorage import SupabaseStorage
        return SupabaseStorage()
    elif config.STORAGE_TYPE == "mongodb":
        from storage.MongoDBStorage import MongoDBStorage
        return MongoDBStorage(config.MONGODB_URI)
    elif config.STORAGE_TYPE == "postgres":
        from storage.PostgreSQLStorage import PostgreSQLStorage
        return PostgreSQLStorage(config.POSTGRES_URI)
    else:
        raise ValueError(f"Unknown storage type: {config.STORAGE_TYPE}")

# Model Factory - Choose your embedding model
def create_model(config):
    """Create embedding model instance based on configuration"""
    if config.MODEL_TYPE == "qwen3":
        from model.Qwen3 import Qwen3
        return Qwen3()
    elif config.MODEL_TYPE == "openai":
        from model.OpenAIEmbeddings import OpenAIEmbeddings
        return OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
    elif config.MODEL_TYPE == "huggingface":
        from model.HuggingFaceEmbeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=config.HUGGINGFACE_MODEL)
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")

# Usage in app.py
storage = create_storage(config)
embed_model = create_model(config)

report_generator = ReportGenerator(storage, embed_model)
knowledge_generator = KnowledgeGenerator(storage, embed_model)
```

### Task 3: Add Logging

```python
# utils/logger.py
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

# Usage
from utils.logger import get_logger

logger = get_logger(__name__)

class KnowledgeGenerator:
    def generate_knowledge(self):
        logger.info("Starting knowledge generation")
        try:
            queue = self.storage.get_factories()
            logger.debug(f"Processing {len(queue)} factories")
        except Exception as e:
            logger.error(f"Error: {e}")
```

### Task 4: Add Performance Metrics

```python
# utils/metrics.py
import time
from functools import wraps

def measure_execution_time(func):
    """Decorator to measure function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper

# Usage
class ReportGenerator:
    @measure_execution_time
    def generate_report(self, type, factory_id, query, top_k=5):
        # ... implementation
        pass
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'interfaces'"

**Solution**: 
1. Ensure you're running from project root
2. Check Python path includes project
3. Verify virtual environment is activated

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Verify you're in right directory
pwd  # Should end with summerizer_1.1
```

### Issue: "TypeError: Can't instantiate abstract class X"

**Solution**: You forgot to implement an abstract method

```python
# This will fail
class MyStorage(Storage):
    def get_logs(self, factory_id):
        pass
    # Missing other abstract methods!

# This will work
class MyStorage(Storage):
    def get_logs(self, factory_id):
        return []
    
    def push_embedding(self, vector, factory_id, type, statement, code):
        return True
    
    # ... all abstract methods implemented
```

### Issue: "Database connection failed"

**Solution**: Check your database credentials based on your storage backend

```bash
# Verify .env file exists
cat .env

# For Supabase
echo $SUPABASE_URL
echo $SUPABASE_KEY
python -c "from integration.supabase_client import sp; print(sp.table('factory').select('*').execute())"

# For MongoDB
echo $MONGODB_URI
python -c "from pymongo import MongoClient; client = MongoClient('mongodb://...'); print(client.server_info())"

# For PostgreSQL
echo $POSTGRES_URI
python -c "import psycopg2; conn = psycopg2.connect('postgresql://...'); print(conn.get_dsn_parameters())"
```

### Issue: Tests fail with "AssertionError"

**Solution**: Check your test data and assertions

```python
# Add debugging
def test_something(self):
    result = service.method()
    print(f"Expected: {expected}, Got: {result}")  # Debug output
    self.assertEqual(result, expected)

# Run with verbose output
python -m unittest tests.test_file -v
```

---

## Contributing Best Practices

### Before Submitting Code

1. **Run all tests** (tests use mocks and work with any storage/model):
```bash
python -m unittest discover -s tests -p "test_*.py"
```

2. **Format code**:
```bash
black .
```

3. **Check style**:
```bash
flake8 .
```

4. **Type check**:
```bash
mypy .
```

5. **Write tests** for new features:
```bash
# Ensure coverage
coverage run -m unittest discover -s tests
coverage report
```

6. **Test with your chosen storage/model** (optional, if making storage/model-specific changes):
```bash
# Set environment variables for your backend
export STORAGE_TYPE=mongodb  # or supabase, postgres
export MODEL_TYPE=openai     # or qwen3, huggingface

# Run integration tests
python app.py
```

### Commit Message Guidelines

```
[type]: Brief description

- Detailed explanation of changes
- List of modifications
- Any breaking changes

Examples:
[feature]: Add caching layer to report generator
[bugfix]: Fix embedding dimension mismatch in similarity
[refactor]: Extract common logic into helper functions
[docs]: Update API reference with new methods
[test]: Add integration tests for scenarios
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added/updated tests
- [ ] All tests pass
- [ ] Tested locally with storage backend: _________ (Supabase/MongoDB/PostgreSQL/etc.)
- [ ] Tested locally with model backend: _________ (Qwen3/OpenAI/HuggingFace/etc.)

## Checklist
- [ ] Code formatted with Black
- [ ] Type hints added
- [ ] Docstrings included
- [ ] Tests written
- [ ] No hardcoded credentials
- [ ] Works with any Storage/EmbeddingModel implementation (if applicable)
```

---

## Resources

- [Python ABC Documentation](https://docs.python.org/3/library/abc.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Unit Testing Python](https://docs.python.org/3/library/unittest.html)
- [Type Hints PEP 484](https://www.python.org/dev/peps/pep-0484/)
- [Black Code Formatter](https://github.com/psf/black)

---

## Questions?

Refer to:
- [README.md](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design patterns
- [API_REFERENCE.md](API_REFERENCE.md) - Class/method reference
- Test files for usage examples
