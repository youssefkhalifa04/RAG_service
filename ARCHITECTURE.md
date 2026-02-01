# Architecture Documentation - Summerizer 1.1

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Design Patterns](#design-patterns)
3. [Abstraction Layers](#abstraction-layers)
4. [Polymorphism in Action](#polymorphism-in-action)
5. [Data Flow](#data-flow)
6. [Class Hierarchy](#class-hierarchy)
7. [Module Interactions](#module-interactions)
8. [Extension Points](#extension-points)

---

## System Architecture

### Architectural Overview

Summerizer uses a **layered, plugin-based architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
│                       (Flask App)                           │
├─────────────────────────────────────────────────────────────┤
│                    SERVICE LAYER                            │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ KnowledgeGenerat │  │  ReportGenerator │                 │
│  │       or         │  │                  │                 │
│  └──────────────────┘  └──────────────────┘                 │
│  ┌──────────────────┐                                       │
│  │   DataEncoder    │                                       │
│  └──────────────────┘                                       │
├─────────────────────────────────────────────────────────────┤
│                  ABSTRACTION LAYER                          │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │  Storage (ABC)   │  │ EmbeddingModel   │                 │
│  │                  │  │     (ABC)        │                 │
│  └──────────────────┘  └──────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│                IMPLEMENTATION LAYER                         │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │SupabaseStorage   │  │     Qwen3        │                 │
│  │  (Database)      │  │ (Model/LLM)      │                 │
│  └──────────────────┘  └──────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│                  UTILITY LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ helpers  │  │ employee │  │ machine  │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
├─────────────────────────────────────────────────────────────┤
│                  EXTERNAL SERVICES                          │
│                  ┌─────────────────┐                        │
│                  │  Supabase (DB)  │                        │
│                  │  Qwen3 (LLM)    │                        │
│                  │ SentenceTransf. │                        │
│                  └─────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Layered Architecture**: Clear separation between presentation, business logic, abstraction, and implementation
2. **Plugin-Based**: New implementations can be added without modifying existing code
3. **Loose Coupling**: Services depend on abstractions, not concrete implementations
4. **High Cohesion**: Each module has a single, well-defined responsibility
5. **Dependency Injection**: Dependencies are provided via constructors

---

## Design Patterns

### 1. Abstract Factory Pattern (Abstraction)

**Location**: `interfaces/Storage.py`, `interfaces/EmbeddingModel.py`

**Purpose**: Define contracts that concrete implementations must follow

**Storage Interface**:
```python
class Storage(ABC):
    @abstractmethod
    def get_logs(self, factory_id: str):
        """Retrieve production logs for a factory"""
        
    @abstractmethod
    def push_embedding(self, vector, factory_id, type, statement, code) -> bool:
        """Store embedding and statement in knowledge base"""
        
    @abstractmethod
    def get_knowledge(self, factory_id, type):
        """Retrieve knowledge base entries"""
        
    # ... more abstract methods
```

**Why**:
- Services don't care about database implementation details
- Easy to swap databases (Supabase → PostgreSQL → MongoDB)
- Testable with mock implementations
- Enforces consistent interface across all storage backends

**EmbeddingModel Interface**:
```python
class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, documents):
        """Convert text to vector embeddings"""
        
    @abstractmethod
    def similarity(self, query_vec, doc_vecs):
        """Calculate semantic similarity"""
        
    @abstractmethod
    def getResponse(self, messages):
        """Generate LLM response"""
```

**Why**:
- Model-agnostic service layer
- Can upgrade models without changing services
- Support multiple models simultaneously
- Easy to test with mock embeddings

---

### 2. Polymorphism in Implementation

**Concrete Storage Implementation**:
```python
class SupabaseStorage(Storage):
    """Concrete implementation using Supabase"""
    
    def __init__(self):
        self.client = sp  # Supabase client
    
    def get_logs(self, factory_id: str):
        data = sp.table("production_logs").select(...).eq("factory_id", factory_id).execute()
        return data.data
    
    def push_embedding(self, vector, factory_id, type, statement, code) -> bool:
        # Insert into Supabase table
        sp.table("factory_knowledge_base").insert({...}).execute()
        return True
```

**Concrete EmbeddingModel Implementation**:
```python
class Qwen3(EmbeddingModel):
    """Concrete implementation using Qwen3 + SentenceTransformer"""
    
    def __init__(self):
        self.embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.generator = pipeline("text-generation", model="Qwen/Qwen3-Embedding-0.6B")
    
    def encode(self, documents):
        return self.embed_model.encode(documents)
    
    def similarity(self, doc_vecs, query_vec):
        return self.embed_model.similarity(query_vec, doc_vecs)[0].cpu().numpy()
```

**Polymorphic Usage in Services**:
```python
class KnowledgeGenerator:
    def __init__(self, storage: Storage, embed_model: EmbeddingModel):
        # Works with ANY Storage implementation
        # Works with ANY EmbeddingModel implementation
        self.storage = storage
        self.embed_model = embed_model
    
    def generate_knowledge(self):
        # Uses polymorphic methods
        logs = self.storage.get_logs(factory_id)  # Works with any storage
        vectors = self.embed_model.encode(text)    # Works with any model
```

**Benefits**:
- **Flexibility**: Swap implementations at runtime
- **Testability**: Inject mock implementations
- **Maintainability**: Change implementation without touching services
- **Extensibility**: Add new implementations without modifying existing code

---

### 3. Dependency Injection Pattern

**In app.py** (Configuration/Composition Root):
```python
from model.Qwen3 import Qwen3
from storage.SupabaseStorage import SupabaseStorage
from services.knowledge_generator import KnowledgeGenerator
from services.report_generator import ReportGenerator

# Create instances
model = Qwen3()
supabase = SupabaseStorage()

# Inject dependencies
report_generator = ReportGenerator(supabase, model)
knowledge_generator = KnowledgeGenerator(supabase, model)
```

**Why**:
- Single point of configuration
- Easy to switch implementations (just change one line)
- Services don't need to know how to create dependencies
- Testable with mock injections

---

### 4. Strategy Pattern (Similarity Algorithms)

Different similarity strategies can be swapped:

```python
class EmbeddingModel(ABC):
    @abstractmethod
    def similarity(self, query_vec, doc_vecs):
        """Strategy method - can have different implementations"""
```

**Qwen3 Strategy**:
```python
def similarity(self, doc_vecs, query_vec):
    # Uses SentenceTransformer's cosine similarity
    db_tensor = torch.tensor(np.array(doc_vecs), dtype=torch.float32)
    q_tensor = torch.tensor(query_vec, dtype=torch.float32)
    return self.embed_model.similarity(q_tensor, db_tensor)[0].cpu().numpy()
```

**Could easily swap for**:
```python
def similarity(self, doc_vecs, query_vec):
    # Uses euclidean distance
    return compute_euclidean_distance(doc_vecs, query_vec)

def similarity(self, doc_vecs, query_vec):
    # Uses dot product
    return np.dot(doc_vecs, query_vec)
```

---

### 5. Decorator Pattern (DataEncoder)

`DataEncoder` wraps the encode → store workflow:

```python
class DataEncoder:
    def __init__(self, storage: Storage, embed_model: EmbeddingModel):
        self.storage = storage
        self.embed_model = embed_model
    
    def encode_data(self, documents, type_label, factory_id, code):
        # Decorator combines encoding + storage
        vectors = self.embed_model.encode(np.array([documents]))
        result = self.storage.push_embedding(vectors, factory_id, type_label, documents, code)
        return result
```

**Benefits**:
- Single responsibility principle
- Reusable encode-then-store workflow
- Can add logging, caching, validation without changing services

---

### 6. Pipeline Pattern (KnowledgeGenerator)

Orchestrates multi-step workflow:

```
Logs → Extract Metrics → Generate Statements → Encode → Store
```

```python
def generate_knowledge(self):
    queue = self.storage.get_factories()  # Step 0: Get factories
    
    for factory_id in queue:
        logs = self.storage.get_logs(factory_id)  # Step 1: Fetch logs
        
        # Step 2: Extract metrics
        stats = {
            "total_breakdowns": len([...]),
            "total_good_pieces": len([...]),
            # ...
        }
        
        for emp_code, machine_code in emp_mach_codes:
            # Step 3: Generate statements
            deeds = get_single_employee_deeds(emp_code, logs)
            statement = generate_employee_statement(...)
            
            # Step 4: Encode and store
            self.encoder.encode_data(statement, "employee", factory_id, emp_code)
```

---

## Abstraction Layers

### Layer 1: Service Layer
Services orchestrate business logic using abstractions.

**Services**:
- `KnowledgeGenerator`: Processes logs → generates knowledge base
- `ReportGenerator`: Accepts queries → returns semantic search results + LLM report
- `DataEncoder`: Encodes text → stores embeddings

**Characteristics**:
- Don't import concrete implementations
- Only depend on abstract interfaces
- Testable with mock implementations

### Layer 2: Abstraction Layer (Interfaces)
Defines contracts that implementations must fulfill.

**Interfaces**:
- `Storage`: Database operations contract
- `EmbeddingModel`: Embedding and LLM operations contract

**Characteristics**:
- Abstract base classes using Python's `ABC` module
- All methods are abstract (must be implemented)
- Service layer only knows about these interfaces

### Layer 3: Implementation Layer
Concrete classes that implement abstractions.

**Implementations**:
- `SupabaseStorage`: Uses Supabase PostgreSQL database
- `Qwen3`: Uses Qwen3 LLM + SentenceTransformer embeddings

**Characteristics**:
- Inherit from abstract interfaces
- Implement all abstract methods
- Can be swapped without affecting services

### Layer 4: External Services
Third-party services used by implementations.

**Services**:
- Supabase: PostgreSQL database
- Qwen3: Large language model
- SentenceTransformer: Text encoding

---

## Polymorphism in Action

### Example 1: Storage Polymorphism

**Service layer**:
```python
class KnowledgeGenerator:
    def __init__(self, storage: Storage, ...):
        self.storage = storage  # Could be ANY Storage implementation
```

**Usage at runtime**:
```python
# Production: Use Supabase
storage = SupabaseStorage()
generator = KnowledgeGenerator(storage)

# Testing: Use Mock
storage = MockStorage()
generator = KnowledgeGenerator(storage)

# Future: Use PostgreSQL directly
storage = PostgreSQLStorage()
generator = KnowledgeGenerator(storage)
```

**Why this is powerful**:
- Same service code works with different storage backends
- No if/else statements for different storage types
- New storage implementations don't require service changes

### Example 2: EmbeddingModel Polymorphism

**Service layer**:
```python
class ReportGenerator:
    def __init__(self, storage: Storage, embed_model: EmbeddingModel):
        self.embed_model = embed_model  # Could be ANY EmbeddingModel
```

**Usage at runtime**:
```python
# Current: Use Qwen3 + SentenceTransformer
model = Qwen3(
    embed_model_id="sentence-transformers/all-mpnet-base-v2"
)

# Upgrade: Use better model
model = Qwen3(
    embed_model_id="sentence-transformers/all-mpnet-base-v2-large"
)

# Future: Use different model
class OpenAIEmbeddings(EmbeddingModel):
    def encode(self, documents):
        return openai.Embedding.create(input=documents)

model = OpenAIEmbeddings()

# Same report generator works with all
report_gen = ReportGenerator(storage, model)
```

---

## Data Flow

### Knowledge Generation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ START: generate_knowledge()                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Fetch all factories                                         │
│ storage.get_factories() → [factory_id1, factory_id2, ...]   │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────┐
        │ For each factory_id:                    │
        └─────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────┐
        │ Fetch production logs                   │
        │ storage.get_logs(factory_id)            │
        │ → logs = [{employee, machine, event}...]│
        └─────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────┐
        │ Extract metrics from logs               │
        │ stats = {                               │
        │   total_breakdowns: count,              │
        │   total_good_pieces: count,             │
        │   ...                                   │
        │ }                                       │
        │ avg_comp = {                            │
        │   avg_breakdowns: total/employees,      │
        │   ...                                   │
        │ }                                       │
        └─────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────┐
        │ For each (employee, machine) pair:      │
        └─────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────┐
        │ Extract employee deeds                  │
        │ deeds = get_single_employee_deeds(...)  │
        │ → {breakdowns, pauses, pieces, ...}     │
        └─────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────┐
        │ Generate narrative statement            │
        │ statement = generate_employee_statement │
        │ → "Employee EMP001 had 5 breakdowns..." │
        └─────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────┐
        │ Encode statement to vector              │
        │ vector = embed_model.encode(statement)  │
        │ → 768-dimensional vector                │
        └─────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────┐
        │ Store embedding                         │
        │ storage.push_embedding(                 │
        │   vector,                               │
        │   factory_id,                           │
        │   "employee",                           │
        │   statement,                            │
        │   employee_code                         │
        │ )                                       │
        └─────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────────┐
        │ Repeat for machines...                  │
        └─────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ END: Knowledge base populated with embeddings               │
└─────────────────────────────────────────────────────────────┘
```

### Report Generation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ START: generate_report(query="which employee has...")       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 1. RETRIEVAL: Get knowledge base entries                    │
│    records = storage.get_knowledge(                         │
│      factory_id, "employee"                                 │
│    )                                                        │
│    → [{embedding, statement}, ...]                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. ENCODING: Convert query to vector                        │
│    query_vec = embed_model.encode(                          │
│      [query],                                               │
│      prompt_name="query"                                    │
│    )                                                        │
│    → 768-dimensional vector                                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. SIMILARITY: Calculate cosine similarity                  │
│    similarities = embed_model.similarity(                   │
│      doc_vecs=[all embeddings],                             │
│      query_vec=query_vector                                 │
│    )                                                        │
│    → [0.85, 0.72, 0.91, ...]                                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. RANKING: Get top-k most similar                          │
│    top_indices = np.argsort(similarities)[-top_k:][::-1]    │
│    → [2, 0, 3, ...] (indices of top matches)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. GENERATION: LLM generates reports for each top result    │
│    For each top record:                                     │
│      messages = [                                           │
│        {role: "system", content: prompt},                   │
│        {role: "user", content: query}                       │
│      ]                                                      │
│      output = embed_model.getResponse(messages)             │
│      → "Based on the log: ..."                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. AGGREGATION: Combine reports                             │
│    final_report = "\n---\n".join([                          │
│      report1, report2, report3, ...                         │
│    ])                                                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ END: Return aggregated report                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Class Hierarchy

### Storage Hierarchy

```
ABC (Python)
  │
  └─ Storage (Abstract Interface)
      │
      ├─ SupabaseStorage (Concrete Implementation)
      │   ├─ get_logs()
      │   ├─ push_embedding()
      │   ├─ get_knowledge()
      │   ├─ get_factories()
      │   ├─ get_employees()
      │   ├─ get_machines()
      │   └─ ...
      │
      └─ [Future] PostgreSQLStorage (Potential)
      │   └─ [Implements all Storage methods]
      │
      └─ [Testing] MockStorage (For tests)
          └─ [In-memory implementation]
```

### EmbeddingModel Hierarchy

```
ABC (Python)
  │
  └─ EmbeddingModel (Abstract Interface)
      │
      ├─ Qwen3 (Concrete Implementation)
      │   ├─ encode()
      │   ├─ similarity()
      │   ├─ getResponse()
      │   └─ get_model_id()
      │
      └─ [Future] OpenAIEmbeddings (Potential)
      │   └─ [Uses OpenAI API]
      │
      └─ [Testing] MockEmbeddingModel (For tests)
          └─ [Deterministic embeddings]
```

---

## Module Interactions

### Service Dependencies

```
                    ┌──────────────────┐
                    │   app.py         │
                    │ (Configuration)  │
                    └──────────────────┘
                           │ │
            ┌──────────────┘ └──────────────┐
            │                               │
    ┌──────▼──────────┐           ┌────────▼─────────┐
    │ KnowledgeGenerat│           │ ReportGenerator  │
    │     or          │           │                  │
    └──────┬──────────┘           └────────┬─────────┘
           │                               │
           │                    ┌──────────┴──────────┐
           │                    │                     │
    ┌──────▼──────────┐  ┌──────▼────────┐  ┌────────▼─────┐
    │  DataEncoder    │  │  DataEncoder  │  │  Interfaces  │
    └──────┬──────────┘  └──────┬────────┘  │              │
           │                    │           │ Storage      │
           │                    │           │ EmbeddingMod │
           │                    │           │              │
           └────────┬───────────┘           └──────┬───────┘
                    │                              │
         ┌──────────┴──────────────┬───────────────┘
         │                         │
    ┌────▼────────────┐    ┌──────▼──────────┐
    │ SupabaseStorage │    │     Qwen3       │
    │ (Implements     │    │  (Implements    │
    │   Storage)      │    │ EmbeddingModel) │
    └─────────────────┘    └─────────────────┘
         │                         │
         ├─→ Supabase DB    ├─→ SentenceTransformer
         │                 └─→ Qwen3 LLM
         └─→ Production logs
```

---

## Extension Points

### 1. Adding a New Storage Backend

**Step 1**: Create new class inheriting from Storage

```python
# storage/MongoDBStorage.py
from interfaces.Storage import Storage

class MongoDBStorage(Storage):
    def __init__(self, connection_string):
        self.client = MongoClient(connection_string)
        self.db = self.client.factory_analytics
    
    def get_logs(self, factory_id: str):
        # MongoDB specific implementation
        return self.db.production_logs.find({"factory_id": factory_id})
    
    def push_embedding(self, vector, factory_id, type, statement, code):
        # MongoDB specific implementation
        self.db.knowledge_base.insert_one({
            "factory_id": factory_id,
            "type": type,
            "embedding": vector,
            "statement": statement
        })
        return True
    
    # ... implement other abstract methods
```

**Step 2**: Update app.py

```python
from storage.MongoDBStorage import MongoDBStorage

# Replace
# supabase = SupabaseStorage()

# With
mongo = MongoDBStorage("mongodb://localhost:27017")

report_generator = ReportGenerator(mongo, model)
knowledge_generator = KnowledgeGenerator(mongo, model)
```

**Step 3**: Tests automatically work

```python
# All existing tests now work with MongoDBStorage
# Just inject the new implementation
storage = MongoDBStorage(...)
generator = KnowledgeGenerator(storage, model)
```

### 2. Adding a New Embedding Model

**Step 1**: Create new class inheriting from EmbeddingModel

```python
# model/OpenAIEmbeddings.py
from interfaces.EmbeddingModel import EmbeddingModel
import openai

class OpenAIEmbeddings(EmbeddingModel):
    def __init__(self, api_key, embedding_model="text-embedding-3-large"):
        self.api_key = api_key
        self.embedding_model = embedding_model
        openai.api_key = api_key
    
    def encode(self, documents):
        response = openai.Embedding.create(
            input=documents,
            model=self.embedding_model
        )
        return [item['embedding'] for item in response['data']]
    
    def similarity(self, query_vec, doc_vecs):
        # Use cosine similarity
        import numpy as np
        norm_query = query_vec / np.linalg.norm(query_vec)
        norm_docs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        return np.dot(norm_docs, norm_query)
    
    def get_model_id(self) -> str:
        return self.embedding_model
    
    def getResponse(self, messages):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return response['choices'][0]['message']['content']
```

**Step 2**: Update app.py

```python
from model.OpenAIEmbeddings import OpenAIEmbeddings

# Replace
# model = Qwen3()

# With
model = OpenAIEmbeddings(api_key="sk-...")

report_generator = ReportGenerator(supabase, model)
knowledge_generator = KnowledgeGenerator(supabase, model)
```

**Step 3**: All services work automatically

```python
# No changes needed to services
# They only depend on the interface
report_gen.generate_report(...)  # Works with OpenAI embeddings
```

### 3. Adding a New Service

**Example**: Create a FeedbackService

```python
# services/feedback_service.py
from interfaces.Storage import Storage
from interfaces.EmbeddingModel import EmbeddingModel

class FeedbackService:
    def __init__(self, storage: Storage, embed_model: EmbeddingModel):
        self.storage = storage
        self.embed_model = embed_model
    
    def store_feedback(self, factory_id, statement, rating):
        """Store user feedback about generated reports"""
        vector = self.embed_model.encode([statement])[0]
        self.storage.push_embedding(
            vector,
            factory_id,
            "feedback",
            statement,
            f"feedback_{rating}"
        )
        return True
    
    def get_helpful_reports(self, factory_id, threshold=4.0):
        """Get highly-rated reports"""
        records = self.storage.get_knowledge(factory_id, "feedback")
        return [r for r in records if float(r['code'].split('_')[1]) >= threshold]
```

**Usage in app.py**:

```python
feedback_service = FeedbackService(supabase, model)

@app.route("/feedback", methods=["POST"])
def feedback():
    factory_id = request.json['factory_id']
    statement = request.json['statement']
    rating = request.json['rating']
    
    feedback_service.store_feedback(factory_id, statement, rating)
    return {"message": "Feedback stored"}
```

---

## Conclusion

The Summerizer architecture exemplifies:

1. **Abstraction**: Abstract interfaces define contracts
2. **Polymorphism**: Multiple implementations of same interface
3. **Loose Coupling**: Services depend on abstractions, not implementations
4. **Extensibility**: New implementations added without modifying existing code
5. **Testability**: Mock implementations easily injected for testing

This design allows the system to remain flexible and maintainable as requirements evolve.
