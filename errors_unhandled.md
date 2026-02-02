# Unhandled Errors Review

This file highlights runtime failure points and unhandled exceptions that can surface during execution.

---

## 1) App startup failures (uncaught)

- `Qwen3()` and `SupabaseStorage()` are instantiated at import time. If model download fails or credentials/env are missing, the app crashes before endpoints load.  
  [app.py](app.py#L10-L13)

- Model loading (`AutoTokenizer.from_pretrained`, `AutoModelForCausalLM.from_pretrained`, `SentenceTransformer`) can raise network/IO errors.  
  [model/Qwen3.py](model/Qwen3.py#L16-L26)

- Supabase client creation occurs at import time and can raise if env vars are missing or invalid.  
  [integration/supabase_client.py](integration/supabase_client.py#L18-L21)

---

## 2) Storage errors returned as strings (type mismatch)

Several `Storage` methods return error **strings**, but callers expect **lists/dicts**. This leads to downstream exceptions like `TypeError` or `KeyError` when iterating:

- `get_logs`, `get_knowledge`, `get_employees`, `get_machines`, `get_*_performance` return `"DATABASE ERROR: ..."`  
  [storage/SupabaseStorage.py](storage/SupabaseStorage.py#L8-L85)

Impact on services:

- Iterates `logs` and accesses keys directly → fails if `logs` is a string.  
  [services/knowledge_generator.py](services/knowledge_generator.py#L18-L38)

- Iterates `records` and accesses `r["embedding"]` → fails if `records` is a string.  
  [services/report_generator.py](services/report_generator.py#L11-L21)

---

## 3) Uncaught exceptions in database access

- `get_factories()` has no try/except. Any DB error bubbles up.  
  [storage/SupabaseStorage.py](storage/SupabaseStorage.py#L51-L54)

---

## 4) Assumed log schema (KeyError risk)

Multiple places assume log entries always contain `employee`, `machine`, `event_type`, `event_duration`:

- `KnowledgeGenerator` assumes `log["employee"]["code"]` and `log["machine"]["code"]`.  
  [services/knowledge_generator.py](services/knowledge_generator.py#L25-L38)

- Employee helpers assume same structure.  
  [utils/employee.py](utils/employee.py#L2-L13)

- Machine helpers assume same structure.  
  [utils/machine.py](utils/machine.py#L2-L11)

If any log entry is missing these keys, a `KeyError` will be raised.

---

## 5) Report generation assumptions

- Assumes `get_knowledge()` returns list of dicts with `embedding` and `statement`.  
  [services/report_generator.py](services/report_generator.py#L11-L33)

- Assumes model response has `output[0]["generated_text"]`.  
  [services/report_generator.py](services/report_generator.py#L44-L46)

- `getResponse()` can raise or return unexpected format from model pipeline.  
  [model/Qwen3.py](model/Qwen3.py#L41-L44)

---

## 6) Data encoding failures not handled in `DataEncoder`

- `encode()` or `push_embedding()` errors bubble out without handling here.  
  [services/data_encoder.py](services/data_encoder.py#L10-L15)

---

## Summary

Main risks are:

- **Startup crashes** due to model/client initialization.
- **Inconsistent return types** from `Storage` methods.
- **Schema assumptions** on logs and model responses.

These can be addressed by:

- Raising exceptions instead of returning strings in storage.
- Validating inputs/outputs in services.
- Defensive checks for expected schema keys and model response structure.
