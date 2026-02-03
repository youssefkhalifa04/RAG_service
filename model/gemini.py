from google import genai
from google.genai import types
import numpy as np
import torch
from interfaces.EmbeddingModel import EmbeddingModel

class GeminiModel(EmbeddingModel):
    def __init__(self, 
                 api_key: str,
                 model_id: str = "gemini-3.0-flash", # Latest stable in 2026
                 embed_model_id: str = "text-embedding-004"):
        
        # New SDK uses a Client object
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.embed_model_id = embed_model_id

    def encode(self, documents, prompt_name: str = "default"):
        if isinstance(documents, str):
            documents = [documents]
            
        # Mapping task types for the new SDK
        task = "RETRIEVAL_QUERY" if prompt_name == "query" else "RETRIEVAL_DOCUMENT"
        
        result = self.client.models.embed_content(
            model=self.embed_model_id,
            contents=documents,
            config=types.EmbedContentConfig(task_type=task)
        )
        
        # Extract the values from the response
        return np.array([e.values for e in result.embeddings])

    def similarity(self, doc_vecs, query_vec):
        q_tensor = torch.tensor(np.array(query_vec), dtype=torch.float32)
        db_tensor = torch.tensor(np.array(doc_vecs), dtype=torch.float32)
        
        # Handle dimensionality for single vs batch queries
        if q_tensor.ndim == 1: q_tensor = q_tensor.unsqueeze(0)
        
        cos = torch.nn.CosineSimilarity(dim=1)
        return cos(q_tensor, db_tensor).cpu().numpy()

    def get_model_id(self) -> str:
        return self.model_id

    def getResponse(self, messages):
        # Extract the system instruction and user content
        system_instruction = next((m['content'] for m in messages if m['role'] == 'system'), "")
        user_content = next((m['content'] for m in messages if m['role'] == 'user'), "")

        # New generate_content call structure
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
                max_output_tokens=500
            )
        )
        
        return [{"generated_text": response.text}]