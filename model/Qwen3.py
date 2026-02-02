from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from interfaces.EmbeddingModel import EmbeddingModel


class Qwen3(EmbeddingModel):
    def __init__(self, 
                 model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",  # ✓ Use instruction-tuned model
                 embed_model_id: str = "sentence-transformers/all-mpnet-base-v2",  # ✓ Use proper embedding model
                 device: str = "cpu"):
        self.device = device
        self.model_id = model_id
        self.embed_model_id = embed_model_id

        # Load generation model (instruction-tuned)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            torch_dtype=torch.float32, 
            device_map="auto"
        ).to(self.device) 

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Load embedding model (separate from generation)
        self.embed_model = SentenceTransformer(self.embed_model_id)

    def encode(self, documents, prompt_name: str = "default"):
        if prompt_name == "query":
            embeddings = self.embed_model.encode(documents, prompt_name=prompt_name)
        else:
            embeddings = self.embed_model.encode(documents)
        return embeddings
        
    def similarity(self, doc_vecs, query_vec):
        db_tensor = torch.tensor(np.array(doc_vecs), dtype=torch.float32)
        q_tensor = torch.tensor(query_vec, dtype=torch.float32)
        return self.embed_model.similarity(q_tensor, db_tensor)[0].cpu().numpy()
    
    def get_model_id(self) -> str:
        return self.model_id
    
    def getResponse(self, messages):
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        output = self.generator(
            prompt,
            max_new_tokens=150,
            do_sample=True,           # Enable sampling for better quality
            temperature=0.3,          # Low temperature for focused output
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return output