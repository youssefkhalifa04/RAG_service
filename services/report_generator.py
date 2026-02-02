from interfaces.Storage import Storage
from interfaces.EmbeddingModel import EmbeddingModel
from services.data_encoder import DataEncoder
from utils.helpers import safe_parse_embedding
import numpy as np

class ReportGenerator:
    def __init__(self, storage : Storage , embed_model : EmbeddingModel ):
        self.storage = storage
        self.embed_model = embed_model
        self.encoder = DataEncoder(self.storage , self.embed_model)
    def generate_report(self, type: str, factory_id: str, top_k: int = 5 , query: str = None):
        records = self.storage.get_knowledge(factory_id, type)
        if not query:
            raise ValueError("Query must be provided for report generation.")
        try : 
            query_vec = self.embed_model.encode([query])
        except Exception as e:
            print(f"Error encoding query: {e}")
            raise ValueError("Error encoding query")

        valid_vectors = []
        valid_records = []
        for r in records:
            
            vec = safe_parse_embedding(r["embedding"])

            print(f"\033[92mParsed vector: \033[91m{vec[0:10]}\033[0m")
            if vec is not None:
                valid_vectors.append(vec)
                valid_records.append(r)
        print(f"\033[92mFound {len(valid_vectors)} valid vectors out of {len(records)} records.\033[0m")
        if not valid_vectors:
            return "ERROR: No valid vectors found."
        sim = self.embed_model.similarity(np.array(valid_vectors), query_vec)

        print(f"\033[92mSimilarity scores: \033[91m{sim}\033[0m")
        top_indices = sim.argsort()[-top_k:][::-1]

        reports = []

        for idx in top_indices:
            statement = valid_records[idx]["statement"]

            # Better structured prompt that won't hallucinate
            messages = [
                {"role": "system", "content": "You are a factory audit extraction tool. Extract only factual information."},
                {"role": "user", "content": f"Extract employee code, machine code, downtime minutes, and waste count from this log:\n\n{statement}\n\nProvide ONLY: Code: [CODE], Downtime: [MIN], Waste: [COUNT]"}
            ]

            try:
                output = self.embed_model.getResponse(messages)
                report = output[0]["generated_text"].split("<|im_start|>assistant\n")[-1].strip()
                reports.append(report)
            except Exception as e:
                print(f"Error generating report: {e}")
                raise ValueError("Error generating report")

        final_report = "\n\n---\n\n".join(reports)
        return final_report