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
        query_vec = self.embed_model.encode([query], prompt_name="query")

        valid_vectors = []
        valid_records = []
        for r in records:
            vec = safe_parse_embedding(r["embedding"])
            if vec is not None:
                valid_vectors.append(vec)
                valid_records.append(r)
        print(f"\033[92mFound {len(valid_vectors)} valid vectors out of {len(records)} records.\033[0m")
        if not valid_vectors:
            return "ERROR: No valid vectors found."
        sim = self.embed_model.similarity(np.array(valid_vectors), query_vec)
        top_indices = sim.argsort()[-top_k:][::-1]

        reports = []

        for idx in top_indices:
            statement = valid_records[idx]["statement"]

            system_prompt = (
                "You are a factual Data Extraction tool.\n"
                "Extract only factual information from the log.\n"
                "List codes (EMP001, MACH01), downtime, waste if present.\n\n"
                f"LOG TO ANALYZE:\n{statement}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the structured audit report."}
            ]

            output = self.embed_model.getResponse(messages)

            report = output[0]["generated_text"].split("<|im_start|>assistant\n")[-1].strip()
            reports.append(report)

        
        final_report = "\n\n---\n\n".join(reports)

        return final_report