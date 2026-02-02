from interfaces.Storage import Storage
from interfaces.EmbeddingModel import EmbeddingModel
from services.data_encoder import DataEncoder
from utils.helpers import safe_parse_embedding
import numpy as np

class ReportGenerator:
    def __init__(self, storage: Storage, embed_model: EmbeddingModel):
        self.storage = storage
        self.embed_model = embed_model
        self.encoder = DataEncoder(self.storage, self.embed_model)
    
    def generate_report(self, type: str, factory_id: str, top_k: int = 5, query: str = None, use_llm: bool = False):
        """
        Generate a report based on semantic search.
        
        Args:
            type: "employee" or "machine"
            factory_id: Factory identifier
            top_k: Number of top results to return
            query: Search query
            use_llm: If False, return raw statements (recommended). If True, use LLM extraction.
        """
        records = self.storage.get_knowledge(factory_id, type)
        if not records or isinstance(records, str):
            return f"ERROR: {records}"
        
        if not query:
            raise ValueError("Query must be provided for report generation.")
        
        # Enrich query with domain keywords for better similarity
        enriched_query = f"{query} performance breakdown downtime waste production efficiency metrics"
        
        try:
            query_vec = self.embed_model.encode([enriched_query], prompt_name="query")
        except Exception as e:
            print(f"Error encoding query: {e}")
            raise ValueError("Error encoding query")

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
        print(f"\033[92mSimilarity scores: \033[91m{sim}\033[0m")
        
        # Filter by minimum threshold
        MIN_SIMILARITY = 0.3
        valid_indices = [i for i, score in enumerate(sim) if score >= MIN_SIMILARITY]
        
        if not valid_indices:
            return f"ERROR: No relevant records found (all similarities < {MIN_SIMILARITY})"
        
        # Sort by similarity and take top_k
        sorted_indices = sorted(valid_indices, key=lambda i: sim[i], reverse=True)[:top_k]
        
        # Option 1: Return raw statements (RECOMMENDED - no hallucination)
        if not use_llm:
            reports = []
            for idx in sorted_indices:
                statement = valid_records[idx]["statement"]
                similarity = sim[idx]
                reports.append(f"[Relevance: {similarity:.2%}]\n{statement}")
            return "\n\n" + "="*80 + "\n\n".join(reports)
        
        # Option 2: Use LLM extraction (can hallucinate)
        reports = []
        for idx in sorted_indices:
            statement = valid_records[idx]["statement"]
            similarity = sim[idx]
            
            # Improved prompt with better structure
            if type == "employee":
                extraction_prompt = (
                    f"Extract the following from this employee performance report:\n\n"
                    f"{statement}\n\n"
                    f"Format your answer EXACTLY as:\n"
                    f"Employee: [CODE]\n"
                    f"Machine: [CODE]\n"
                    f"Breakdowns: [COUNT] incidents\n"
                    f"Downtime: [MINUTES] minutes\n"
                    f"Good units: [COUNT]\n"
                    f"Defective units: [COUNT]\n"
                    f"Pauses: [COUNT] incidents, [MINUTES] minutes"
                )
            else:
                extraction_prompt = (
                    f"Extract the following from this machine performance report:\n\n"
                    f"{statement}\n\n"
                    f"Format your answer EXACTLY as:\n"
                    f"Machine: [CODE]\n"
                    f"Breakdowns: [COUNT] incidents\n"
                    f"Downtime: [MINUTES] minutes\n"
                    f"Good units: [COUNT]\n"
                    f"Defective units: [COUNT]\n"
                    f"Pauses: [COUNT] incidents, [MINUTES] minutes"
                )
            
            messages = [
                {"role": "system", "content": "You are a data extraction assistant. Extract only the numerical facts from performance reports. Never add information not present in the text."},
                {"role": "user", "content": extraction_prompt}
            ]

            try:
                output = self.embed_model.getResponse(messages)
                report = output[0]["generated_text"].split("<|im_start|>assistant")[-1].strip()
                # Clean up any remaining template markers
                report = report.replace("<|im_end|>", "").strip()
                reports.append(f"[Relevance: {similarity:.2%}]\n{report}")
            except Exception as e:
                print(f"Error generating report: {e}")
                # Fallback to raw statement
                reports.append(f"[Relevance: {similarity:.2%}]\n{statement}")

        final_report = "\n\n" + "="*80 + "\n\n".join(reports)
        return final_report
    
    def generate_summary(self, type: str, factory_id: str, query: str = None, top_k: int = 5):
        """Generate a comparative summary across multiple records."""
        
        # Get the raw statements
        report = self.generate_report(type, factory_id, top_k, query, use_llm=False)
        
        # Now ask LLM to summarize/compare them
        summary_prompt = [
            {"role": "system", "content": "You are a factory performance analyst. Provide concise insights comparing multiple performance reports."},
            {"role": "user", "content": f"Analyze these {type} performance reports and provide:\n1. Key trends\n2. Best performer\n3. Areas needing attention\n\nReports:\n{report}"}
        ]
        
        output = self.embed_model.getResponse(summary_prompt)
        summary = output[0]["generated_text"].split("<|im_start|>assistant")[-1].strip()
        summary = summary.replace("<|im_end|>", "").strip()
        
        return f"{report}\n\n{'='*80}\n\nSUMMARY & INSIGHTS:\n{summary}"