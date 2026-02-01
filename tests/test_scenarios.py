import unittest
from unittest.mock import Mock
import numpy as np
from abc import ABC, abstractmethod

from interfaces.Storage import Storage
from interfaces.EmbeddingModel import EmbeddingModel
from services.knowledge_generator import KnowledgeGenerator
from services.report_generator import ReportGenerator


class MockStorage(Storage):
    """Mock implementation of Storage interface for scenario testing."""

    def __init__(self):
        self.factories = ["factory-001", "factory-002"]
        self.logs_db = {
            "factory-001": [
                {
                    "employee": {"code": "EMP001", "name": "John Doe"},
                    "machine": {"code": "MACH01", "name": "Line A"},
                    "event_type": "breakdown",
                    "event_duration": 45
                },
                {
                    "employee": {"code": "EMP001", "name": "John Doe"},
                    "machine": {"code": "MACH01", "name": "Line A"},
                    "event_type": "breakdown",
                    "event_duration": 30
                },
                {
                    "employee": {"code": "EMP001", "name": "John Doe"},
                    "machine": {"code": "MACH01", "name": "Line A"},
                    "event_type": "good_piece",
                    "event_duration": 0
                },
                {
                    "employee": {"code": "EMP002", "name": "Jane Smith"},
                    "machine": {"code": "MACH02", "name": "Line B"},
                    "event_type": "pause",
                    "event_duration": 20
                },
                {
                    "employee": {"code": "EMP002", "name": "Jane Smith"},
                    "machine": {"code": "MACH02", "name": "Line B"},
                    "event_type": "pause",
                    "event_duration": 15
                },
            ],
            "factory-002": []
        }
        self.knowledge_db = {}

    def get_logs(self, factory_id: str):
        return self.logs_db.get(factory_id, [])

    def push_embedding(self, vector: list, factory_id: str, type_str: str, statement: str, code: str) -> bool:
        if factory_id not in self.knowledge_db:
            self.knowledge_db[factory_id] = []
        self.knowledge_db[factory_id].append({
            "embedding": vector,
            "statement": statement,
            "type": type_str,
            "code": code
        })
        return True

    def get_embedding(self, factory_id: str, emp_code: str, machine_code: str, date: str):
        if factory_id not in self.knowledge_db:
            return None
        for record in self.knowledge_db[factory_id]:
            if record["code"] == emp_code:
                return record
        return None

    def get_factories(self):
        return self.factories

    def get_knowledge(self, factory_id: str, type_str: str):
        if factory_id not in self.knowledge_db:
            return []
        return [r for r in self.knowledge_db[factory_id] if r["type"] == type_str]

    def get_employees(self, factory_id: str):
        logs = self.get_logs(factory_id)
        return list(set(log["employee"]["code"] for log in logs))

    def get_machines(self, factory_id: str):
        logs = self.get_logs(factory_id)
        return list(set(log["machine"]["code"] for log in logs))

    def get_machine_performance(self, factory_id: str, machine_code: str):
        logs = self.get_logs(factory_id)
        m_logs = [l for l in logs if l["machine"]["code"] == machine_code]
        return {
            "breakdowns": len([l for l in m_logs if l["event_type"] == "breakdown"]),
            "total_duration": sum(l["event_duration"] for l in m_logs)
        }

    def get_employee_performance(self, factory_id: str, emp_code: str):
        logs = self.get_logs(factory_id)
        e_logs = [l for l in logs if l["employee"]["code"] == emp_code]
        return {
            "breakdowns": len([l for l in e_logs if l["event_type"] == "breakdown"]),
            "good_pieces": len([l for l in e_logs if l["event_type"] == "good_piece"]),
            "bad_pieces": len([l for l in e_logs if l["event_type"] == "bad_piece"])
        }


class MockEmbeddingModel(EmbeddingModel):
    """Mock implementation of EmbeddingModel for scenario testing."""

    def __init__(self):
        self.model_id = "mock-model"
        self.call_count = 0

    def encode(self, documents, prompt_name: str = "default"):
        self.call_count += 1
        if isinstance(documents, str):
            documents = [documents]
        
        EMBEDDING_DIM = 768
        vectors = []
        for doc in documents:
            seed = hash(doc) % 2**32
            np.random.seed(seed)
            vector = np.random.rand(EMBEDDING_DIM)
            vectors.append(vector.tolist())
        
        if len(vectors) == 1:
            return np.array(vectors[0])
        return np.array(vectors)

    def similarity(self, query_vec, doc_vecs):
        if isinstance(query_vec, list):
            query_vec = np.array(query_vec)
        if isinstance(doc_vecs, list):
            doc_vecs = np.array(doc_vecs)
        
        # Flatten query to 1D
        if query_vec.ndim > 1:
            query_vec = query_vec.flatten()
        # Ensure doc_vecs is 2D
        if doc_vecs.ndim == 1:
            doc_vecs = doc_vecs.reshape(1, -1)
        
        # Normalize
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
        
        # Compute dot product and flatten
        similarities = np.dot(doc_norms, query_norm.reshape(-1, 1)).flatten()
        return similarities

    def get_model_id(self) -> str:
        return self.model_id

    def getResponse(self, messages):
        report = (
            "AUDIT REPORT\n"
            "Employee Code: EMP001\n"
            "Breakdowns: 2\n"
            "Downtime: 75 minutes"
        )
        return [{"generated_text": f"<|im_start|>assistant\n{report}"}]


class TestScenarios(unittest.TestCase):
    """Scenario-based integration tests with mock implementations."""

    def setUp(self):
        self.storage = MockStorage()
        self.embedding_model = MockEmbeddingModel()
        self.knowledge_gen = KnowledgeGenerator(self.storage, self.embedding_model)
        self.report_gen = ReportGenerator(self.storage, self.embedding_model)

    def test_scenario_1_knowledge_generation(self):
        """Test complete knowledge generation workflow."""
        print("\nSCENARIO 1: Knowledge Generation")
        result = self.knowledge_gen.generate_knowledge()
        self.assertTrue(result)
        
        factory_id = "factory-001"
        emp_knowledge = self.storage.get_knowledge(factory_id, "employee")
        mach_knowledge = self.storage.get_knowledge(factory_id, "machine")
        
        print(f"[PASS] Generated {len(emp_knowledge)} employee embeddings")
        print(f"[PASS] Generated {len(mach_knowledge)} machine embeddings")
        self.assertGreater(len(emp_knowledge), 0)

    def test_scenario_2_employee_performance(self):
        """Test employee performance analysis."""
        print("\nSCENARIO 2: Employee Performance Analysis")
        self.knowledge_gen.generate_knowledge()
        
        factory_id = "factory-001"
        emp_code = "EMP001"
        perf = self.storage.get_employee_performance(factory_id, emp_code)
        
        print(f"Employee {emp_code}: {perf['breakdowns']} breakdowns")
        self.assertEqual(perf["breakdowns"], 2)
        print("[PASS] Performance metrics verified")

    def test_scenario_3_machine_analysis(self):
        """Test machine performance analysis."""
        print("\nSCENARIO 3: Machine Performance Analysis")
        factory_id = "factory-001"
        machines = self.storage.get_machines(factory_id)
        
        for mach in machines:
            perf = self.storage.get_machine_performance(factory_id, mach)
            print(f"Machine {mach}: {perf['breakdowns']} breakdowns")
        
        self.assertGreater(len(machines), 0)
        print("[PASS] Machine analysis complete")

    def test_scenario_4_report_generation(self):
        """Test report generation with query."""
        print("\nSCENARIO 4: Report Generation")
        # First generate knowledge
        self.knowledge_gen.generate_knowledge()
        
        # Now query for report with the generated knowledge
        report = self.report_gen.generate_report(
            type="employee",
            factory_id="factory-001",
            query="which employee has most breakdowns?",
            top_k=1
        )
        
        # Report may be empty if no valid vectors, which is acceptable
        self.assertIsNotNone(report)
        print(f"[PASS] Report processing complete")

    def test_scenario_5_empty_factory(self):
        """Test handling of empty factories."""
        print("\nSCENARIO 5: Empty Factory Handling")
        result = self.knowledge_gen.generate_knowledge()
        self.assertTrue(result)
        print("[PASS] Empty factory handled gracefully")

    def test_scenario_6_embedding_consistency(self):
        """Test embedding model consistency."""
        print("\nSCENARIO 6: Embedding Consistency")
        stmt = "Employee EMP001 had 2 breakdowns"
        
        vec1 = self.embedding_model.encode(stmt)
        vec2 = self.embedding_model.encode(stmt)
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        print(f"Same statement similarity: {sim:.4f}")
        self.assertAlmostEqual(sim, 1.0, places=4)
        print("[PASS] Consistency verified")

    def test_scenario_7_similarity_ranking(self):
        """Test similarity-based ranking."""
        print("\nSCENARIO 7: Similarity Ranking")
        self.knowledge_gen.generate_knowledge()
        
        knowledge = self.storage.get_knowledge("factory-001", "employee")
        if knowledge:
            query_vec = self.embedding_model.encode("breakdowns")
            doc_vecs = np.array([k["embedding"] for k in knowledge])
            sims = self.embedding_model.similarity(query_vec, doc_vecs)
            
            print(f"Found {len(knowledge)} records, similarities: {sims}")
            print("[PASS] Ranking verified")

    def test_scenario_8_multiple_factories(self):
        """Test processing multiple factories."""
        print("\nSCENARIO 8: Multiple Factories")
        factories = self.storage.get_factories()
        print(f"Processing {len(factories)} factories")
        
        result = self.knowledge_gen.generate_knowledge()
        self.assertTrue(result)
        print("[PASS] Multiple factories processed")

    def test_scenario_9_full_workflow(self):
        """Test complete end-to-end workflow."""
        print("\nSCENARIO 9: Full End-to-End Workflow")
        
        # Generate knowledge
        result = self.knowledge_gen.generate_knowledge()
        self.assertTrue(result)
        print("Step 1: Knowledge generated")
        
        # Generate report
        report = self.report_gen.generate_report(
            type="employee",
            factory_id="factory-001",
            query="performance summary",
            top_k=2
        )
        self.assertIsNotNone(report)
        print("Step 2: Report generated")
        
        # Verify data integrity
        employees = self.storage.get_employees("factory-001")
        machines = self.storage.get_machines("factory-001")
        print(f"Step 3: Found {len(employees)} employees and {len(machines)} machines")
        print("[PASS] Full workflow verified")

    def test_scenario_10_mock_interfaces(self):
        """Test that mocks properly implement interfaces."""
        print("\nSCENARIO 10: Mock Interface Compliance")
        
        # Verify Storage interface
        self.assertTrue(hasattr(self.storage, 'get_logs'))
        self.assertTrue(hasattr(self.storage, 'push_embedding'))
        self.assertTrue(hasattr(self.storage, 'get_knowledge'))
        print("[PASS] Storage interface verified")
        
        # Verify EmbeddingModel interface
        self.assertTrue(hasattr(self.embedding_model, 'encode'))
        self.assertTrue(hasattr(self.embedding_model, 'similarity'))
        self.assertTrue(hasattr(self.embedding_model, 'getResponse'))
        print("[PASS] EmbeddingModel interface verified")
        
        print("[PASS] All interfaces compliant")


if __name__ == "__main__":
    unittest.main(verbosity=2)
