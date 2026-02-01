import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from services.report_generator import ReportGenerator


class TestReportGenerator(unittest.TestCase):
    """Test suite for ReportGenerator service."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_storage = Mock()
        self.mock_embed_model = Mock()
        self.generator = ReportGenerator(self.mock_storage, self.mock_embed_model)

    def test_initialization(self):
        """Test ReportGenerator initialization."""
        self.assertIsNotNone(self.generator.storage)
        self.assertIsNotNone(self.generator.embed_model)
        self.assertIsNotNone(self.generator.encoder)

    def test_generate_report_no_query_raises_error(self):
        """Test that report generation raises error when query is not provided."""
        self.mock_storage.get_knowledge.return_value = []
        
        with self.assertRaises(ValueError) as context:
            self.generator.generate_report(
                type="employee",
                factory_id="factory-123",
                query=None
            )
        self.assertIn("Query must be provided", str(context.exception))

    def test_generate_report_no_records(self):
        """Test report generation with no records found."""
        self.mock_storage.get_knowledge.return_value = []
        
        with patch.object(self.mock_embed_model, 'encode', return_value=np.array([[0.1, 0.2]])):
            result = self.generator.generate_report(
                type="employee",
                factory_id="factory-123",
                query="test query",
                top_k=5
            )
            self.assertEqual(result, "ERROR: No valid vectors found.")

    def test_generate_report_invalid_vectors(self):
        """Test report generation with invalid vectors."""
        records = [
            {"embedding": None, "statement": "test statement 1"},
            {"embedding": "invalid", "statement": "test statement 2"}
        ]
        self.mock_storage.get_knowledge.return_value = records
        
        with patch('services.report_generator.safe_parse_embedding', return_value=None):
            with patch.object(self.mock_embed_model, 'encode', return_value=np.array([[0.1, 0.2]])):
                result = self.generator.generate_report(
                    type="employee",
                    factory_id="factory-123",
                    query="test query",
                    top_k=5
                )
                self.assertEqual(result, "ERROR: No valid vectors found.")

    def test_generate_report_valid_records(self):
        """Test report generation with valid records."""
        records = [
            {
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "statement": "Employee EMP001 breakdown report"
            },
            {
                "embedding": [0.2, 0.3, 0.4, 0.5],
                "statement": "Employee EMP002 breakdown report"
            },
            {
                "embedding": [0.3, 0.4, 0.5, 0.6],
                "statement": "Employee EMP003 breakdown report"
            }
        ]
        self.mock_storage.get_knowledge.return_value = records
        
        # Mock the embedding model responses
        query_vec = np.array([[0.1, 0.2, 0.3, 0.4]])
        self.mock_embed_model.encode.return_value = query_vec
        
        # Mock similarity calculation
        sim = np.array([0.3, 0.5, 0.8])
        self.mock_embed_model.similarity.return_value = sim
        
        # Mock LLM response
        self.mock_embed_model.getResponse.return_value = [
            {"generated_text": "<|im_start|>assistant\nReport content here"}
        ]
        
        with patch('services.report_generator.safe_parse_embedding') as mock_parse:
            mock_parse.side_effect = lambda x: x if isinstance(x, list) else None
            
            result = self.generator.generate_report(
                type="employee",
                factory_id="factory-123",
                query="which employee has most breakdowns?",
                top_k=2
            )
            
            self.assertIsNotNone(result)
            self.assertIn("Report content here", result)
            self.mock_embed_model.getResponse.assert_called()

    def test_generate_report_top_k_parameter(self):
        """Test that top_k parameter limits results correctly."""
        records = [
            {"embedding": [0.1 * i, 0.2 * i], "statement": f"Statement {i}"}
            for i in range(1, 6)
        ]
        self.mock_storage.get_knowledge.return_value = records
        
        query_vec = np.array([[0.1, 0.2]])
        self.mock_embed_model.encode.return_value = query_vec
        
        # Create similarity scores
        sim = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        self.mock_embed_model.similarity.return_value = sim
        
        self.mock_embed_model.getResponse.return_value = [
            {"generated_text": "<|im_start|>assistant\nReport"}
        ]
        
        with patch('services.report_generator.safe_parse_embedding') as mock_parse:
            mock_parse.side_effect = lambda x: x if isinstance(x, list) else None
            
            self.generator.generate_report(
                type="employee",
                factory_id="factory-123",
                query="test query",
                top_k=2
            )
            
            # Should call getResponse 2 times (top_k=2)
            self.assertEqual(self.mock_embed_model.getResponse.call_count, 2)

    def test_generate_report_system_prompt_format(self):
        """Test that system prompt is correctly formatted."""
        records = [
            {
                "embedding": [0.1, 0.2],
                "statement": "Test statement with employee code EMP001"
            }
        ]
        self.mock_storage.get_knowledge.return_value = records
        
        query_vec = np.array([[0.1, 0.2]])
        self.mock_embed_model.encode.return_value = query_vec
        
        sim = np.array([0.9])
        self.mock_embed_model.similarity.return_value = sim
        
        self.mock_embed_model.getResponse.return_value = [
            {"generated_text": "<|im_start|>assistant\nFinal report"}
        ]
        
        with patch('services.report_generator.safe_parse_embedding') as mock_parse:
            mock_parse.side_effect = lambda x: x if isinstance(x, list) else None
            
            self.generator.generate_report(
                type="employee",
                factory_id="factory-123",
                query="test query",
                top_k=1
            )
            
            # Verify the message structure
            call_args = self.mock_embed_model.getResponse.call_args
            messages = call_args[0][0]
            
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[1]["role"], "user")
            self.assertIn("LOG TO ANALYZE", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
