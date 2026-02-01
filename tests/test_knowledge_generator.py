import unittest
from unittest.mock import Mock, MagicMock, patch
from services.knowledge_generator import KnowledgeGenerator


class TestKnowledgeGenerator(unittest.TestCase):
    """Test suite for KnowledgeGenerator service."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_storage = Mock()
        self.mock_embed_model = Mock()
        self.generator = KnowledgeGenerator(self.mock_storage, self.mock_embed_model)

    def test_initialization(self):
        """Test KnowledgeGenerator initialization."""
        self.assertIsNotNone(self.generator.storage)
        self.assertIsNotNone(self.generator.embed_model)
        self.assertIsNotNone(self.generator.encoder)

    def test_generate_knowledge_no_factories(self):
        """Test generate_knowledge with no factories."""
        self.mock_storage.get_factories.return_value = []
        result = self.generator.generate_knowledge()
        self.assertTrue(result)
        self.mock_storage.get_factories.assert_called_once()

    def test_generate_knowledge_with_empty_logs(self):
        """Test generate_knowledge when factory has no logs."""
        factory_id = "factory-123"
        self.mock_storage.get_factories.return_value = [factory_id]
        self.mock_storage.get_logs.return_value = []

        result = self.generator.generate_knowledge()
        self.assertTrue(result)
        self.mock_storage.get_logs.assert_called_with(factory_id)

    def test_generate_knowledge_with_logs(self):
        """Test generate_knowledge with actual logs."""
        factory_id = "factory-123"
        logs = [
            {
                "employee": {"code": "EMP001"},
                "machine": {"code": "MACH01"},
                "event_type": "breakdown",
                "event_duration": 30
            },
            {
                "employee": {"code": "EMP001"},
                "machine": {"code": "MACH01"},
                "event_type": "good_piece",
                "event_duration": 0
            },
            {
                "employee": {"code": "EMP002"},
                "machine": {"code": "MACH02"},
                "event_type": "pause",
                "event_duration": 15
            }
        ]

        self.mock_storage.get_factories.return_value = [factory_id]
        self.mock_storage.get_logs.return_value = logs
        self.generator.encoder.encode_data = Mock(return_value=None)

        with patch('services.knowledge_generator.generate_employee_statement') as mock_emp_stmt, \
             patch('services.knowledge_generator.generate_machine_statement') as mock_mach_stmt, \
             patch('services.knowledge_generator.get_single_employee_deeds') as mock_deeds, \
             patch('services.knowledge_generator.get_average_comparison') as mock_avg_comp:
            
            mock_emp_stmt.return_value = "employee statement"
            mock_mach_stmt.return_value = "machine statement"
            mock_deeds.return_value = {"breakdowns": 1, "good_pieces": 1}
            mock_avg_comp.return_value = {"avg_breakdowns": 0.5}

            result = self.generator.generate_knowledge()
            self.assertTrue(result)

    def test_generate_knowledge_encode_error_handling(self):
        """Test error handling during encoding."""
        factory_id = "factory-123"
        logs = [
            {
                "employee": {"code": "EMP001"},
                "machine": {"code": "MACH01"},
                "event_type": "breakdown",
                "event_duration": 30
            }
        ]

        self.mock_storage.get_factories.return_value = [factory_id]
        self.mock_storage.get_logs.return_value = logs
        self.generator.encoder.encode_data = Mock(side_effect=Exception("Encoding error"))

        with patch('services.knowledge_generator.generate_employee_statement') as mock_emp_stmt, \
             patch('services.knowledge_generator.generate_machine_statement') as mock_mach_stmt, \
             patch('services.knowledge_generator.get_single_employee_deeds') as mock_deeds, \
             patch('services.knowledge_generator.get_average_comparison') as mock_avg_comp:
            
            mock_emp_stmt.return_value = "employee statement"
            mock_mach_stmt.return_value = "machine statement"
            mock_deeds.return_value = {"breakdowns": 1}
            mock_avg_comp.return_value = {"avg_breakdowns": 0.5}

            result = self.generator.generate_knowledge()
            self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
