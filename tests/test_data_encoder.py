import unittest
from unittest.mock import Mock, patch
from services.data_encoder import DataEncoder


class TestDataEncoder(unittest.TestCase):
    """Test suite for DataEncoder service."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_storage = Mock()
        self.mock_embed_model = Mock()
        self.encoder = DataEncoder(self.mock_storage, self.mock_embed_model)

    def test_initialization(self):
        """Test DataEncoder initialization."""
        self.assertIsNotNone(self.encoder.storage)
        self.assertIsNotNone(self.encoder.embed_model)

    def test_encode_data_calls_storage_push_embedding(self):
        """Test that encode_data calls storage push_embedding."""
        statement = "Test employee statement"
        factory_id = "factory-123"
        type_str = "employee"
        code = "EMP001"
        
        # Mock the embed_model to return a vector
        mock_vector = [0.1, 0.2, 0.3, 0.4]
        self.mock_embed_model.encode.return_value = mock_vector
        
        self.encoder.encode_data(statement, type_str, factory_id, code)
        
        # Verify storage.push_embedding was called
        self.mock_storage.push_embedding.assert_called_once()
        call_args = self.mock_storage.push_embedding.call_args
        
        # Verify the arguments
        self.assertEqual(call_args[0][1], factory_id)  # factory_id
        self.assertEqual(call_args[0][2], type_str)     # type
        self.assertEqual(call_args[0][3], statement)    # statement
        self.assertEqual(call_args[0][4], code)         # code

    def test_encode_data_with_machine_type(self):
        """Test encode_data with machine type."""
        statement = "Machine MACH01 performance data"
        factory_id = "factory-456"
        type_str = "machine"
        code = "MACH01"
        
        mock_vector = [0.5, 0.6, 0.7, 0.8]
        self.mock_embed_model.encode.return_value = mock_vector
        
        self.encoder.encode_data(statement, type_str, factory_id, code)
        
        self.mock_storage.push_embedding.assert_called_once()
        call_args = self.mock_storage.push_embedding.call_args
        self.assertEqual(call_args[0][2], "machine")


if __name__ == "__main__":
    unittest.main()
