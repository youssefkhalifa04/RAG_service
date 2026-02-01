import unittest
from utils.helpers import percentage, get_comparison, safe_parse_embedding
import json


class TestHelperFunctions(unittest.TestCase):
    """Test suite for helper utility functions."""

    def test_percentage_basic(self):
        """Test basic percentage calculation."""
        self.assertEqual(percentage(50, 100), 50.0)
        self.assertEqual(percentage(25, 100), 25.0)
        self.assertEqual(percentage(1, 2), 50.0)

    def test_percentage_zero_whole(self):
        """Test percentage when whole is zero."""
        self.assertEqual(percentage(10, 0), 0.0)
        self.assertEqual(percentage(0, 0), 0.0)

    def test_percentage_decimal(self):
        """Test percentage with decimal results."""
        result = percentage(1, 3)
        self.assertAlmostEqual(result, 33.33333, places=4)

    def test_get_comparison_above_average(self):
        """Test comparison when value is above average."""
        self.assertEqual(get_comparison(100, 50), "above")

    def test_get_comparison_below_average(self):
        """Test comparison when value is below average."""
        self.assertEqual(get_comparison(25, 100), "below")

    def test_get_comparison_at_average(self):
        """Test comparison when value equals average."""
        self.assertEqual(get_comparison(100, 100), "at")

    def test_get_comparison_zero_average(self):
        """Test comparison when average is zero."""
        self.assertEqual(get_comparison(10, 0), "at")

    def test_safe_parse_embedding_list(self):
        """Test parsing embedding that is already a list."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        result = safe_parse_embedding(embedding)
        self.assertEqual(result, embedding)

    def test_safe_parse_embedding_string_brackets(self):
        """Test parsing embedding from string with brackets."""
        embedding_str = "[0.1, 0.2, 0.3, 0.4]"
        result = safe_parse_embedding(embedding_str)
        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4])

    def test_safe_parse_embedding_postgres_format(self):
        """Test parsing embedding from Postgres bracket format."""
        embedding_str = "{0.1, 0.2, 0.3, 0.4}"
        result = safe_parse_embedding(embedding_str)
        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4])

    def test_safe_parse_embedding_invalid_string(self):
        """Test parsing invalid embedding string."""
        embedding_str = "invalid"
        result = safe_parse_embedding(embedding_str)
        self.assertIsNone(result)

    def test_safe_parse_embedding_invalid_type(self):
        """Test parsing embedding with invalid type."""
        result = safe_parse_embedding(12345)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
