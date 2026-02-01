import unittest
from unittest.mock import Mock, patch
from utils.employee import (
    get_single_employee_deeds,
    get_single_employee_deeds_compare_avg,
    get_average_comparison,
    generate_employee_statement
)


class TestEmployeeUtils(unittest.TestCase):
    """Test suite for employee utility functions."""

    def setUp(self):
        """Set up test data."""
        self.sample_logs = [
            {
                "employee": {"code": "EMP001"},
                "machine": {"code": "MACH01"},
                "event_type": "breakdown",
                "event_duration": 30
            },
            {
                "employee": {"code": "EMP001"},
                "machine": {"code": "MACH01"},
                "event_type": "breakdown",
                "event_duration": 20
            },
            {
                "employee": {"code": "EMP001"},
                "machine": {"code": "MACH01"},
                "event_type": "good_piece",
                "event_duration": 0
            },
            {
                "employee": {"code": "EMP001"},
                "machine": {"code": "MACH01"},
                "event_type": "good_piece",
                "event_duration": 0
            },
            {
                "employee": {"code": "EMP001"},
                "machine": {"code": "MACH01"},
                "event_type": "bad_piece",
                "event_duration": 0
            },
            {
                "employee": {"code": "EMP002"},
                "machine": {"code": "MACH02"},
                "event_type": "pause",
                "event_duration": 15
            },
            {
                "employee": {"code": "EMP002"},
                "machine": {"code": "MACH02"},
                "event_type": "pause",
                "event_duration": 10
            }
        ]

    def test_get_single_employee_deeds_breakdown_count(self):
        """Test counting breakdowns for an employee."""
        deeds = get_single_employee_deeds("EMP001", self.sample_logs)
        self.assertEqual(deeds["breakdowns"], 2)

    def test_get_single_employee_deeds_breakdown_duration(self):
        """Test counting breakdown duration for an employee."""
        deeds = get_single_employee_deeds("EMP001", self.sample_logs)
        self.assertEqual(deeds["breakdowns_duration"], 50)

    def test_get_single_employee_deeds_good_pieces(self):
        """Test counting good pieces for an employee."""
        deeds = get_single_employee_deeds("EMP001", self.sample_logs)
        self.assertEqual(deeds["good_pieces"], 2)

    def test_get_single_employee_deeds_bad_pieces(self):
        """Test counting bad pieces for an employee."""
        deeds = get_single_employee_deeds("EMP001", self.sample_logs)
        self.assertEqual(deeds["bad_pieces"], 1)

    def test_get_single_employee_deeds_pause_count(self):
        """Test counting pauses for an employee."""
        deeds = get_single_employee_deeds("EMP002", self.sample_logs)
        self.assertEqual(deeds["pauses_count"], 2)

    def test_get_single_employee_deeds_pause_duration(self):
        """Test counting pause duration for an employee."""
        deeds = get_single_employee_deeds("EMP002", self.sample_logs)
        self.assertEqual(deeds["pauses"], 25)

    def test_get_single_employee_deeds_nonexistent_employee(self):
        """Test getting deeds for non-existent employee."""
        deeds = get_single_employee_deeds("EMP999", self.sample_logs)
        self.assertEqual(deeds["breakdowns"], 0)
        self.assertEqual(deeds["good_pieces"], 0)
        self.assertEqual(deeds["pauses"], 0)

    def test_get_average_comparison_no_employees(self):
        """Test average comparison with no employees."""
        overall = {"total_breakdowns": 0, "total_breakdowns_duration": 0,
                   "total_good_pieces": 0, "total_bad_pieces": 0,
                   "total_pauses": 0, "total_pauses_count": 0}
        avg_comp = get_average_comparison(overall, [])
        self.assertEqual(avg_comp["avg_breakdowns"], 0)
        self.assertEqual(avg_comp["avg_good_pieces"], 0)

    def test_get_average_comparison_multiple_employees(self):
        """Test average comparison with multiple employees."""
        overall = {
            "total_breakdowns": 2,
            "total_breakdowns_duration": 50,
            "total_good_pieces": 2,
            "total_bad_pieces": 1,
            "total_pauses": 25,
            "total_pauses_count": 2
        }
        avg_comp = get_average_comparison(overall, self.sample_logs)
        self.assertEqual(avg_comp["avg_breakdowns"], 1)
        self.assertEqual(avg_comp["avg_breakdowns_duration"], 25)
        self.assertEqual(avg_comp["avg_good_pieces"], 1)

    def test_generate_employee_statement_structure(self):
        """Test that employee statement is generated with expected content."""
        deeds = {
            "breakdowns": 2,
            "breakdowns_duration": 50,
            "pauses": 25,
            "pauses_count": 2,
            "good_pieces": 2,
            "bad_pieces": 1
        }
        overall = {
            "total_breakdowns": 2,
            "total_breakdowns_duration": 50,
            "total_good_pieces": 2,
            "total_bad_pieces": 1,
            "total_pauses": 25,
            "total_pauses_count": 2
        }
        avg_comp = {
            "avg_breakdowns": 1,
            "avg_breakdowns_duration": 25,
            "avg_good_pieces": 1,
            "avg_bad_pieces": 0.5,
            "avg_pauses": 12.5,
            "avg_pauses_count": 1
        }

        with patch('utils.employee.get_comparison', return_value='above'):
            statement = generate_employee_statement("EMP001", "MACH01", deeds, overall, avg_comp)
            self.assertIn("EMP001", statement)
            self.assertIn("MACH01", statement)
            self.assertIn("Performance Report", statement)
            self.assertIn("breakdowns", statement)

    def test_get_single_employee_deeds_compare_avg(self):
        """Test comparing employee deeds against average."""
        overall = {
            "total_breakdowns": 4,
            "total_breakdowns_duration": 100,
            "total_good_pieces": 4,
            "total_bad_pieces": 2
        }
        deeds = get_single_employee_deeds_compare_avg("EMP001", self.sample_logs, overall)
        self.assertIn("breakdowns_vs_avg", deeds)
        self.assertIn("good_pieces_vs_avg", deeds)


if __name__ == "__main__":
    unittest.main()
