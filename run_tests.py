#!/usr/bin/env python
"""
Quick Test Runner - Summarizer Project
Run this script to execute all tests with a summary
"""

import subprocess
import sys

def run_tests():
    """Run all unit tests and display results."""
    print("=" * 70)
    print("Running Unit Tests for Summarizer Project")
    print("=" * 70)
    print()
    
    try:
        # Run the test discovery
        result = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
            capture_output=True,
            text=True
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        # Check result
        if result.returncode == 0:
            print()
            print("=" * 70)
            print("✅ ALL TESTS PASSED!")
            print("=" * 70)
            return 0
        else:
            print()
            print("=" * 70)
            print("❌ SOME TESTS FAILED")
            print("=" * 70)
            return 1
            
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
