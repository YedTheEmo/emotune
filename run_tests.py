#!/usr/bin/env python3
"""
Test runner for EmoTune - runs tests from their organized locations
"""

import sys
import os
import subprocess
import argparse

def run_test_file(test_path):
    """Run a specific test file"""
    print(f"Running: {test_path}")
    result = subprocess.run([sys.executable, test_path], capture_output=False)
    return result.returncode == 0

def run_all_tests():
    """Run all tests in organized directories"""
    test_dirs = [
        "tests/integration",
        "tests/unit", 
        "tests/pyo"
    ]
    
    all_passed = True
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"\n{'='*50}")
            print(f"Running tests in: {test_dir}")
            print(f"{'='*50}")
            
            for file in os.listdir(test_dir):
                if file.endswith('.py') and file.startswith('test_'):
                    test_path = os.path.join(test_dir, file)
                    success = run_test_file(test_path)
                    if not success:
                        all_passed = False
                        print(f"❌ {file} FAILED")
                    else:
                        print(f"✅ {file} PASSED")
    
    return all_passed

def main():
    parser = argparse.ArgumentParser(description='Run EmoTune tests')
    parser.add_argument('--test', help='Run specific test file')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if args.test:
        # Run specific test
        if os.path.exists(args.test):
            success = run_test_file(args.test)
            sys.exit(0 if success else 1)
        else:
            print(f"Test file not found: {args.test}")
            sys.exit(1)
    elif args.all:
        # Run all tests
        success = run_all_tests()
        if success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ Some tests failed!")
        sys.exit(0 if success else 1)
    else:
        # Default: run integration tests
        print("Running integration tests...")
        success = run_all_tests()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 