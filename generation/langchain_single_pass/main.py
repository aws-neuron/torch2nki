"""
Main entry point for the NKI kernel generation pipeline.
"""
import sys
import os

# Add the current directory to the Python path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from operators.operator_runner import operator_runner
from core.config import OperatorConfig


def main():
    """Main entry point for the kernel generation pipeline."""
    print("=== NKI Kernel Generation Pipeline ===")
    
    # For now, run the product operators (which currently only includes "sort")
    # This matches the original behavior in all_in_one_generator_new.py
    product_operators = OperatorConfig.get_product_operators()
    product_test_names = OperatorConfig.get_product_test_names()
    
    print(f"Running {len(product_operators)} product operators: {', '.join(product_operators)}")
    
    # Run the operators
    results = operator_runner.run_operators(product_operators, product_test_names)
    
    # Print summary
    print("\n=== Generation Summary ===")
    successful = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"Successful: {successful}/{total}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    for operator, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {operator}: {status}")
    
    from core.config import config
    print(f"\nResults saved to: {config.tests_passed_dict_path}")


if __name__ == "__main__":
    main() 