"""
Test script to verify the refactored pipeline structure and imports.
"""
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from core.config import config, OperatorConfig
        print("✓ Core config imported successfully")
        
        from core.llm_manager import llm_manager
        print("✓ LLM manager imported successfully")
        
        from core.logging_manager import logging_manager
        print("✓ Logging manager imported successfully")
        
        # Test pipeline imports
        from pipeline.kernel_generator import kernel_generator
        print("✓ Kernel generator imported successfully")
        
        from pipeline.error_handler import error_handler
        print("✓ Error handler imported successfully")
        
        from pipeline.test_runner import test_runner
        print("✓ Test runner imported successfully")
        
        from pipeline.iteration_manager import iteration_manager
        print("✓ Iteration manager imported successfully")
        
        # Test utils imports
        from utils.file_utils import read_file, write_file
        print("✓ File utils imported successfully")
        
        from utils.prompt_utils import extract_kernel_from_llm_response
        print("✓ Prompt utils imported successfully")
        
        from utils.json_utils import parse_json_response
        print("✓ JSON utils imported successfully")
        
        # Test operators imports
        from operators.operator_runner import operator_runner
        print("✓ Operator runner imported successfully")
        
        print("\n✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration functionality."""
    print("\nTesting configuration...")
    
    try:
        from core.config import config, OperatorConfig
        
        # Test basic config
        print(f"✓ Base path: {config.base_path}")
        print(f"✓ LLM model: {config.llm_model_id}")
        print(f"✓ Max retries: {config.max_retries_per_operator}")
        
        # Test operator config
        elementwise_ops = OperatorConfig.get_elementwise_operators()
        print(f"✓ Elementwise operators: {len(elementwise_ops)} operators")
        
        multi_element_ops = OperatorConfig.get_multi_element_operators()
        print(f"✓ Multi-element operators: {len(multi_element_ops)} operators")
        
        product_ops = OperatorConfig.get_product_operators()
        print(f"✓ Product operators: {len(product_ops)} operators")
        
        # Test operator mapping
        mapping = OperatorConfig.get_operator_test_mapping()
        print(f"✓ Operator-test mapping: {len(mapping)} mappings")
        
        print("✓ Configuration test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_utility_functions():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils.json_utils import extract_json_array, parse_json_response
        
        # Test JSON extraction
        test_text = "Some text [\"test1\", \"test2\"] more text"
        extracted = extract_json_array(test_text)
        print(f"✓ JSON extraction: {extracted}")
        
        # Test JSON parsing
        test_response = "[\"test1\", \"test2\"]"
        parsed = parse_json_response(test_response)
        print(f"✓ JSON parsing: {parsed}")
        
        from utils.prompt_utils import extract_reasoning
        
        # Test reasoning extraction
        test_completion = "***This is reasoning***\nSome code here"
        reasoning = extract_reasoning(test_completion)
        print(f"✓ Reasoning extraction: {reasoning}")
        
        print("✓ Utility functions test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing Refactored Pipeline ===\n")
    
    tests = [
        test_imports,
        test_config,
        test_utility_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("✓ All tests passed! The refactored pipeline is ready to use.")
    else:
        print("✗ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main() 