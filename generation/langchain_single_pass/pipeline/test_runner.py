"""
Test execution and validation for the NKI kernel generation pipeline.
"""
from typing import Optional
from torch_xla.core import xla_model as xm

from extraction import run


class TestRunner:
    """Handles test execution and validation."""
    
    def __init__(self):
        """Initialize the test runner."""
        pass
    
    def run_test(self, test_func_name: str, kernel_func_name: str, 
                 kernel_module_path: str, test_script_output: str, 
                 device: Optional[str] = None) -> str:
        """
        Run a test for a generated kernel.
        
        Args:
            test_func_name: Name of the test function to run
            kernel_func_name: Name of the kernel function being tested
            kernel_module_path: Path to the kernel module file
            test_script_output: Path to save test output
            device: Device to run the test on (defaults to XLA device)
            
        Returns:
            str: Error message or success indicator
        """
        if device is None:
            device = xm.xla_device()
        
        try:
            error_message = run(test_func_name, kernel_func_name, kernel_module_path, test_script_output, device)
            return error_message
        except Exception as e:
            return f"Test execution failed: {str(e)}"
    
    def is_test_successful(self, error_message: str) -> bool:
        """Check if the test was successful based on the error message."""
        # Check for common error indicators
        error_indicators = ["Error", "error", "ERROR", "Exception", "exception", "Failed", "failed"]
        return not any(indicator in error_message for indicator in error_indicators)


# Global test runner instance
test_runner = TestRunner() 