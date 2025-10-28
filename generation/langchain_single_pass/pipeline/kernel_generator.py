"""
Main kernel generation logic for the NKI kernel generation pipeline.
"""
import traceback
from typing import Optional

from ..core.config import config
from ..core.logging_manager import logging_manager
from ..utils.file_utils import read_file, write_file
from .iteration_manager import iteration_manager


class KernelGenerator:
    """Main kernel generation orchestrator."""
    
    def __init__(self):
        """Initialize the kernel generator."""
        pass
    
    def generate_kernel_with_direct_docs_and_error_loop(self, kernel_func_name: str,
                                                       system_prompt_path: str,
                                                       user_prompt_path: str,
                                                       output_address: str,
                                                       kernel_module_path: str,
                                                       test_func_name: str,
                                                       test_script_output: str,
                                                       reasoning_log_path: str,
                                                       error_doc_path: str,
                                                       docs_dir: str,
                                                       max_iterations: Optional[int] = None) -> bool:
        """
        Generate a NKI kernel using direct function documentation access and iteratively 
        improve it based on error feedback with detailed error documentation.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if max_iterations is None:
            max_iterations = config.max_iterations_per_attempt
        
        try:
            # Set up consolidated iteration log file
            consolidated_log_path = logging_manager.initialize_consolidated_log(
                output_address, kernel_module_path
            )
            
            # Load the initial prompts
            system_prompt = read_file(system_prompt_path)
            user_prompt = read_file(user_prompt_path)
            
            # Generate initial kernel
            kernel_code, selected_functions, function_docs = iteration_manager.generate_initial_kernel(
                kernel_func_name, system_prompt, user_prompt, output_address, kernel_module_path
            )
            
            # Run initial test
            error_message = iteration_manager.run_initial_test(
                test_func_name, kernel_func_name, kernel_module_path, test_script_output
            )
            
            # If no errors in the initial code, we're done
            if error_message and not any(keyword in error_message.lower() for keyword in ["error", "exception", "failed"]):
                print("No errors detected in initial kernel! Kernel generation successful.")
                # Log successful initial generation to the consolidated log
                logging_manager.log_iteration_data(
                    consolidated_log_path,
                    1,
                    "No errors detected",
                    None,
                    None,
                    "Initial generation successful without errors",
                    kernel_code,
                    error_message,
                    None
                )
                return True
            
            # Run the iterative improvement loop
            success = iteration_manager.run_iteration_loop(
                kernel_func_name, system_prompt, user_prompt, output_address,
                kernel_module_path, test_func_name, test_script_output,
                selected_functions, function_docs
            )
            
            return success
            
        except Exception as e:
            error_details = traceback.format_exc()
            print(f"Error in kernel generation pipeline: {e}")
            
            # Save the error
            with open(output_address, "w") as f:
                f.write(f"Error generating kernel: {str(e)}\n\n{error_details}")
            
            return False


# Global kernel generator instance
kernel_generator = KernelGenerator() 