"""
Iteration management for the NKI kernel generation pipeline.
Handles the iterative improvement loop for kernel generation.
"""
from typing import List, Dict, Any, Optional, Tuple
import traceback

from ..core.config import config
from ..core.llm_manager import llm_manager
from ..core.logging_manager import logging_manager
from ..utils.file_utils import read_file, write_file
from ..utils.prompt_utils import (
    extract_kernel_from_llm_response, extract_reasoning, update_function_name_in_text,
    create_initial_generation_prompt, create_enhanced_error_prompt_template,
    create_additional_functions_prompt
)
from ..utils.json_utils import parse_json_response
from .error_handler import error_handler
from .test_runner import test_runner
from doc_grabber import get_available_functions, select_relevant_functions, load_function_documentation


class IterationManager:
    """Manages the iterative improvement loop for kernel generation."""
    
    def __init__(self):
        """Initialize the iteration manager."""
        pass
    
    def generate_initial_kernel(self, kernel_func_name: str, system_prompt: str, 
                               user_prompt: str, output_address: str, 
                               kernel_module_path: str) -> Tuple[str, List[str], str]:
        """
        Generate the initial kernel with function documentation.
        
        Returns:
            Tuple of (kernel_code, selected_functions, function_docs)
        """
        # Select relevant functions
        query_llm = None  # We'll use the direct LLM manager instead
        selected_functions = select_relevant_functions(
            query_llm,
            user_prompt,
            get_available_functions(config.docs_dir)
        )
        
        function_docs = load_function_documentation(config.docs_dir, selected_functions)
        
        # Initial kernel generation with function documentation
        initial_generation_prompt = create_initial_generation_prompt(
            system_prompt, user_prompt, function_docs
        )
        
        # Log the full prompt being sent to the LLM
        prompt_path = output_address + ".prompt_path.txt"
        logging_manager.log_prompt(prompt_path, initial_generation_prompt, append=True)
        
        try:
            # Use direct API call with retry logic
            initial_generation = llm_manager.invoke_with_retry(initial_generation_prompt, temperature=0.85)
        except Exception as e:
            print(f"Error in initial kernel generation: {e}")
            initial_generation = f"Error occurred: {str(e)}"
        
        # Save raw output
        write_file(output_address, initial_generation)
        
        # Extract the kernel code
        try:
            kernel_code = extract_kernel_from_llm_response(initial_generation)
            kernel_code = update_function_name_in_text(kernel_code, kernel_func_name)
            write_file(kernel_module_path, kernel_code)
        except ValueError as e:
            error_msg = f"Error extracting kernel code: {e}"
            print(error_msg)
            raise
        
        return kernel_code, selected_functions, function_docs
    
    def run_initial_test(self, test_func_name: str, kernel_func_name: str, 
                        kernel_module_path: str, test_script_output: str) -> str:
        """Run the initial test on the generated kernel."""
        error_message = test_runner.run_test(
            test_func_name, kernel_func_name, kernel_module_path, test_script_output
        )
        return error_message
    
    def select_additional_functions(self, current_functions: List[str], 
                                  error_message: str) -> List[str]:
        """Select additional functions based on error message."""
        additional_functions_prompt = create_additional_functions_prompt(
            current_functions, error_message, get_available_functions(config.docs_dir)
        )
        
        try:
            additional_response = llm_manager.invoke_with_retry(additional_functions_prompt, temperature=0.3)
        except Exception as e:
            print(f"Error in additional function selection: {e}")
            return []
        
        try:
            additional_functions = parse_json_response(additional_response)
            
            # Only include valid functions that weren't already selected
            available_functions = get_available_functions(config.docs_dir)
            new_functions = [f for f in additional_functions 
                           if f in available_functions and f not in current_functions]
            
            if new_functions:
                print(f"Adding additional functions: {', '.join(new_functions)}")
            
            return new_functions
            
        except Exception as e:
            print(f"Error parsing additional functions: {e}")
            return []
    
    def generate_improved_kernel(self, system_prompt: str, user_prompt: str,
                                iteration_history: str, previous_error_message: str,
                                function_docs: str, output_address: str,
                                kernel_module_path: str, kernel_func_name: str) -> Tuple[str, str]:
        """
        Generate an improved kernel based on error feedback.
        
        Returns:
            Tuple of (kernel_code, reasoning_text)
        """
        # Create enhanced error re-injection prompt
        enhanced_error_prompt_template = create_enhanced_error_prompt_template()
        
        # Format the enhanced error prompt
        enhanced_error_prompt = enhanced_error_prompt_template.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            iteration_history=iteration_history,
            previous_error_message=previous_error_message,
            function_docs=function_docs
        )
        
        # Log the full error prompt being sent to the LLM
        prompt_path = output_address + ".prompt_path.txt"
        logging_manager.log_prompt(prompt_path, enhanced_error_prompt, append=False)
        
        try:
            # Use direct API call with retry logic
            improved_generation = llm_manager.invoke_with_retry(enhanced_error_prompt, temperature=0.85)
        except Exception as e:
            improved_generation = f"Error occurred: {str(e)}"
        
        # Save the raw output
        write_file(output_address, improved_generation)
        
        # Extract reasoning and kernel code
        reasoning_text = extract_reasoning(improved_generation)
        
        try:
            kernel_code = extract_kernel_from_llm_response(improved_generation)
            kernel_code = update_function_name_in_text(kernel_code, kernel_func_name)
            write_file(kernel_module_path, kernel_code)
        except ValueError as e:
            error_msg = f"Error extracting kernel code: {e}"
            print(error_msg)
            raise
        
        return kernel_code, reasoning_text
    
    def run_iteration_loop(self, kernel_func_name: str, system_prompt: str, user_prompt: str,
                          output_address: str, kernel_module_path: str, test_func_name: str,
                          test_script_output: str, selected_functions: List[str], 
                          function_docs: str) -> bool:
        """
        Run the iterative improvement loop.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Initialize iteration tracking
        previous_error_message = ""
        previous_iteration_info = []
        
        # Run the initial test
        error_message = self.run_initial_test(test_func_name, kernel_func_name, 
                                            kernel_module_path, test_script_output)
        previous_error_message = error_message
        
        # If no errors in the initial code, we're done
        if error_handler.is_successful_result(error_message):
            print("No errors detected in initial kernel! Kernel generation successful.")
            return True
        
        # Iterative error correction loop
        for iteration in range(config.max_iterations_per_attempt):
            print(f"\n=== Iteration {iteration + 1} ===")
            
            # Store the previous error message before running any new tests
            old_error_message = previous_error_message
            
            # Select relevant errors and load documentation
            selected_errors = error_handler.select_relevant_errors(error_message)
            error_documentation = error_handler.load_error_documentation(selected_errors)
            
            # Log the selected errors and their documentation
            logging_manager.log_error_selection(output_address, error_message, 
                                              selected_errors, error_documentation)
            
            # Check if we need additional functions
            new_functions = self.select_additional_functions(selected_functions, error_message)
            if new_functions:
                selected_functions.extend(new_functions)
                additional_docs = load_function_documentation(config.docs_dir, new_functions)
                function_docs += "\n\n" + additional_docs
            
            # Create iteration history for context
            iteration_history = ""
            if previous_iteration_info:
                iteration_history = "Previous iterations:\n"
                for idx, info in enumerate(previous_iteration_info):
                    iteration_history += f"Iteration {idx + 1}:\n{info}\n\n"
            
            # Generate improved kernel
            print(f"Generating improved kernel (iteration {iteration + 1})...")
            kernel_code, reasoning_text = self.generate_improved_kernel(
                system_prompt, user_prompt, iteration_history, previous_error_message,
                function_docs, output_address, kernel_module_path, kernel_func_name
            )
            
            # Add reasoning to iteration history
            if reasoning_text:
                previous_iteration_info.append(f"Reasoning: {reasoning_text}")
            
            # Add the code snippet to the iteration history
            previous_iteration_info.append(f"Generated code: {kernel_code[:500]}...")
            
            # Run the test
            error_message = test_runner.run_test(test_func_name, kernel_func_name, 
                                               kernel_module_path, test_script_output)
            
            # Add test results to iteration history
            previous_iteration_info.append(f"Test result: {error_message[:500]}...")
            
            # Analyze change effectiveness if not the first iteration
            change_result = None
            if iteration > 0:
                change_result = error_handler.analyze_change_effectiveness(
                    old_error_message, reasoning_text, error_message
                )
                previous_iteration_info.append(f"Change report: correct={change_result.get('correct', False)}, report={change_result.get('report', 'No report')}")
            
            # Update the previous error message for the next iteration
            previous_error_message = error_message
            
            # If no errors, we're done
            if error_handler.is_successful_result(error_message):
                print("No errors detected! Kernel generation successful.")
                return True
        
        # If we get here, we've exhausted all iterations
        print(f"Exhausted {config.max_iterations_per_attempt} iterations without success.")
        return False


# Global iteration manager instance
iteration_manager = IterationManager() 