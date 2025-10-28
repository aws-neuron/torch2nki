"""
Centralized logging management for the NKI kernel generation pipeline.
"""
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

from .config import config


class LoggingManager:
    """Manages all logging operations for the pipeline."""
    
    def __init__(self):
        """Initialize the logging manager."""
        pass
    
    def log_iteration_data(self, iteration_log_path: str, iteration_number: int,
                          error_message: str, error_line: Optional[str],
                          error_description: Optional[str], reasoning_text: str,
                          kernel_code: str, test_result: str,
                          change_result: Optional[Dict[str, Any]] = None,
                          append: bool = True) -> Dict[str, Any]:
        """
        Log all data from a kernel generation iteration to a single consolidated file.
        Also saves the complete kernel code to a separate file.
        """
        # Create a structured dictionary for this iteration
        iteration_data = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration_number,
            "error": {
                "message": error_message,
                "line": error_line,
                "description": error_description
            },
            "solution": {
                "reasoning": reasoning_text,
                "kernel_code": kernel_code
            },
            "test_result": test_result
        }
        
        # Add change analysis if available
        if change_result:
            iteration_data["change_analysis"] = change_result
        
        # Format the data for human-readable output
        formatted_output = f"\n{'='*80}\n"
        formatted_output += f"ITERATION {iteration_number} - {datetime.now().isoformat()}\n"
        formatted_output += f"{'='*80}\n\n"
        
        # ERROR SECTION
        formatted_output += f"--- ERROR INFORMATION ---\n\n"
        if error_line:
            formatted_output += f"ERROR LINE: {error_line}\n"
        if error_description:
            formatted_output += f"ERROR DESCRIPTION: {error_description}\n"
        formatted_output += f"\nFULL ERROR MESSAGE:\n{error_message}\n\n"
        
        # SOLUTION SECTION
        formatted_output += f"--- SOLUTION INFORMATION ---\n\n"
        if reasoning_text:
            formatted_output += f"REASONING:\n{reasoning_text}\n\n"
        
        # Save the COMPLETE kernel code
        formatted_output += f"GENERATED KERNEL CODE:\n{kernel_code}\n\n"
        
        # TEST RESULT SECTION
        formatted_output += f"--- TEST RESULT ---\n\n"
        formatted_output += f"{test_result}\n\n"
        
        # CHANGE ANALYSIS SECTION (if available)
        if change_result:
            formatted_output += f"--- CHANGE ANALYSIS ---\n\n"
            formatted_output += f"FIXED PREVIOUS ERROR: {change_result.get('correct', False)}\n"
            formatted_output += f"ANALYSIS: {change_result.get('report', 'No analysis provided')}\n\n"
        
        # Also include the raw JSON data for easier database ingestion later
        json_data = json.dumps(iteration_data, indent=2)
        formatted_output += f"--- RAW JSON DATA ---\n\n"
        formatted_output += f"{json_data}\n\n"
        
        # Write to file
        mode = "a" if append else "w"
        with open(iteration_log_path, mode, encoding="utf-8") as log_file:
            log_file.write(formatted_output)
        
        # Additionally, save the complete kernel code to a separate file
        # Use the base path without extension to create new paths
        base_path = os.path.splitext(iteration_log_path)[0]
        kernel_path = f"{base_path}_iteration_{iteration_number}_kernel.py"
        with open(kernel_path, "w", encoding="utf-8") as kernel_file:
            kernel_file.write(kernel_code)
        
        # Return the data dictionary for potential further processing
        return iteration_data
    
    def initialize_consolidated_log(self, output_address: str, kernel_module_path: str) -> str:
        """Initialize the consolidated iteration log file."""
        consolidated_log_path = output_address + ".consolidated_iterations.txt"
        
        # Initialize with header only on first write (will be overwritten)
        with open(consolidated_log_path, "w", encoding="utf-8") as f:
            f.write(f"=== CONSOLIDATED ITERATION LOG ===\n")
            f.write(f"Started at: {datetime.now()}\n")
            f.write(f"Output path: {output_address}\n")
            f.write(f"Kernel module path: {kernel_module_path}\n\n")
        
        return consolidated_log_path
    
    def log_prompt(self, prompt_path: str, prompt_content: str, append: bool = True):
        """Log prompt content to a file."""
        from ..utils.file_utils import log_to_file
        log_to_file(prompt_path, f"FULL PROMPT TO LLM:\n{prompt_content}\n", append=append)
    
    def log_error_selection(self, output_address: str, error_message: str, 
                           selected_errors: list, error_documentation: str):
        """Log error selection information."""
        with open(f"{output_address}.error_selection", "w") as f:
            f.write(f"ERROR MESSAGE:\n{error_message}\n\n")
            f.write(f"SELECTED ERRORS:\n{', '.join(selected_errors)}\n\n")
            f.write(f"ERROR DOCUMENTATION:\n{error_documentation}\n\n")


# Global logging manager instance
logging_manager = LoggingManager() 