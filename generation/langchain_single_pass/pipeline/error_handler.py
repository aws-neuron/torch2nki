"""
Error handling and analysis for the NKI kernel generation pipeline.
"""
import re
from typing import Tuple, List, Optional, Dict, Any

from ..core.config import config
from ..core.llm_manager import llm_manager
from ..utils.json_utils import parse_json_response, extract_json_from_response, clean_json_string
from ..utils.prompt_utils import create_error_selection_prompt, create_change_report_prompt
from nki_error_parsing import NKIErrorParser, extract_error_details, get_available_error_codes, load_error_documentation


class ErrorHandler:
    """Handles error parsing, documentation, and analysis."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_parser = NKIErrorParser(config.error_doc_path)
    
    def extract_error_details(self, error_message: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract error line and description from error message."""
        return extract_error_details(error_message)
    
    def get_available_errors(self) -> List[str]:
        """Get all available error codes."""
        return get_available_error_codes(self.error_parser)
    
    def select_relevant_errors(self, error_message: str) -> List[str]:
        """Select relevant error codes for a given error message."""
        available_errors = self.get_available_errors()
        
        # Create prompt for error selection
        error_selection_prompt = create_error_selection_prompt(error_message, available_errors)
        
        try:
            # Use direct API call with retry logic
            error_response = llm_manager.invoke_with_retry(error_selection_prompt, temperature=0.3)
        except Exception as e:
            print(f"Error in error selection: {e}")
            return []
        
        # Parse the response
        selected_errors = parse_json_response(error_response)
        
        # Validate that all selected errors are in available_errors
        selected_errors = [e for e in selected_errors if e in available_errors]
        
        return selected_errors
    
    def load_error_documentation(self, selected_errors: List[str]) -> str:
        """Load documentation for selected errors."""
        error_documentation = load_error_documentation(self.error_parser, selected_errors)
        
        # If no documented errors found, use a fallback message
        if not selected_errors:
            error_documentation = "No specific documentation found for the errors in the output. Please analyze the error message carefully."
        
        return error_documentation
    
    def analyze_change_effectiveness(self, old_error_message: str, reasoning_text: str, 
                                   error_message: str) -> Dict[str, Any]:
        """Analyze whether the changes fixed the previous error."""
        # Extract error line from old error message if possible
        old_error_line, _ = self.extract_error_details(old_error_message)
        new_error_line, _ = self.extract_error_details(error_message)
        
        old_error_line_info = f"Error occurred at line: {old_error_line}" if old_error_line else "Error line could not be determined."
        new_error_line_info = f"Error occurred at line: {new_error_line}" if new_error_line else "Error line could not be determined."
        
        change_report_prompt = create_change_report_prompt(
            old_error_message, old_error_line_info, reasoning_text, 
            error_message, new_error_line_info
        )
        
        try:
            # Use direct API call with retry logic
            change_report_json = llm_manager.invoke_with_retry(change_report_prompt, temperature=0.3)
        except Exception as e:
            print(f"Error in change report generation: {e}")
            return {"correct": False, "report": "Error occurred during report generation"}
        
        # Extract JSON from the response
        json_str = extract_json_from_response(change_report_json)
        json_str = clean_json_string(json_str)
        
        try:
            import json
            report_data = json.loads(json_str)
            correct = report_data.get("correct", False)
            report = report_data.get("report", "No explanation provided")
            return {"correct": correct, "report": report}
        except json.JSONDecodeError:
            # Fallback in case JSON parsing fails
            print("Failed to parse JSON response. Using default values.")
            return {"correct": False, "report": change_report_json}
    
    def is_successful_result(self, error_message: str) -> bool:
        """Check if the error message indicates a successful result."""
        return not any(keyword in error_message.lower() for keyword in ["error", "exception", "failed", "failure"])


# Global error handler instance
error_handler = ErrorHandler() 