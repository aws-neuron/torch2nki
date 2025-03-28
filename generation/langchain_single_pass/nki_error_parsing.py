import os
import re
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

######################
# NKI Error Parser
######################
class NKIErrorParser:
    """Parser for NKI error messages documentation."""
    def __init__(self, error_doc_path):
        """
        Initialize the error parser with the path to the error documentation file.
        Args:
            error_doc_path (str): Path to the NKI error messages documentation file
        """
        self.error_doc_path = error_doc_path
        self.error_database = self._parse_error_file()
        
    def _parse_error_file(self):
        """
        Parse the error documentation file and build an error database.
        Returns:
            dict: A dictionary mapping error codes to their documentation
        """
        if not os.path.exists(self.error_doc_path):
            print(f"Error documentation file not found: {self.error_doc_path}")
            return {}
            
        try:
            with open(self.error_doc_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Split the content by the error separator pattern
                error_pattern = re.compile(r'ERROR: ([a-zA-Z0-9_-]+)\s*\n==+\n(.*?)(?=\n==+\nERROR:|$)', re.DOTALL)
                errors = error_pattern.findall(content)
                
                error_database = {}
                for error_code, error_content in errors:
                    # Parse instructions and code examples
                    instructions = []
                    code_examples = []
                    
                    # Extract instructions
                    instruction_pattern = re.compile(r'Instruction (\d+): (.*?)(?=\nInstruction \d+:|Code Example \d+:|$)', re.DOTALL)
                    for _, instruction_text in instruction_pattern.findall(error_content):
                        instructions.append(instruction_text.strip())
                    
                    # Extract code examples
                    code_pattern = re.compile(r'Code Example (\d+):(.*?)(?=\nCode Example \d+:|$)', re.DOTALL)
                    for _, code_text in code_pattern.findall(error_content):
                        code_examples.append(code_text.strip())
                    
                    error_database[error_code] = {
                        'instructions': instructions,
                        'code_examples': code_examples,
                        'raw_content': error_content.strip()
                    }
                
                return error_database
        except Exception as e:
            print(f"Error parsing documentation file: {e}")
            return {}
    
    def get_error_info(self, error_code):
        """
        Get information about a specific error code.
        Args:
            error_code (str): The error code to look up
        Returns:
            dict: Error information including instructions and code examples
        """
        # Normalize error code by removing "ERROR: " prefix if present
        if error_code.upper().startswith("ERROR: "):
            error_code = error_code[7:]
        
        # Check if we have this error in our database
        if error_code in self.error_database:
            return self.error_database[error_code]
        
        # Try case-insensitive match if exact match fails
        for key in self.error_database.keys():
            if key.lower() == error_code.lower():
                return self.error_database[key]
        
        return None
    
    def list_all_errors(self):
        """
        List all error codes in the database.
        Returns:
            list: A list of all error codes
        """
        return list(self.error_database.keys())
    
    def search_errors(self, keyword):
        """
        Search for errors containing a keyword.
        Args:
            keyword (str): Keyword to search for in error codes and content
        Returns:
            list: A list of matching error codes
        """
        matches = []
        keyword = keyword.lower()
        
        for code, info in self.error_database.items():
            if (keyword in code.lower() or
                keyword in info['raw_content'].lower()):
                matches.append(code)
        
        return matches

######################
# Error Documentation Functions
######################

def get_available_error_codes(error_parser):
    """
    Get a list of all available NKI error codes from the error parser.
    
    Args:
        error_parser (NKIErrorParser): The error parser instance
        
    Returns:
        list: A list of error codes
    """
    return error_parser.list_all_errors()

def select_relevant_errors(llm, error_message, available_errors):
    """
    Use LLM to select relevant error codes from the error output.
    
    Args:
        llm: The LLM instance
        error_message (str): The error output message
        available_errors (list): List of available error codes
        
    Returns:
        list: List of selected error codes
    """
    error_selection_prompt = ChatPromptTemplate.from_template(
        "You are helping to identify relevant NKI error codes from error output.\n\n"
        "Here is the error output:\n{error_message}\n\n"
        "Available error codes:\n{error_list}\n\n"
        "Please identify the most relevant error codes in this output. Return your selection as a JSON list "
        "of error codes (without the 'ERROR: ' prefix). For example: [\"INVALID_TYPE\", \"OUT_OF_BOUNDS\"]"
    )
    
    # Format error list for display
    error_list = "\n".join(sorted(available_errors))
    
    error_selection_chain = (
        error_selection_prompt
        | llm
        | StrOutputParser()
    )
    
    response = error_selection_chain.invoke({
        "error_message": error_message,
        "error_list": error_list
    })
    
    # Parse the JSON response
    try:
        selected_errors = json.loads(response)
        # Validate that all selected errors are in available_errors
        valid_selections = [e for e in selected_errors if e in available_errors]
        return valid_selections
    except Exception as e:
        print(f"Error parsing selected errors: {e}")
        # Extract error codes using regex as fallback
        pattern = re.compile(r'["\']([\w_-]+)["\']')
        matches = pattern.findall(response)
        valid_selections = [e for e in matches if e in available_errors]
        return valid_selections

def load_error_documentation(error_parser, error_codes):
    """
    Load documentation for the selected error codes.
    
    Args:
        error_parser (NKIErrorParser): The error parser instance
        error_codes (list): List of error codes to load
        
    Returns:
        str: Combined error documentation text
    """
    documentation = []
    
    for error_code in error_codes:
        error_info = error_parser.get_error_info(error_code)
        if error_info:
            documentation.append(f"ERROR: {error_code}")
            documentation.append("=" * 50)
            documentation.append(error_info['raw_content'])
            documentation.append("\n" + "=" * 80 + "\n")
    
    return "\n".join(documentation)

# Add this function near the top of your file where other helper functions are defined
def extract_error_details(error_message):
    """
    Extract the actual error message and the line of code that caused the error.
    
    Args:
        error_message (str): The full error output from the test script
        
    Returns:
        tuple: (error_line, error_description) where error_line is the line of code
               that caused the error and error_description is the actual error message
    """
    lines = error_message.strip().split('\n')
    error_description = None
    error_line = None
    
    # Look for the actual error message (usually after 'ERROR:' or before the traceback)
    for i, line in enumerate(lines):
        if line.startswith('ERROR:'):
            error_description = line
            break
    
    # Find the line of code that caused the error (usually the line before 'AssertionError' or other exception)
    for i in range(len(lines) - 1):
        if (i < len(lines) - 1 and 
            ('Error' in lines[i+1] or 'Exception' in lines[i+1]) and 
            'File' not in lines[i] and 
            'line' not in lines[i]):
            error_line = lines[i].strip()
            break
    
    return error_line, error_description