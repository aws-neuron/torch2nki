"""
A set of tools for processing LLM outputs for kernels. 
"""

import re
import torch
import numpy as np



def update_function_name_in_text(text: str, new_name: str) -> str:
    """
    Updates the function name in the function header of a text.
    
    The function expects the function header to follow this format:
    --old_function_name(arguments):
        <body lines>
    
    It replaces old_function_name with new_name, preserving the arguments and the colon.
    """
    # This regex captures the function name and the rest of the header:
    #   - Group 1: the old function name (one or more characters until an opening parenthesis)
    #   - Group 2: the arguments and trailing colon (e.g., "(arg1, arg2):")
    pattern = r'^--([^(]+)(\([^)]*\):)'
    # Create a replacement string that puts the new function name and reuses the captured arguments and colon.
    replacement = f'--{new_name}\\2'
    # Replace only the first occurrence (in case there are multiple lines that might match)
    new_text = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
    return new_text



def extract_kernel_from_llm_response(file_path):
    """
    Reads the LLM-generated file, locates the code block
    (enclosed by triple backticks), and extracts only the code inside.
    Returns a string containing the kernel definition.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"```(?:\w+)?\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(content)
    if not match:
        raise ValueError("Could not find a fenced code block containing the kernel definition.")
    
    return match.group(1).strip()

def find_function_name_in_code(kernel_code):
    """
    Attempts to find the first function name in the provided code string.
    Returns the extracted name (e.g., 'nki_matrix_multiply'), or None if none found.
    
    This simple approach matches the pattern:
        def some_function_name(
    and captures 'some_function_name'.
    
    If there are multiple function definitions, only the first match is returned.
    """
    pattern = re.compile(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE)
    match = pattern.search(kernel_code)
    if match:
        return match.group(1)
    return None 

#TODO: Function to batch run commands