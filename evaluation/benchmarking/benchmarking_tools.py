"""
A set of tools for processing LLM outputs for kernels. 
"""

import re
import torch
import numpy as np


def extract_kernel_from_llm_response(file_path):
    #TODO: Deal with edge cases defined by Emily

    """
    Reads the LLM-generated file, locates the Python code block
    (enclosed by triple backticks), and extracts only the code inside.
    Returns a string containing the kernel definition.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to match a fenced code block marked with ```python ... ```
    pattern = re.compile(r"```python\s+(.*?)\s+```", re.DOTALL)
    match = pattern.search(content)
    if not match:
        raise ValueError("Could not find a fenced code block containing the kernel definition.")
    
    # Extract and return only the code portion
    kernel_code = match.group(1)
    return kernel_code.strip()

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