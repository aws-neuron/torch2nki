"""
A set of tools for processing LLM outputs for kernels. 
"""

import re
import torch
import numpy as np


def process(kernel_list, kernel_name_type):
    """
    Processes a list of kernels by:

    1. Extracting the kernel code from the LLM response
    2. Updating the function name in the kernel code
    3. Writing the updated kernel code to a file

    Args:
        kernel_list (list): List of kernel names to process
        kernel_name_type (str): Type of kernel being tested
    """
    make_txt_file(f'{kernel_name_time}.txt', '')
    for kernel_name in kernel_list:
        kernel_text = extract_kernel_from_llm_response(f'{kernel_name}.txt')
        kernel_text = update_function_name_in_text(kernel_text, kernel_name)
        write_txt_file(f'{kernel_name_type}.py', kernel_text)





def make_txt_file(file_path, content):
    """
    Creates a text file with the given content.

    Args:
        file_path (str): Path to the file to create.
        content (str): Content to write to the file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def write_txt_file(file_path, content):
    """
    Writes the given content to a text file.

    Args:
        file_path (str): Path to the file to write.
        content (str): Content to write to the file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def update_function_name_in_text(text, new_name):
    """
    Updates the function name in the function header of a text string.
    
    The function expects the function header to follow this format:
    --old_function_name(arguments):
        <body lines>
    
    Args:
        text (str): The text content to update
        new_name (str): New function name to replace the old one with
    
    Returns:
        str: The updated text content with the new function name
    """
    # This regex captures the function name and the rest of the header:
    #   - Group 1: the old function name (one or more characters until an opening parenthesis)
    #   - Group 2: the arguments and trailing colon (e.g., "(arg1, arg2):")
    pattern = r'^--([^(]+)(\([^)]*\):)'
    
    # Create a replacement string that puts the new function name and reuses the captured arguments and colon
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