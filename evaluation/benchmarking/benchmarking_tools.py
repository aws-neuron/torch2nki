"""
A set of tools for processing LLM outputs for kernels. 
"""

import re


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
    # Create or clear the output file
    output_file = f'{kernel_name_type}.py'
    make_txt_file(output_file, '')
    
    # Add imports at the top of the file
    append_txt_file(output_file, 'import torch\nimport torch.nn.functional as F\n\n')
    
    for kernel_name in kernel_list:
        try:
            kernel_text = extract_kernel_from_llm_response(f'{kernel_name}.txt')
            kernel_text = update_function_name_in_text(kernel_text, f'nki_{kernel_name}')
            # Append each kernel to the file
            append_txt_file(output_file, kernel_text + '\n\n')
        except Exception as e:
            print(f"Error processing kernel {kernel_name}: {str(e)}")

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
    Writes the given content to a text file, overwriting any existing content.

    Args:
        file_path (str): Path to the file to write.
        content (str): Content to write to the file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def append_txt_file(file_path, content):
    """
    Appends the given content to a text file.

    Args:
        file_path (str): Path to the file to append to.
        content (str): Content to append to the file.
    """
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content)


def update_function_name_in_text(text, new_name):
    """
    Updates the function name in the function header of a text string.

    The function expects the function header to follow this format:
    def old_function_name(arguments):
        <body lines>

    Args:
        text (str): The text content to update
        new_name (str): New function name to replace the old one with

    Returns:
        str: The updated text content with the new function name
    """
    # Updated regex to capture standard Python function definitions
    pattern = r'^(def\s+)([^\s(]+)(\s*\(.*\):)'  # Matches 'def function_name(args):'
    # Replace with new function name while preserving 'def' and arguments
    replacement = r'\1' + new_name + r'\3'
    # Replace the first occurrence of the function definition
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

    
