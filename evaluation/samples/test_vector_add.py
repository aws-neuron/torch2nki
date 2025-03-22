import re
import torch
import numpy as np
import importlib
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import os
import sys
import time

def extract_kernel_from_llm_response(input_data):
    """
    Extracts kernel code from LLM response.
    Input can be either a file path (string) or the content itself.
    """
    # Determine if input_data is a file path or content
    if isinstance(input_data, str) and os.path.isfile(input_data):
        # It's a file path, read the content
        with open(input_data, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        # Assume it's already the content
        content = input_data
    
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
    """
    pattern = re.compile(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE)
    match = pattern.search(kernel_code)
    if match:
        return match.group(1)
    return None

def main():
    # Define paths
    llm_output_path = "/home/ubuntu/torch2nki/generation/langchain_single_pass/langchain_files/langchain_outputs/vector_add.txt"
    kernel_module_name = "vector_add_kernel"
    kernel_module_path = f"{kernel_module_name}.py"
    
    # Create a timestamp for uniqueness
    timestamp = int(time.time())
    unique_module_name = f"{kernel_module_name}_{timestamp}"
    unique_module_path = f"{unique_module_name}.py"
    
    print(f"Reading LLM output from: {llm_output_path}")
    
    # Check if file exists
    if not os.path.exists(llm_output_path):
        print(f"ERROR: LLM output file not found at {llm_output_path}")
        return
        
    # Extract kernel code from LLM output
    try:
        # Read the file content first
        with open(llm_output_path, "r", encoding="utf-8") as f:
            file_content = f.read()
            
        print(f"Read {len(file_content)} characters from file")
        print(f"First 100 characters: {file_content[:100]}...")
        
        # Extract kernel code
        kernel_code = extract_kernel_from_llm_response(file_content)
        print(f"Extracted {len(kernel_code)} characters of kernel code")
        print(f"First 100 characters of extracted code: {kernel_code[:100]}...")
        
        # Find function name
        func_name = find_function_name_in_code(kernel_code)
        print(f"Detected function name: {func_name}")
        
        # Write kernel to both the standard and unique files
        with open(kernel_module_path, "w", encoding="utf-8") as f:
            f.write(kernel_code)
        with open(unique_module_path, "w", encoding="utf-8") as f:
            f.write(kernel_code)
            
        print(f"Wrote kernel code to: {kernel_module_path}")
        print(f"Also wrote to unique module: {unique_module_path}")
        
        # Import the unique module to avoid caching issues
        spec = importlib.util.spec_from_file_location(unique_module_name, unique_module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        print(f"Successfully imported module: {unique_module_name}")
        
        # Get the kernel function from the module
        if func_name and hasattr(module, func_name):
            kernel_func = getattr(module, func_name)
            print(f"Using detected function: {func_name}")
        elif hasattr(module, "vector_add_kernel"):
            kernel_func = getattr(module, "vector_add_kernel")
            print("Using default function: vector_add_kernel")
        else:
            print(f"ERROR: Could not find kernel function in module. Available attributes: {dir(module)}")
            return
            
        # Create random 1D tensors
        np.random.seed(0)
        lhs_small = torch.rand((128,))
        rhs_small = torch.rand((128,))
        
        print("Running NKI kernel simulation...")
        # Run NKI kernel using simulate_kernel
        output_nki = nki.simulate_kernel(
            kernel_func,
            np.array(lhs_small),
            np.array(rhs_small)
        )
        
        # Compare with PyTorch reference
        output_torch = torch.add(lhs_small, rhs_small)
        
        # Print comparison
        print("\n--- Results Comparison ---")
        print("NKI output (first 5):", output_nki[:5])
        print("PyTorch output (first 5):", output_torch[:5].numpy())
        
        # allclose check
        if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
            print("\n✅ SUCCESS: NKI and PyTorch outputs match!")
        else:
            print("\n❌ ERROR: NKI and PyTorch outputs differ!")
            # Print detailed comparison
            diff_count = 0
            for i in range(len(output_nki)):
                diff = abs(float(output_torch[i]) - float(output_nki[i]))
                if diff > 1e-4:
                    print(f"Element {i}: PyTorch={float(output_torch[i]):.6f}, NKI={float(output_nki[i]):.6f}, Diff={diff:.6f}")
                    diff_count += 1
                    if diff_count >= 10:  # Limit to 10 differences
                        print("...")
                        break
                        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()