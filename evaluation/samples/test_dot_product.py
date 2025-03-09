import re
import torch
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

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


def main():
    # Extract the kernel code from the LLM-generated file
    file_path = "../../generation/samples/2_naive_o4_generator.txt"  # TODO: change to correct file path or make file path an arg
    
    # Extract the actual code from the LLM output
    kernel_code = extract_kernel_from_llm_response(file_path)

    # Extract the defined kernel name from the LLM output
    func_name = find_function_name_in_code(kernel_code)
    
    # Sanity check
    print(kernel_code)
    print(func_name)

    # Dynamically define the kernel
    #exec(kernel_code, globals())

    #write kernel code to a python file
    with open("dot_product_kernel.py", "w", encoding="utf-8") as f:
        f.write(kernel_code)


    #import kernel code
    from dot_product_kernel import nki_dot_product  




    # Create random 1D tensors
    np.random.seed(0)
    lhs_small = torch.rand((128,))
    rhs_small = torch.rand((128,))

    # Convert PyTorch tensors to NumPy for NKI
    # This example uses the kernel's expected shape: (lhs_rows, lhs_cols) and (lhs_cols, rhs_cols).
    # Run NKI kernel using simulate_kernel
    output_nki = nki.simulate_kernel(
        nki_dot_product, 
        np.array(lhs_small), 
        np.array(rhs_small)
    )

    # Compare with PyTorch reference to check correctness
    output_torch = torch.dot(lhs_small, rhs_small)

    # allclose check
    if torch.allclose(output_torch, torch.tensor(output_nki), atol=1e-4, rtol=1e-2):
        print("NKI and Torch match!")
    else:
        print("NKI and Torch differ!")

if __name__ == "__main__":
    main()
