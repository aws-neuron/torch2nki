import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    """
    Vector addition kernel that performs element-wise addition of two input tensors.
    
    Parameters:
    v1: Input tensor (first vector)
    v2: Input tensor (second vector)
    
    Returns:
    result: Output tensor containing the sum of v1 and v2
    """
    # Validate that the input vectors have the same shape
    if v1.shape != v2.shape:
        raise ValueError("Input vectors must have the same shape")
    
    # Create an output tensor of the same shape, filled with zeros
    result = nl.zeros(v1.shape, dtype=v1.dtype)

    # Define the range for the loop
    size = v1.shape[0]  # Assume v1 is 1D for this example
    for i in nl.arange(size):
        # Load the elements from the input tensors
        a = nl.load(v1[i:i+1])  # Load a single element
        b = nl.load(v2[i:i+1])  # Load a single element
        
        # Perform element-wise addition
        c = nl.add(a, b)

        # Store the result back into the output tensor
        nl.store(result[i:i+1], c)

    return result