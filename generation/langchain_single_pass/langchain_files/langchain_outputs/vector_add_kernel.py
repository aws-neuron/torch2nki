from neuronxcc import nki
import neuronxcc.nki.language as nl
import numpy as np

@nki.jit
def vector_add_kernel(v1, v2):
    """
    Vector addition kernel using AWS Neural Kernel Interface (NKI)
    
    Args:
        v1 (nki.Tensor): First input vector
        v2 (nki.Tensor): Second input vector
    
    Returns:
        nki.Tensor: Resulting vector after element-wise addition
    """
    # Get the size of the input vector
    size = v1.shape[0]
    
    # Create output tensor with explicit 2D shape
    result = nl.zeros((size, 1), dtype=v1.dtype)
    
    # Use indexing to perform element-wise addition
    for i in nl.arange(size):
        # Load individual elements
        a = nl.load(v1[i:i+1])
        b = nl.load(v2[i:i+1])
        
        # Add elements
        c = nl.add(a, b)
        
        # Store result
        nl.store(result[i:i+1, 0:1], c)
    
    return result