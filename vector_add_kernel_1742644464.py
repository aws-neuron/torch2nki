import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    """
    A kernel for adding two vectors element-wise using the NKI API.
    
    Parameters:
    v1 -- Input vector 1 (1D tensor)
    v2 -- Input vector 2 (1D tensor)
    
    Returns:
    result -- Element-wise sum of v1 and v2 (1D tensor)
    """
    # Get the size of the input vectors
    size = v1.shape[0]

    # Create an output tensor of the same shape and dtype as the input vectors
    result = nl.zeros((size,), dtype=v1.dtype)  # Ensure it matches the shape and dtype of v1

    # Perform element-wise addition
    for i in nl.affine_range(size):  # Using affine_range for loop iteration
        # Load elements from the input tensors using proper indexing
        a = nl.load(v1[i])  # Load a single element from v1
        b = nl.load(v2[i])  # Load a single element from v2
        
        # Add the two elements
        c = nl.add(a, b)

        # Store the result back into the output tensor
        nl.store(result[i], c)  # Store in the result tensor

    return result