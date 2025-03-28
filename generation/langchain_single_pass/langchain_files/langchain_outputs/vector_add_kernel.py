import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    """
    Kernel function to perform element-wise addition of two vectors using the AWS NKI.
    
    Parameters:
    v1: Input vector 1 (1D tensor)
    v2: Input vector 2 (1D tensor)

    Returns:
    result: A tensor containing the element-wise sum of v1 and v2.
    """
    # Get the size of the input vectors
    size = v1.shape[0]

    # Create an output tensor of zeros of shape (size,) and dtype matching inputs
    result = nl.zeros((size,), dtype=v1.dtype)

    # Define a loop over the range of the vector size
    for i in nl.arange(size):
        # Load each element from input vectors directly as scalars
        a = nl.load(v1[i])  # Load the i-th element from v1
        b = nl.load(v2[i])  # Load the i-th element from v2
        
        # Perform element-wise addition
        c = nl.add(a, b)
        
        # Store the computed result directly into the output tensor
        nl.store(result[i], c)  # Store the result correctly in the result tensor

    return result