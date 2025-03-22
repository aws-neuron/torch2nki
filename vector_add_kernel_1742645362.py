import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    """
    Element-wise addition of two vectors using NKI.
    
    :param v1: Input tensor representing the first vector.
    :param v2: Input tensor representing the second vector.
    :return: Output tensor containing the result of adding v1 and v2.
    """
    # Get the size of the input vectors
    size = v1.shape[0]

    # Create an output tensor of shape (size, 1) to ensure it's at least 2D
    result = nl.zeros((size, 1), dtype=v1.dtype)

    # Define the range for the loop using 1D indexing
    indices = nl.arange(size)[:, None]  # Modified to use 2D indexing
    for i in indices:
        # Load the elements from the input tensors using flat indexing
        a = nl.load(v1[i, 0:1])  # Load a single element from v1
        b = nl.load(v2[i, 0:1])  # Load a single element from v2
        
        # Perform element-wise addition
        c = nl.add(a, b)

        # Store the result back into the output tensor using flat indexing
        nl.store(result[i, 0:1], c)

    return result