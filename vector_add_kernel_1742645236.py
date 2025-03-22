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

    # Create an output tensor of the same size and type as the input tensors
    result = nl.zeros((size,), dtype=v1.dtype)

    # Define the range for the loop using 2D indexing
    indices = nl.arange(size)[:, None]  # Change here to create a 2D tensor for indexing
    for i in indices:
        # Load the elements from the input tensors
        a = nl.load(v1[i, 0:1])  # Load a single element from v1
        b = nl.load(v2[i, 0:1])  # Load a single element from v2
        
        # Perform element-wise addition
        c = nl.add(a, b)

        # Store the result back into the output tensor
        nl.store(result[i, 0:1], c)

    return result