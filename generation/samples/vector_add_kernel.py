import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    """
    A kernel that adds two vectors element-wise.

    :param v1: First input vector (1D tensor)
    :param v2: Second input vector (1D tensor)
    :return: A tensor that contains the element-wise sum of v1 and v2
    """
    
    # Determine the size of the input vectors
    size = v1.shape[0]

    # Create an output tensor of the same size, initialized to zeros
    result = nl.zeros((size,), dtype=v1.dtype)  # Ensure the shape is a tuple

    # Perform element-wise addition using efficient loading and storing
    for i in nl.arange(size):
        # Load the elements from the input tensors
        a = nl.load(v1[i:i+1])
        b = nl.load(v2[i:i+1])
        
        # Perform element-wise addition
        c = nl.add(a, b)

        # Store the result back into the output tensor
        nl.store(result[i:i+1], c)

    return result