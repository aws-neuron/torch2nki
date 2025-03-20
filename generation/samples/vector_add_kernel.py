import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    """
    Vector addition kernel that adds two input vectors element-wise.

    :param v1: First input vector (1D tensor).
    :param v2: Second input vector (1D tensor).
    :return: Resultant vector after addition (1D tensor).
    """
    # Assume v1 and v2 are 1D tensors of the same size
    size = v1.shape[0]

    # Create an output tensor of the same size, ensuring the shape is a tuple
    result = nl.zeros((size,), dtype=v1.dtype)

    # Define the range for the loop using affine_range
    for i in nl.affine_range(size):  # Use affine_range instead of arange for compatibility
        # Load the elements from the input tensors
        a = nl.load(v1[i:i + 1])  # Load one element for current index
        b = nl.load(v2[i:i + 1])  # Load one element for current index
        
        # Perform element-wise addition
        c = nl.add(a, b)

        # Store the result back into the output tensor
        nl.store(result[i:i + 1], c)  # Store the computed value

    return result