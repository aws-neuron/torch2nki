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

    # Create an output tensor of shape (size, 1) in HBM
    result = nl.zeros((size, 1), dtype=v1.dtype, buffer=nl.hbm)

    # Iterate over the size using affine_range
    for i in nl.affine_range(size):  # Using affine_range for loop control
        # Load the elements from the input tensors using flat indexing
        a = nl.load(v1[i:i+1])  # Load a single element from v1 (should match output shape)
        b = nl.load(v2[i:i+1])  # Load a single element from v2 (should match output shape)
        
        # Perform element-wise addition
        c = nl.add(a, b)

        # Store the result back into the output tensor using flat indexing
        nl.store(result[i:i+1], c)

    return result