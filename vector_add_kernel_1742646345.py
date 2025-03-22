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
    
    # Ensure both tensors have a valid shape for NKI
    if len(v1.shape) < 1 or len(v2.shape) < 1:
        raise ValueError("Input vectors must have at least one dimension")
    
    # Create an output tensor of the same shape with zeros
    result_shape = v1.shape if len(v1.shape) > 1 else (v1.shape[0], 1)
    result = nl.zeros(result_shape, dtype=v1.dtype)

    # Using multi-dimensional indexing to properly iterate over the tensor
    # Create index arrays for each dimension
    idxs = nl.mgrid[0:v1.shape[0], 0:v1.shape[1]] if len(v1.shape) > 1 else nl.mgrid[0:v1.shape[0], 0:1]

    # Iterate through each index and compute the addition
    for i in nl.arange(v1.shape[0]):  # Loop over the first dimension (rows)
        for j in nl.arange(v1.shape[1]):  # Loop over the second dimension (columns)
            # Compute the multi-dimensional index
            multi_idx = (i, j) if len(v1.shape) > 1 else (i, 0)

            # Load the elements from the input tensors
            a = nl.load(v1[multi_idx])
            b = nl.load(v2[multi_idx])

            # Perform element-wise addition
            c = nl.add(a, b)

            # Store the result back into the output tensor
            nl.store(result[multi_idx], c)

    return result