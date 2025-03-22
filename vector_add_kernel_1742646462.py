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
    result = nl.zeros(shape=[v1.shape, 1], dtype=np.float32, buffer=nl.sbuf)

    # Check if the input tensors are 1D or 2D and handle accordingly
    if len(v1.shape) == 1:  # 1D case
        i = nl.arange(v1.shape[0])[:, None]  # Reshape to have two dimensions
        a = nl.load(v1[i])
        b = nl.load(v2[i])
        c = nl.add(a, b)
        nl.store(result[i], c)
    else:  # 2D case
        for i in nl.arange(v1.shape[0]):  # Loop over the first dimension (rows)
            for j in nl.arange(v1.shape[1]):  # Loop over the second dimension (columns)
                # Load the elements from the input tensors
                a = nl.load(v1[i, j])
                b = nl.load(v2[i, j])
                
                # Perform element-wise addition
                c = nl.add(a, b)

                # Store the result back into the output tensor
                nl.store(result[i, j], c)

    return result