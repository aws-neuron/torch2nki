import neuronxcc.nki.language as nl

@nki.jit
def vector_add_kernel(v1, v2):
    # Assuming v1 and v2 are 1D tensors of the same length
    size = v1.shape[0]

    # Create an output tensor with the same shape as the input tensors
    result = nl.zeros(size, dtype=v1.dtype)

    # Define the index range for processing
    i = nl.arange(size)

    # Load the input tensors
    v1_tile = nl.load(v1[i])
    v2_tile = nl.load(v2[i])

    # Perform element-wise addition
    result_tile = nl.add(v1_tile, v2_tile)

    # Store the result back to the output tensor
    nl.store(result[i], result_tile)

    return result