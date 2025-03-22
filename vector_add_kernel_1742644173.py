import neuronxcc.nki.language as nl
from neuronxcc import nki

@nki.jit
def vector_add_kernel(v1, v2):
    # Assume v1 and v2 are 1D tensors of the same size
    size = v1.shape[0]  # Get the size of the input vectors

    # Create an output tensor of the same size, filled with zeros
    result = nl.zeros(size, dtype=v1.dtype)  # Ensure shape is a single integer for 1D

    # Load the entire input tensors at once
    a = nl.load(v1)  # Load all elements
    b = nl.load(v2)  # Load all elements

    # Perform element-wise addition
    c = nl.add(a, b)

    # Store the result back into the output tensor
    nl.store(result, c)

    return result  # Return the result tensor