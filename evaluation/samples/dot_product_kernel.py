from neuronxcc import nki
import neuronxcc.nki.language as nl

@nki.jit
def nki_dot_product(a_tensor, b_tensor):
    # Ensure both tensors are 1D
    if a_tensor.shape[0] != b_tensor.shape[0]:
        raise ValueError("Vectors must be of the same length")
        
    # Initialize a scalar to hold the sum result
    sum_result = nl.zeros((), dtype=nl.float32, buffer=nl.psum)

    # Process the dot product
    for i in nl.affine_range(a_tensor.shape[0]):
        a_value = nl.load(a_tensor[i])
        b_value = nl.load(b_tensor[i])
        sum_result += nl.multiply(a_value, b_value)

    return sum_result