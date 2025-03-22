# Importing necessary NKI modules
from neuronxcc import nki
import neuronxcc.nki.language as nl

def vector_addition_kernel(a_tensor, b_tensor, c_tensor, vector_shape):
    """
    Vector addition kernel using NKI for element-wise addition of two tensors.

    :param a_tensor: Input tensor for the first vector (on HBM)
    :param b_tensor: Input tensor for the second vector (on HBM)
    :param c_tensor: Output tensor for storing the result (on HBM)
    :param vector_shape: Tuple representing the shape of the vectors
    """
    
    # Ensure that the vector dimensions are valid for addition
    assert len(vector_shape) == 2, "Vector shape must be 2-dimensional (e.g., [N, M])."
    
    # Define the partition dimensions
    batch_size = vector_shape[0]
    feature_size = vector_shape[1]

    # Process the vector addition in batches
    for i_b in nl.affine_range(batch_size):
        # Load tensors from HBM to SBUF
        a_tile = nl.load(a_tensor[i_b, :])
        b_tile = nl.load(b_tensor[i_b, :])
        
        # Perform element-wise addition
        c_tile = nl.add(a_tile, b_tile)
        
        # Store the result back to HBM
        nl.store(c_tensor[i_b, :], c_tile)

# Example usage
if __name__ == "__main__":
    # Placeholder for tensor shapes and HBM memory allocation
    # a_tensor, b_tensor, c_tensor should be defined and allocated in HBM

    # Assuming `a_tensor`, `b_tensor`, and `c_tensor` are allocated in HBM
    # and their shapes are compatible for addition
    vector_shape = (128, 512)  # Example shape
    vector_addition_kernel(a_tensor, b_tensor, c_tensor, vector_shape)