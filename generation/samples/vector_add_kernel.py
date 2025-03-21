import neuronxcc as nki

# Define the kernel for vector addition
@nki.kernel
def vector_add_kernel(v1, v2, result):
    """
    Kernel to add two vectors element-wise.
    
    :param v1: Input vector 1 (tile)
    :param v2: Input vector 2 (tile)
    :param result: Output vector (tile)
    """
    # Define the range for the vector elements
    for i in nki.language.affine_range(v1.shape[0]):
        result[i] = v1[i] + v2[i]

def vector_add(v1, v2):
    """
    Wrapper function to perform vector addition using the NKI kernel.

    :param v1: List of numbers (first vector)
    :param v2: List of numbers (second vector)
    :return: List representing the sum of the two vectors
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")

    # Convert input lists to NKI tiles
    tile_v1 = nki.make_tile(v1)
    tile_v2 = nki.make_tile(v2)
    result_tile = nki.make_tile([0] * len(v1))  # Initialize output tile

    # Launch the kernel
    vector_add_kernel(tile_v1, tile_v2, result_tile)

    # Retrieve and return the result as a list
    return result_tile.to_list()

# Example usage
if __name__ == "__main__":
    v1 = [1, 2, 3, 4]
    v2 = [5, 6, 7, 8]
    result = vector_add(v1, v2)
    print("Result of vector addition:", result)