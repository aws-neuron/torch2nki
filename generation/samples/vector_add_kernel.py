from neuronxcc import nki

def vector_add_kernel(v1, v2):
    """
    Kernel for adding two vectors element-wise using NKI.
    
    :param v1: First input vector (tile).
    :param v2: Second input vector (tile).
    :return: Resulting vector (tile) after addition.
    """
    # Using NKI to perform element-wise addition
    result = nki.language.add(v1, v2)
    return result

def main(v1, v2):
    """
    Main function to launch the vector addition kernel.
    
    :param v1: First vector (list or tile).
    :param v2: Second vector (list or tile).
    :return: Resulting vector after addition.
    """
    # Validate input dimensions
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    
    # Convert input lists to NKI tiles if necessary
    v1_tile = nki.tile(v1)
    v2_tile = nki.tile(v2)
    
    # Execute the kernel
    result_tile = vector_add_kernel(v1_tile, v2_tile)
    
    # Convert the result tile back to a list (if needed)
    result = result_tile.to_list()  # assuming to_list() method converts tile to list
    return result

# Example usage
if __name__ == "__main__":
    vec1 = [1, 2, 3, 4]
    vec2 = [5, 6, 7, 8]
    result = main(vec1, vec2)
    print("Result of vector addition:", result)