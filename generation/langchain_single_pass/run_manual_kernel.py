from neuronxcc import nki
import neuronxcc.nki.language as nl
import math

# *** IMPORTANT: Define the feature size your kernel should be compiled for ***
STATIC_FEATURE_SIZE = 8 # Or choose a relevant size for your use case

@nki.jit
def nki_sort_(a_tensor):
    # Kernel assumes it operates on a conceptually 2D input
    # and *always* produces a 2D output.
    # Inner loops are unrolled using static_range.

    # Get tensor dimensions - determine logical sz_p, sz_f
    if len(a_tensor.shape) == 1:
        sz_f_input = a_tensor.shape[0]
        sz_p = 1
        is_1d_input = True
    elif len(a_tensor.shape) == 2:
        sz_p, sz_f_input = a_tensor.shape
        is_1d_input = False
    else:
        raise ValueError("nki_sort currently only supports 1D or 2D tensors")

    # *** ASSERTION: Check if input matches the compiled static size ***
    # Note: NKI doesn't have direct assert, this check happens before kernel launch usually.
    #       We add it here conceptually. In practice, you ensure input matches.
    if sz_f_input != STATIC_FEATURE_SIZE:
         # This error wouldn't happen inside the kernel, but indicates a mismatch
         # before calling the compiled kernel.
         raise ValueError(f"Input feature size ({sz_f_input}) does not match "
                          f"compiled static size ({STATIC_FEATURE_SIZE})")

    # Use the static size for internal allocations and loops
    sz_f = STATIC_FEATURE_SIZE

    # ---- Step 2: Workspace (SBUF) - Always (sz_p, STATIC_FEATURE_SIZE) ----
    values_sbuf = nl.ndarray((sz_p, sz_f), dtype=a_tensor.dtype, buffer=nl.sbuf)
    indices_sbuf = nl.ndarray((sz_p, sz_f), dtype=nl.int32, buffer=nl.sbuf)

    # ---- Step 3: Load (HBM -> SBUF) ----
    for p in nl.affine_range(sz_p): # Keep outer loop affine
        # Use static_range for the dimension matching STATIC_FEATURE_SIZE
        for j in nl.static_range(sz_f):
            if is_1d_input:
                values_sbuf[p, j] = nl.load(a_tensor[j])
            else:
                values_sbuf[p, j] = nl.load(a_tensor[p, j])
            indices_sbuf[p, j] = j

    # ---- Step 4: Compute (within SBUF) - Using static_range ----
    for p in nl.affine_range(sz_p): # Keep outer loop affine
        # *** Use static_range for i and j loops ***
        for i in nl.static_range(sz_f - 1):
            # The inner loop bound still logically depends on 'i',
            # but static_range unrolls it based on the constant sz_f.
            # We iterate j from 0 up to sz_f - i - 2
            for j in nl.static_range(sz_f - 1 - i): # Max value for j is sz_f - 2
                # Load values for comparison
                val_j = values_sbuf[p, j]
                val_j1 = values_sbuf[p, j+1]
                idx_j = indices_sbuf[p, j]
                idx_j1 = indices_sbuf[p, j+1]

                # Comparison and conditional swap using nl.where
                swap_condition = nl.greater(val_j, val_j1)
                new_val_j = nl.where(swap_condition, val_j1, val_j)
                new_val_j1 = nl.where(swap_condition, val_j, val_j1)
                new_idx_j = nl.where(swap_condition, idx_j1, idx_j)
                new_idx_j1 = nl.where(swap_condition, idx_j, idx_j1)

                # Store back to SBUF
                values_sbuf[p, j] = new_val_j
                values_sbuf[p, j+1] = new_val_j1
                indices_sbuf[p, j] = new_idx_j
                indices_sbuf[p, j+1] = new_idx_j1

    # ---- Step 5: Output Buffers (HBM) - Always 2D ----
    values_out = nl.ndarray((sz_p, sz_f), dtype=a_tensor.dtype, buffer=nl.hbm)
    indices_out = nl.ndarray((sz_p, sz_f), dtype=nl.int32, buffer=nl.hbm)

    # ---- Step 6: Store (SBUF -> HBM) - Always 2D ----
    for p in nl.affine_range(sz_p): # Keep outer loop affine
        # Use static_range for the dimension matching STATIC_FEATURE_SIZE
        for j in nl.static_range(sz_f):
             nl.store(values_out[p, j], values_sbuf[p, j])
             nl.store(indices_out[p, j], indices_sbuf[p, j])

    return (values_out, indices_out)

# --- Example Usage (Needs adjustment and careful size matching) ---
# import torch
# import torch_neuronx

# # *** Ensure input tensor matches STATIC_FEATURE_SIZE ***
# FEATURE_DIM = STATIC_FEATURE_SIZE
# BATCH_DIM = 2 # Example batch size

# # Compile the kernel (specific to STATIC_FEATURE_SIZE)
# compiled_sort_static = torch_neuronx.kompile(nki_sort_static_inner)

# def nki_sort_static_wrapper(input_tensor):
#     input_shape = input_tensor.shape
#     input_is_1d = len(input_shape) == 1
#     feature_size = input_shape[-1]

#     # *** Crucial Check before calling kernel ***
#     if feature_size != STATIC_FEATURE_SIZE:
#         raise ValueError(f"Input feature size ({feature_size}) does not match "
#                          f"compiled static size ({STATIC_FEATURE_SIZE})")

#     values_2d, indices_2d = compiled_sort_static(input_tensor)

#     if input_is_1d:
#         return values_2d[0], indices_2d[0]
#     else:
#         return values_2d, indices_2d

# # Example 1: 1D Tensor (MUST have size STATIC_FEATURE_SIZE)
# try:
#    input_tensor_1d = torch.rand(FEATURE_DIM, dtype=torch.bfloat16).cuda() * 10
#    sorted_values_1d, sorted_indices_1d = nki_sort_static_wrapper(input_tensor_1d)
#    print("Input 1D:", input_tensor_1d)
#    print("Sorted Values 1D:", sorted_values_1d.cpu().float())
#    print("Sorted Indices 1D:", sorted_indices_1d.cpu())
# except ValueError as e:
#    print(f"Error processing 1D tensor: {e}")


# # Example 2: 2D Tensor (MUST have inner size STATIC_FEATURE_SIZE)
# try:
#    input_tensor_2d = torch.rand(BATCH_DIM, FEATURE_DIM, dtype=torch.bfloat16).cuda() * 10
#    sorted_values_2d, sorted_indices_2d = nki_sort_static_wrapper(input_tensor_2d)
#    print("\nInput 2D:", input_tensor_2d)
#    print("Sorted Values 2D:", sorted_values_2d.cpu().float())
#    print("Sorted Indices 2D:", sorted_indices_2d.cpu())
# except ValueError as e:
#    print(f"Error processing 2D tensor: {e}")