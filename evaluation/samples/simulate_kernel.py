#A basic script to ensure that the kernels run successfully



import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import numpy as np
from matrix_multiplication_nki_kernels import nki_matmul_basic_, nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_



@nki.jit
def print_kernel():
  a = nl.ndarray([4, 4], dtype=nl.float32, buffer=nl.shared_hbm)

  # Create (4, 4) tensor in sbuf
  y = nl.zeros([4, 4], dtype=np.float32)

  # Print tensor y
  nl.device_print("value of y:", y)

  # Directly store tensor y as a single tile
  nl.store(a, value=y)

  return a

np.random.seed(0)

import torch

lhs_small = torch.rand((64, 128))
rhs_small = torch.rand((128, 512))

output_small = nki.simulate_kernel(nki_matmul_basic_, np.array(lhs_small.T), np.array(rhs_small))


# Run torch reference
output_small_torch = torch.matmul(lhs_small, rhs_small)

  # Compare results
print("Checking correctness of nki_matmul_basic")
if torch.allclose(output_small_torch, torch.Tensor(output_small), atol=1e-4, rtol=1e-2):
   print("NKI and Torch match")
else:
   print("NKI and Torch differ")




