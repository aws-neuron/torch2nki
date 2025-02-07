'''
Copyright (C) 2024, Amazon.com. All Rights Reserved

Example for implementing PyTorch ATen operations as NKI kernels, starting with eager mode

- eager mode: https://github.com/pytorch/xla/blob/de7568651486a4d2e45b589090d2b328e2f8f3f7/docs/source/learn/eager.md 
- Main ATen folder for contributions: https://github.com/pytorch/pytorch/tree/b8f383107eebd9495a0f132d58a970e178e15930/aten/src/ATen/native 

Env creation:
1. create a new python env, install jupyter lab
2. set up pip repository pointing to Neuron
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
3. Install the compiler and pytorch with 
python -m pip install neuronx-cc==2.* torch-neuronx torchvision
4. check the NKI import with python -c 'import neuronxcc.nki'
5. check the PyTorch import with python -c 'import torch_neuronx'

Alternatively use a pre-built virtual environment following these steps:
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/multiframework/multi-framework-ubuntu22-neuron-dlami.html#setup-ubuntu22-multi-framework-dlami
'''

import torch
import torch.nn as nn
import torch_xla
from torch_xla.core import xla_model as xm
import os

from matrix_multiplication_nki_kernels import nki_matmul_basic_, nki_matmul_tiled_, nki_matmul_hoist_load_, nki_matmul_block_free_dimension_, nki_matmul_fully_optimized_

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

import time

# enables eager execution mode for torch xla
torch_xla.experimental.eager_mode(True)

# writes the pytorch xla operations to disk, including aten
os.environ['XLA_SAVE_TENSORS_FILE'] = 'pytorch_nki_matmul.txt'

def test_torch_matmul(device):
  # Test the small workload with basic kernel
  lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
  rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)

  # Run NKI kernel
  output_small = nki_matmul_basic_(lhs_small.T, rhs_small)

  # Run torch reference
  output_small_torch = torch.matmul(lhs_small, rhs_small)

  # Compare results
  print("Checking correctness of nki_matmul_basic")
  if torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2):
    print("NKI and Torch match")
  else:
    print("NKI and Torch differ")


if __name__ == "__main__":

  device = xm.xla_device()

  test_torch_matmul(device)



    
