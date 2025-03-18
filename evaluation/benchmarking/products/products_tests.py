import torch
import torch.nn as nn
import torch_xla
from torch_xla.core import xla_model as xm
import os

import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import numpy as np

from products_nki_kernels import (
    nki_inner, nki_outer, nki_dot, nki_vdot, nki_cross, nki_matmul,
    nki_mm, nki_mv, nki_bmm, nki_tensordot, nki_einsum, nki_kron,
    nki_hadamard, nki_linalg_vecdot, nki_linalg_multi_dot
)

def test_torch_inner(device):
    """Test inner product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand(10, dtype=torch.bfloat16, device=device)
    b = torch.rand(10, dtype=torch.bfloat16, device=device)
    output_nki = nki_inner(a, b)
    output_torch = torch.inner(a, b)
    print("Checking correctness of inner product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_outer(device):
    """Test outer product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand(10, dtype=torch.bfloat16, device=device)
    b = torch.rand(12, dtype=torch.bfloat16, device=device)
    output_nki = nki_outer(a, b)
    output_torch = torch.outer(a, b)
    print("Checking correctness of outer product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_dot(device):
    """Test dot product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand(10, dtype=torch.bfloat16, device=device)
    b = torch.rand(10, dtype=torch.bfloat16, device=device)
    output_nki = nki_dot(a, b)
    output_torch = torch.dot(a, b)
    print("Checking correctness of dot product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_vdot(device):
    """Test vdot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand(10, dtype=torch.bfloat16, device=device)
    b = torch.rand(10, dtype=torch.bfloat16, device=device)
    output_nki = nki_vdot(a, b)
    output_torch = torch.vdot(a, b)
    print("Checking correctness of vdot operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_cross(device):
    """Test cross product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.randn(3, dtype=torch.bfloat16, device=device)
    b = torch.randn(3, dtype=torch.bfloat16, device=device)
    output_nki = nki_cross(a, b)
    output_torch = torch.cross(a, b)
    print("Checking correctness of cross product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_matmul(device):
    """Test matrix multiplication (matmul) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((128, 32), dtype=torch.bfloat16, device=device)
    output_nki = nki_matmul(a, b)
    output_torch = torch.matmul(a, b)
    print("Checking correctness of matmul operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_mm(device):
    """Test matrix-matrix multiplication (mm) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((128, 32), dtype=torch.bfloat16, device=device)
    output_nki = nki_mm(a, b)
    output_torch = torch.mm(a, b)
    print("Checking correctness of mm operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_mv(device):
    """Test matrix-vector multiplication (mv) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((128,), dtype=torch.bfloat16, device=device)
    output_nki = nki_mv(a, b)
    output_torch = torch.mv(a, b)
    print("Checking correctness of mv operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_bmm(device):
    """Test batch matrix-matrix multiplication (bmm) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((10, 64, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((10, 128, 32), dtype=torch.bfloat16, device=device)
    output_nki = nki_bmm(a, b)
    output_torch = torch.bmm(a, b)
    print("Checking correctness of bmm operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_tensordot(device):
    """Test tensordot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((4, 5, 6), dtype=torch.bfloat16, device=device)
    b = torch.rand((6, 7, 8), dtype=torch.bfloat16, device=device)
    output_nki = nki_tensordot(a, b, dims=([2], [0]))
    output_torch = torch.tensordot(a, b, dims=([2], [0]))
    print("Checking correctness of tensordot operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_einsum(device):
    """Test einsum operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((128, 32), dtype=torch.bfloat16, device=device)
    equation = "ij,jk->ik"
    output_nki = nki_einsum(equation, a, b)
    output_torch = torch.einsum(equation, a, b)
    print("Checking correctness of einsum operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_kron(device):
    """Test Kronecker product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((3, 3), dtype=torch.bfloat16, device=device)
    b = torch.rand((3, 3), dtype=torch.bfloat16, device=device)
    output_nki = nki_kron(a, b)
    output_torch = torch.kron(a, b)
    print("Checking correctness of Kronecker product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_hadamard(device):
    """Test Hadamard (element-wise multiplication) operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    b = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_nki = nki_hadamard(a, b)
    output_torch = torch.mul(a, b)
    print("Checking correctness of Hadamard product operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_vecdot(device):
    """Test linalg_vecdot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    a = torch.rand((5, 10), dtype=torch.bfloat16, device=device)
    b = torch.rand((5, 10), dtype=torch.bfloat16, device=device)
    output_nki = nki_linalg_vecdot(a, b, dim=1)
    output_torch = torch.linalg.vecdot(a, b, dim=1)
    print("Checking correctness of linalg_vecdot operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_linalg_multi_dot(device):
    """Test linalg_multi_dot operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    A = torch.rand((10, 20), dtype=torch.bfloat16, device=device)
    B = torch.rand((20, 30), dtype=torch.bfloat16, device=device)
    C = torch.rand((30, 40), dtype=torch.bfloat16, device=device)
    matrices = [A, B, C]
    output_nki = nki_linalg_multi_dot(matrices)
    output_torch = torch.linalg.multi_dot(matrices)
    print("Checking correctness of linalg_multi_dot operation...")
    match = torch.allclose(output_torch, output_nki, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def main():
    device = xm.xla_device()
    test_torch_inner(device)
    test_torch_outer(device)
    test_torch_dot(device)
    test_torch_vdot(device)
    test_torch_cross(device)
    test_torch_matmul(device)
    test_torch_mm(device)
    test_torch_mv(device)
    test_torch_bmm(device)
    test_torch_tensordot(device)
    test_torch_einsum(device)
    test_torch_kron(device)
    test_torch_hadamard(device)
    test_torch_linalg_vecdot(device)
    test_torch_linalg_multi_dot(device)

if __name__ == "__main__":
    main()
