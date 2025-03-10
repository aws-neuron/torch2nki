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

from multielement_nki_kernels import (
    nki_softmax, nki_log_softmax, nki_max, nki_min, nki_sum, nki_mean,
    nki_var, nki_std, nki_norm,
    nki_cumsum, nki_cumprod, nki_prod, nki_round, nki_floor, nki_ceil,
    nki_trunc, nki_sign, nki_where, nki_eq, nki_ne, nki_gt, nki_lt,
    nki_clamp, nki_sort, nki_topk, nki_kthvalue, nki_median, nki_mode,
    nki_percentile, nki_logsumexp, nki_amax, nki_amin, nki_all, nki_any,
    nki_bincount, nki_unique, nki_unique_consecutive
)

def test_torch_softmax(device):
    """Test softmax operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test with a small workload
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    
    # Run NKI kernel
    output_small = nki_softmax(x_small)
    
    # Run torch reference
    output_small_torch = torch.softmax(x_small, dim=-1)
    
    # Compare results
    print("Checking correctness of softmax operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_log_softmax(device):
    """Test log softmax operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_log_softmax(x_small)
    output_small_torch = torch.log_softmax(x_small, dim=-1)
    print("Checking correctness of log softmax operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_max(device):
    """Test element-wise maximum operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    y_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_max(x_small, y_small)
    output_small_torch = torch.max(x_small, y_small)
    print("Checking correctness of element-wise maximum operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_min(device):
    """Test element-wise minimum operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    y_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_min(x_small, y_small)
    output_small_torch = torch.min(x_small, y_small)
    print("Checking correctness of element-wise minimum operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_sum(device):
    """Test summation operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_sum(x_small)
    output_small_torch = torch.sum(x_small)
    print("Checking correctness of summation operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_mean(device):
    """Test mean operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_mean(x_small)
    output_small_torch = torch.mean(x_small)
    print("Checking correctness of mean operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_var(device):
    """Test variance operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_var(x_small)
    output_small_torch = torch.var(x_small)
    print("Checking correctness of variance operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_std(device):
    """Test standard deviation operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_std(x_small)
    output_small_torch = torch.std(x_small)
    print("Checking correctness of standard deviation operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_norm(device):
    """Test norm operation between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_norm(x_small)
    output_small_torch = torch.norm(x_small)
    print("Checking correctness of norm operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_cumsum(device):
    """Test cumulative sum operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_cumsum(x_small, dim=-1)
    output_small_torch = torch.cumsum(x_small, dim=-1)
    print("Checking correctness of cumulative sum operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_cumprod(device):
    """Test cumulative product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Add a small constant to avoid multiplying by zero
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) + 0.1
    output_small = nki_cumprod(x_small, dim=-1)
    output_small_torch = torch.cumprod(x_small, dim=-1)
    print("Checking correctness of cumulative product operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_prod(device):
    """Test product operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) + 0.1
    output_small = nki_prod(x_small)
    output_small_torch = torch.prod(x_small)
    print("Checking correctness of product operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_round(device):
    """Test rounding operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 10 - 5
    output_small = nki_round(x_small)
    output_small_torch = torch.round(x_small)
    print("Checking correctness of rounding operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_floor(device):
    """Test floor operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 10 - 5
    output_small = nki_floor(x_small)
    output_small_torch = torch.floor(x_small)
    print("Checking correctness of floor operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_ceil(device):
    """Test ceil operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 10 - 5
    output_small = nki_ceil(x_small)
    output_small_torch = torch.ceil(x_small)
    print("Checking correctness of ceil operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_trunc(device):
    """Test truncation operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 10 - 5
    output_small = nki_trunc(x_small)
    output_small_torch = torch.trunc(x_small)
    print("Checking correctness of truncation operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_sign(device):
    """Test sign operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_sign(x_small)
    output_small_torch = torch.sign(x_small)
    print("Checking correctness of sign operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_where(device):
    """Test element-wise conditional selection (where) between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    condition = x_small > 0.5
    other = torch.zeros_like(x_small)
    output_small = nki_where(condition, x_small, other)
    output_small_torch = torch.where(condition, x_small, other)
    print("Checking correctness of element-wise conditional selection (where)...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_eq(device):
    """Test element-wise equality comparison between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    y_small = x_small.clone()
    output_small = nki_eq(x_small, y_small)
    output_small_torch = torch.eq(x_small, y_small)
    print("Checking correctness of element-wise equality comparison...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_ne(device):
    """Test element-wise inequality comparison between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    y_small = x_small + 1.0  # ensure differences
    output_small = nki_ne(x_small, y_small)
    output_small_torch = torch.ne(x_small, y_small)
    print("Checking correctness of element-wise inequality comparison...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_gt(device):
    """Test element-wise greater than comparison between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    y_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_gt(x_small, y_small)
    output_small_torch = torch.gt(x_small, y_small)
    print("Checking correctness of element-wise greater than comparison...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_lt(device):
    """Test element-wise less than comparison between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    y_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_lt(x_small, y_small)
    output_small_torch = torch.lt(x_small, y_small)
    print("Checking correctness of element-wise less than comparison...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_clamp(device):
    """Test clamping operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2  # values in [0,2]
    output_small = nki_clamp(x_small, min=0.5, max=1.5)
    output_small_torch = torch.clamp(x_small, min=0.5, max=1.5)
    print("Checking correctness of clamping operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_sort(device):
    """Test sort operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    values_small, indices_small = nki_sort(x_small, dim=-1)
    output_small_torch = torch.sort(x_small, dim=-1)
    values_small_torch, indices_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of sort operation...")
    match = torch.allclose(values_small_torch, values_small, atol=1e-4, rtol=1e-2) and torch.equal(indices_small_torch, indices_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_topk(device):
    """Test top-k operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    k = 5
    values_small, indices_small = nki_topk(x_small, k=k, dim=-1)
    output_small_torch = torch.topk(x_small, k=k, dim=-1)
    values_small_torch, indices_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of top-k operation...")
    match = torch.allclose(values_small_torch, values_small, atol=1e-4, rtol=1e-2) and torch.equal(indices_small_torch, indices_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_kthvalue(device):
    """Test kth value operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    k = 10
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    value_small, index_small = nki_kthvalue(x_small, k=k, dim=-1)
    output_small_torch = torch.kthvalue(x_small, k=k, dim=-1)
    value_small_torch, index_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of kth value operation...")
    match = torch.allclose(value_small_torch, value_small, atol=1e-4, rtol=1e-2) and torch.equal(index_small_torch, index_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_median(device):
    """Test median operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    value_small, index_small = nki_median(x_small, dim=-1)
    output_small_torch = torch.median(x_small, dim=-1)
    value_small_torch, index_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of median operation...")
    match = torch.allclose(value_small_torch, value_small, atol=1e-4, rtol=1e-2) and torch.equal(index_small_torch, index_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_mode(device):
    """Test mode operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Use an integer tensor with limited range to obtain meaningful mode
    x_small = torch.randint(0, 5, (64, 128), device=device)
    value_small, index_small = nki_mode(x_small, dim=-1)
    output_small_torch = torch.mode(x_small, dim=-1)
    value_small_torch, index_small_torch = output_small_torch.values, output_small_torch.indices
    print("Checking correctness of mode operation...")
    match = torch.equal(value_small_torch, value_small) and torch.equal(index_small_torch, index_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_percentile(device):
    """Test percentile operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Use 50th percentile as a test (equivalent to median)
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_percentile(x_small, q=50, dim=-1)
    output_small_torch = torch.percentile(x_small, q=50, dim=-1)
    print("Checking correctness of percentile operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_logsumexp(device):
    """Test logsumexp operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_logsumexp(x_small, dim=-1)
    output_small_torch = torch.logsumexp(x_small, dim=-1)
    print("Checking correctness of logsumexp operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_amax(device):
    """Test amax operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_amax(x_small, dim=-1)
    output_small_torch = torch.amax(x_small, dim=-1)
    print("Checking correctness of amax operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_amin(device):
    """Test amin operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    output_small = nki_amin(x_small, dim=-1)
    output_small_torch = torch.amin(x_small, dim=-1)
    print("Checking correctness of amin operation...")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_all(device):
    """Test all operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    condition = x_small > 0.5
    output_small = nki_all(condition)
    output_small_torch = torch.all(condition)
    print("Checking correctness of all operation...")
    match = output_small_torch.item() == output_small.item()
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_any(device):
    """Test any operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    condition = x_small > 0.5
    output_small = nki_any(condition)
    output_small_torch = torch.any(condition)
    print("Checking correctness of any operation...")
    match = output_small_torch.item() == output_small.item()
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_bincount(device):
    """Test bincount operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.randint(0, 10, (100,), device=device)
    output_small = nki_bincount(x_small)
    output_small_torch = torch.bincount(x_small)
    print("Checking correctness of bincount operation...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_unique(device):
    """Test unique operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.randint(0, 10, (100,), device=device)
    output_small = nki_unique(x_small, return_counts=True)
    output_small_torch = torch.unique(x_small, return_counts=True)
    print("Checking correctness of unique operation...")
    match = torch.equal(output_small_torch[0], output_small[0]) and torch.equal(output_small_torch[1], output_small[1])
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_unique_consecutive(device):
    """Test unique consecutive operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
        
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.tensor([1, 1, 2, 2, 3, 3, 2, 2, 1, 1], device=device)
    output_small = nki_unique_consecutive(x_small)
    output_small_torch = torch.unique_consecutive(x_small)
    print("Checking correctness of unique consecutive operation...")
    match = torch.equal(output_small_torch, output_small)
    print("NKI and Torch match!" if match else "NKI and Torch differ")
    return 1 if match else 0


if __name__ == "__main__":
    device = xm.xla_device()
    main()


def main():
    device = xm.xla_device()
    test_torch_softmax(device)
    test_torch_log_softmax(device)
    test_torch_max(device)
    test_torch_min(device)
    test_torch_sum(device)
    test_torch_mean(device)
    test_torch_var(device)
    test_torch_std(device)
    test_torch_norm(device)
    test_torch_cumsum(device)
    test_torch_cumprod(device)
    test_torch_prod(device)
    test_torch_round(device)
    test_torch_floor(device)
    test_torch_ceil(device)
    test_torch_trunc(device)
    test_torch_sign(device)
    test_torch_where(device)
    test_torch_eq(device)
    test_torch_ne(device)
    test_torch_gt(device)
    test_torch_lt(device)
    test_torch_clamp(device)
    test_torch_sort(device)
    test_torch_topk(device)
    test_torch_kthvalue(device)
    test_torch_median(device)
    test_torch_mode(device)
    test_torch_percentile(device)
    test_torch_logsumexp(device)
    test_torch_amax(device)
    test_torch_amin(device)
    test_torch_all(device)
    test_torch_any(device)
    test_torch_bincount(device)
    test_torch_unique(device)
    test_torch_unique_consecutive(device)

