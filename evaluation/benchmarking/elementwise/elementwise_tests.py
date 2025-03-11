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

from elementwise_nki_kernels import nki_addition, nki_subtraction, nki_multiplication, nki_division, nki_abs, nki_exp, nki_log, nki_sqrt, nki_rsqrt, nki_pow, nki_sin, nki_cos, nki_tan, nki_asin, nki_acos, nki_atan, nki_sinh, nki_cosh, nki_tanh, nki_sigmoid, nki_relu, nki_threshold


def test_torch_addition(device):
    """Test elementwise addition between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)

    # Run NKI kernel
    output_small = nki_addition(lhs_small, rhs_small)

    # Run torch reference
    output_small_torch = torch.add(lhs_small, rhs_small)

    # Compare results
    print("Checking correctness of nki_addition")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0


def test_torch_subtraction(device):
    """Test elementwise subtraction between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    # Test the small workload with basic kernel
    lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)

    # Run NKI kernel
    output_small = nki_subtraction(lhs_small, rhs_small)

    # Run torch reference
    output_small_torch = torch.sub(lhs_small, rhs_small)

    # Compare results
    print("Checking correctness of nki_subtraction")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0


def test_torch_multiplication(device):
    """Test elementwise multiplication between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)
    
    # Run NKI kernel
    output_small = nki_multiplication(lhs_small, rhs_small)
    
    # Run torch reference
    output_small_torch = torch.mul(lhs_small, rhs_small)
    
    print("Checking correctness of nki_multiplication")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_division(device):
    """Test elementwise division between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    lhs_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    rhs_small = torch.rand((128, 512), dtype=torch.bfloat16, device=device)
    
    # Run NKI kernel
    output_small = nki_division(lhs_small, rhs_small)
    
    # Run torch reference
    output_small_torch = torch.div(lhs_small, rhs_small)
    
    print("Checking correctness of nki_division")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_abs(device):
    """Test elementwise absolute value between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = nki_abs(x_small)
    
    # Run torch reference
    output_small_torch = torch.abs(x_small)
    
    print("Checking correctness of nki_abs")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_exp(device):
    """Test elementwise exponential between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    
    # Run NKI kernel
    output_small = nki_exp(x_small)
    
    # Run torch reference
    output_small_torch = torch.exp(x_small)
    
    print("Checking correctness of nki_exp")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_log(device):
    """Test elementwise natural logarithm between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) + 0.1  # Ensure positive values
    
    # Run NKI kernel
    output_small = nki_log(x_small)
    
    # Run torch reference
    output_small_torch = torch.log(x_small)
    
    print("Checking correctness of nki_log")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_sqrt(device):
    """Test elementwise square root between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    
    # Run NKI kernel
    output_small = nki_sqrt(x_small)
    
    # Run torch reference
    output_small_torch = torch.sqrt(x_small)
    
    print("Checking correctness of nki_sqrt")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_rsqrt(device):
    """Test elementwise reciprocal square root between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    
    # Run NKI kernel
    output_small = nki_rsqrt(x_small)
    
    # Run torch reference
    output_small_torch = torch.rsqrt(x_small)
    
    print("Checking correctness of nki_rsqrt")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_pow(device):
    """Test elementwise power operation between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device)
    exponent = 2.0
    
    # Run NKI kernel
    output_small = nki_pow(x_small, exponent)
    
    # Run torch reference
    output_small_torch = torch.pow(x_small, exponent)
    
    print("Checking correctness of nki_pow")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_sin(device):
    """Test elementwise sine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * np.pi
    
    # Run NKI kernel
    output_small = nki_sin(x_small)
    
    # Run torch reference
    output_small_torch = torch.sin(x_small)
    
    print("Checking correctness of nki_sin")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_cos(device):
    """Test elementwise cosine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * np.pi
    
    # Run NKI kernel
    output_small = nki_cos(x_small)
    
    # Run torch reference
    output_small_torch = torch.cos(x_small)
    
    print("Checking correctness of nki_cos")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_tan(device):
    """Test elementwise tangent between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * np.pi
    
    # Run NKI kernel
    output_small = nki_tan(x_small)
    
    # Run torch reference
    output_small_torch = torch.tan(x_small)
    
    print("Checking correctness of nki_tan")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_asin(device):
    """
    Test elementwise inverse sine between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = nki_asin(x_small)
    
    # Run torch reference
    output_small_torch = torch.asin(x_small)
    
    print("Checking correctness of nki_asin")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_acos(device):
    """
    Test elementwise inverse cosine between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = nki_acos(x_small)
    
    # Run torch reference
    output_small_torch = torch.acos(x_small)
    
    print("Checking correctness of nki_acos")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_atan(device):
    """
    Test elementwise inverse tangent between NKI and PyTorch implementations.
    
    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1  # Values between -1 and 1
    
    # Run NKI kernel
    output_small = nki_atan(x_small)
    
    # Run torch reference
    output_small_torch = torch.atan(x_small)
    
    print("Checking correctness of nki_atan")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_sinh(device):
    """
    Test elementwise hyperbolic sine between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1
    
    # Run NKI kernel
    output_small = nki_sinh(x_small)
    
    # Run torch reference
    output_small_torch = torch.sinh(x_small)
    
    print("Checking correctness of nki_sinh")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_cosh(device):
    """
    Test elementwise hyperbolic cosine between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1
    
    # Run NKI kernel
    output_small = nki_cosh(x_small)
    
    # Run torch reference
    output_small_torch = torch.cosh(x_small)
    
    print("Checking correctness of nki_cosh")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_tanh(device):
    """
    Test elementwise hyperbolic tangent between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1
    
    # Run NKI kernel
    output_small = nki_tanh(x_small)
    
    # Run torch reference
    output_small_torch = torch.tanh(x_small)
    
    print("Checking correctness of nki_tanh")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_sigmoid(device):
    """
    Test elementwise sigmoid between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1
    
    # Run NKI kernel
    output_small = nki_sigmoid(x_small)
    
    # Run torch reference
    output_small_torch = torch.sigmoid(x_small)
    
    print("Checking correctness of nki_sigmoid")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_relu(device):
    """
    Test elementwise ReLU between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1
    
    # Run NKI kernel
    output_small = nki_relu(x_small)
    
    # Run torch reference
    output_small_torch = torch.relu(x_small)
    
    print("Checking correctness of nki_relu")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

def test_torch_threshold(device):
    """
    Test elementwise threshold between NKI and PyTorch implementations.

    Args:
        device: The device to run the test on (CPU/GPU/NPU)
    
    Returns:
        int: Returns 1 if NKI and PyTorch results match, 0 otherwise
    """
    x_small = torch.rand((64, 128), dtype=torch.bfloat16, device=device) * 2 - 1
    threshold = 0.5
    value = 0.0
    
    # Run NKI kernel
    output_small = nki_threshold(x_small, threshold, value)
    
    # Run torch reference
    output_small_torch = torch.threshold(x_small, threshold, value)
    
    print("Checking correctness of nki_threshold")
    match = torch.allclose(output_small_torch, output_small, atol=1e-4, rtol=1e-2)
    print("NKI and Torch match" if match else "NKI and Torch differ")
    return 1 if match else 0

if __name__ == "__main__":
    device = xm.xla_device()
    main()

def main():
    device = xm.xla_device()
    
    # Dictionary to store test results
    test_results = {}
    
    # Run all tests and store results
    test_results['addition'] = test_torch_addition(device)
    test_results['subtraction'] = test_torch_subtraction(device)
    test_results['multiplication'] = test_torch_multiplication(device)
    test_results['division'] = test_torch_division(device)
    test_results['abs'] = test_torch_abs(device)
    test_results['exp'] = test_torch_exp(device)
    test_results['log'] = test_torch_log(device)
    test_results['sqrt'] = test_torch_sqrt(device)
    test_results['rsqrt'] = test_torch_rsqrt(device)
    test_results['pow'] = test_torch_pow(device)
    test_results['sin'] = test_torch_sin(device)
    test_results['cos'] = test_torch_cos(device)
    test_results['tan'] = test_torch_tan(device)
    test_results['asin'] = test_torch_asin(device)
    test_results['acos'] = test_torch_acos(device)
    test_results['atan'] = test_torch_atan(device)
    test_results['sinh'] = test_torch_sinh(device)
    test_results['cosh'] = test_torch_cosh(device)
    test_results['tanh'] = test_torch_tanh(device)
    test_results['sigmoid'] = test_torch_sigmoid(device)
    test_results['relu'] = test_torch_relu(device)
    test_results['threshold'] = test_torch_threshold(device)
    
    # Print summary of results
    print("\nTest Results Summary:")
    print("-" * 40)
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Failed Tests: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.2f}%")
    
    # Print failed tests if any
    failed_tests = [test for test, result in test_results.items() if result == 0]
    if failed_tests:
        print("\nFailed Tests:")
        for test in failed_tests:
            print(f"- {test}")
    
    return test_results
