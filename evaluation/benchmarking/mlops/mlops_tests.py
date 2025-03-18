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

from mlops_nki_kernels import (
    mlops_gelu, mlops_elu, mlops_selu, mlops_leaky_relu, mlops_hardswish,
    mlops_mse_loss, mlops_l1_loss, mlops_cross_entropy, mlops_nll_loss,
    mlops_binary_cross_entropy, mlops_hinge_embedding_loss, mlops_kl_div,
    mlops_smooth_l1_loss, mlops_cosine_embedding_loss, mlops_triplet_margin_loss,
    mlops_batch_norm, mlops_layer_norm, mlops_group_norm, mlops_instance_norm,
    mlops_dropout, mlops_alpha_dropout, mlops_feature_alpha_dropout, mlops_softshrink,
    mlops_euclidean_dist, mlops_cosine_similarity, mlops_pairwise_distance,
    mlops_conv1d, mlops_conv2d, mlops_conv3d, mlops_conv_transpose2d,
    mlops_max_pool2d, mlops_avg_pool2d
)

def test_torch_gelu(device):
    """Test GELU activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_gelu(x)
    out_torch = torch.nn.functional.gelu(x)
    print("Checking correctness of GELU activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_elu(device):
    """Test ELU activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_elu(x, alpha=1.0)
    out_torch = torch.nn.functional.elu(x, alpha=1.0)
    print("Checking correctness of ELU activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_selu(device):
    """Test SELU activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_selu(x)
    out_torch = torch.nn.functional.selu(x)
    print("Checking correctness of SELU activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_leaky_relu(device):
    """Test Leaky ReLU activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_leaky_relu(x, negative_slope=0.01)
    out_torch = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
    print("Checking correctness of Leaky ReLU activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_hardswish(device):
    """Test Hard Swish activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_hardswish(x)
    out_torch = torch.nn.functional.hardswish(x)
    print("Checking correctness of Hard Swish activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_mse_loss(device):
    """Test MSE loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    loss_mlops = mlops_mse_loss(input, target)
    loss_torch = torch.nn.functional.mse_loss(input, target)
    print("Checking correctness of MSE loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_l1_loss(device):
    """Test L1 loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    loss_mlops = mlops_l1_loss(input, target)
    loss_torch = torch.nn.functional.l1_loss(input, target)
    print("Checking correctness of L1 loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_cross_entropy(device):
    """Test cross entropy loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 10), dtype=torch.bfloat16, device=device)
    target = torch.randint(0, 10, (64,), device=device)
    loss_mlops = mlops_cross_entropy(input, target)
    loss_torch = torch.nn.functional.cross_entropy(input, target)
    print("Checking correctness of cross entropy loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_nll_loss(device):
    """Test NLL loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 10), dtype=torch.bfloat16, device=device)
    target = torch.randint(0, 10, (64,), device=device)
    loss_mlops = mlops_nll_loss(input, target)
    loss_torch = torch.nn.functional.nll_loss(input, target)
    print("Checking correctness of NLL loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_binary_cross_entropy(device):
    """Test binary cross entropy loss between MLOps and PyTorch implementations."""
    input = torch.sigmoid(torch.randn((64, 128), dtype=torch.bfloat16, device=device))
    target = torch.randint(0, 2, (64, 128), device=device).to(torch.bfloat16)
    loss_mlops = mlops_binary_cross_entropy(input, target)
    loss_torch = torch.nn.functional.binary_cross_entropy(input, target)
    print("Checking correctness of binary cross entropy loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_hinge_embedding_loss(device):
    """Test hinge embedding loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randint(0, 2, (64, 128), device=device) * 2 - 1
    loss_mlops = mlops_hinge_embedding_loss(input, target, margin=1.0)
    loss_torch = torch.nn.functional.hinge_embedding_loss(input, target, margin=1.0)
    print("Checking correctness of hinge embedding loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_kl_div(device):
    """Test KL divergence loss between MLOps and PyTorch implementations."""
    input = torch.log_softmax(torch.randn((64, 10), dtype=torch.bfloat16, device=device), dim=1)
    target = torch.softmax(torch.randn((64, 10), dtype=torch.bfloat16, device=device), dim=1)
    loss_mlops = mlops_kl_div(input, target, log_target=False)
    loss_torch = torch.nn.functional.kl_div(input, target, log_target=False)
    print("Checking correctness of KL divergence loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_smooth_l1_loss(device):
    """Test Smooth L1 loss between MLOps and PyTorch implementations."""
    input = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    loss_mlops = mlops_smooth_l1_loss(input, target)
    loss_torch = torch.nn.functional.smooth_l1_loss(input, target)
    print("Checking correctness of Smooth L1 loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_cosine_embedding_loss(device):
    """Test cosine embedding loss between MLOps and PyTorch implementations."""
    input1 = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    input2 = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    target = torch.randint(0, 2, (64,), device=device) * 2 - 1
    loss_mlops = mlops_cosine_embedding_loss(input1, input2, target)
    loss_torch = torch.nn.functional.cosine_embedding_loss(input1, input2, target)
    print("Checking correctness of cosine embedding loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_triplet_margin_loss(device):
    """Test triplet margin loss between MLOps and PyTorch implementations."""
    anchor = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    positive = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    negative = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    loss_mlops = mlops_triplet_margin_loss(anchor, positive, negative)
    loss_torch = torch.nn.functional.triplet_margin_loss(anchor, positive, negative)
    print("Checking correctness of triplet margin loss...")
    match = torch.allclose(loss_torch, loss_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_batch_norm(device):
    """Test batch normalization between MLOps and PyTorch implementations."""
    x = torch.randn((16, 64, 32), dtype=torch.bfloat16, device=device)
    weight = torch.randn(64, dtype=torch.bfloat16, device=device)
    bias = torch.randn(64, dtype=torch.bfloat16, device=device)
    out_mlops = mlops_batch_norm(x, weight, bias, training=True)
    out_torch = torch.nn.functional.batch_norm(x, None, None, weight, bias, training=True)
    print("Checking correctness of batch normalization...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_layer_norm(device):
    """Test layer normalization between MLOps and PyTorch implementations."""
    x = torch.randn((16, 64, 32), dtype=torch.bfloat16, device=device)
    normalized_shape = (64, 32)
    weight = torch.randn(normalized_shape, dtype=torch.bfloat16, device=device)
    bias = torch.randn(normalized_shape, dtype=torch.bfloat16, device=device)
    out_mlops = mlops_layer_norm(x, normalized_shape, weight, bias)
    out_torch = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias)
    print("Checking correctness of layer normalization...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_group_norm(device):
    """Test group normalization between MLOps and PyTorch implementations."""
    x = torch.randn((16, 64, 32, 32), dtype=torch.bfloat16, device=device)
    weight = torch.randn(64, dtype=torch.bfloat16, device=device)
    bias = torch.randn(64, dtype=torch.bfloat16, device=device)
    out_mlops = mlops_group_norm(x, num_groups=8, weight=weight, bias=bias)
    out_torch = torch.nn.functional.group_norm(x, num_groups=8, weight=weight, bias=bias)
    print("Checking correctness of group normalization...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_instance_norm(device):
    """Test instance normalization between MLOps and PyTorch implementations."""
    x = torch.randn((16, 64, 32, 32), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_instance_norm(x, training=True)
    out_torch = torch.nn.functional.instance_norm(x, training=True)
    print("Checking correctness of instance normalization...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_dropout(device):
    """Test dropout between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_dropout(x, p=0.5, training=True)
    out_torch = torch.nn.functional.dropout(x, p=0.5, training=True)
    print("Checking correctness of dropout...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_alpha_dropout(device):
    """Test alpha dropout between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_alpha_dropout(x, p=0.5, training=True)
    out_torch = torch.nn.functional.alpha_dropout(x, p=0.5, training=True)
    print("Checking correctness of alpha dropout...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_feature_alpha_dropout(device):
    """Test feature alpha dropout between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_feature_alpha_dropout(x, p=0.5, training=True)
    out_torch = torch.nn.functional.feature_alpha_dropout(x, p=0.5, training=True)
    print("Checking correctness of feature alpha dropout...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_softshrink(device):
    """Test softshrink activation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_softshrink(x, lambd=0.5)
    out_torch = torch.nn.functional.softshrink(x, lambd=0.5)
    print("Checking correctness of softshrink activation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_euclidean_dist(device):
    """Test Euclidean distance computation between MLOps and a reference implementation."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    y = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    dist_mlops = mlops_euclidean_dist(x, y)
    dist_ref = torch.norm(x - y, dim=1)
    print("Checking correctness of Euclidean distance computation...")
    match = torch.allclose(dist_ref, dist_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and reference match!" if match else "MLOps and reference differ")
    return 1 if match else 0

def test_torch_cosine_similarity(device):
    """Test cosine similarity computation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    y = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    sim_mlops = mlops_cosine_similarity(x, y, dim=1)
    sim_torch = torch.nn.functional.cosine_similarity(x, y, dim=1)
    print("Checking correctness of cosine similarity computation...")
    match = torch.allclose(sim_torch, sim_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_pairwise_distance(device):
    """Test pairwise distance computation between MLOps and PyTorch implementations."""
    x = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    y = torch.randn((64, 128), dtype=torch.bfloat16, device=device)
    dist_mlops = mlops_pairwise_distance(x, y)
    dist_torch = torch.nn.functional.pairwise_distance(x, y)
    print("Checking correctness of pairwise distance computation...")
    match = torch.allclose(dist_torch, dist_mlops, atol=1e-3, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_conv1d(device):
    """Test 1D convolution between MLOps and PyTorch implementations."""
    x = torch.randn((8, 3, 50), dtype=torch.bfloat16, device=device)
    weight = torch.randn((6, 3, 5), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_conv1d(x, weight, bias=None, stride=1, padding=2)
    out_torch = torch.nn.functional.conv1d(x, weight, bias=None, stride=1, padding=2)
    print("Checking correctness of conv1d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_conv2d(device):
    """Test 2D convolution between MLOps and PyTorch implementations."""
    x = torch.randn((8, 3, 32, 32), dtype=torch.bfloat16, device=device)
    weight = torch.randn((6, 3, 5, 5), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_conv2d(x, weight, bias=None, stride=1, padding=2)
    out_torch = torch.nn.functional.conv2d(x, weight, bias=None, stride=1, padding=2)
    print("Checking correctness of conv2d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_conv3d(device):
    """Test 3D convolution between MLOps and PyTorch implementations."""
    x = torch.randn((4, 3, 16, 16, 16), dtype=torch.bfloat16, device=device)
    weight = torch.randn((6, 3, 3, 3, 3), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_conv3d(x, weight, bias=None, stride=1, padding=1)
    out_torch = torch.nn.functional.conv3d(x, weight, bias=None, stride=1, padding=1)
    print("Checking correctness of conv3d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_conv_transpose2d(device):
    """Test transposed 2D convolution between MLOps and PyTorch implementations."""
    x = torch.randn((8, 6, 32, 32), dtype=torch.bfloat16, device=device)
    weight = torch.randn((3, 6, 5, 5), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_conv_transpose2d(x, weight, bias=None, stride=1, padding=2)
    out_torch = torch.nn.functional.conv_transpose2d(x, weight, bias=None, stride=1, padding=2)
    print("Checking correctness of conv_transpose2d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_max_pool2d(device):
    """Test 2D max pooling between MLOps and PyTorch implementations."""
    x = torch.randn((8, 3, 32, 32), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_max_pool2d(x, kernel_size=2, stride=2)
    out_torch = torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)
    print("Checking correctness of max_pool2d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def test_torch_avg_pool2d(device):
    """Test 2D average pooling between MLOps and PyTorch implementations."""
    x = torch.randn((8, 3, 32, 32), dtype=torch.bfloat16, device=device)
    out_mlops = mlops_avg_pool2d(x, kernel_size=2, stride=2)
    out_torch = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    print("Checking correctness of avg_pool2d operation...")
    match = torch.allclose(out_torch, out_mlops, atol=1e-2, rtol=1e-2)
    print("MLOps and Torch match!" if match else "MLOps and Torch differ")
    return 1 if match else 0

def main():
    device = xm.xla_device()
    
    # Dictionary to store test results
    test_results = {}
    
    # Run all tests and store results
    test_results['gelu'] = test_torch_gelu(device)
    test_results['elu'] = test_torch_elu(device)
    test_results['selu'] = test_torch_selu(device)
    test_results['leaky_relu'] = test_torch_leaky_relu(device)
    test_results['hardswish'] = test_torch_hardswish(device)
    test_results['mse_loss'] = test_torch_mse_loss(device)
    test_results['l1_loss'] = test_torch_l1_loss(device)
    test_results['cross_entropy'] = test_torch_cross_entropy(device)
    test_results['nll_loss'] = test_torch_nll_loss(device)
    test_results['binary_cross_entropy'] = test_torch_binary_cross_entropy(device)
    test_results['hinge_embedding_loss'] = test_torch_hinge_embedding_loss(device)
    test_results['kl_div'] = test_torch_kl_div(device)
    test_results['smooth_l1_loss'] = test_torch_smooth_l1_loss(device)
    test_results['cosine_embedding_loss'] = test_torch_cosine_embedding_loss(device)
    test_results['triplet_margin_loss'] = test_torch_triplet_margin_loss(device)
    test_results['batch_norm'] = test_torch_batch_norm(device)
    test_results['layer_norm'] = test_torch_layer_norm(device)
    test_results['group_norm'] = test_torch_group_norm(device)
    test_results['instance_norm'] = test_torch_instance_norm(device)
    test_results['dropout'] = test_torch_dropout(device)
    test_results['alpha_dropout'] = test_torch_alpha_dropout(device)
    test_results['feature_alpha_dropout'] = test_torch_feature_alpha_dropout(device)
    test_results['softshrink'] = test_torch_softshrink(device)
    test_results['euclidean_dist'] = test_torch_euclidean_dist(device)
    test_results['cosine_similarity'] = test_torch_cosine_similarity(device)
    test_results['pairwise_distance'] = test_torch_pairwise_distance(device)
    test_results['conv1d'] = test_torch_conv1d(device)
    test_results['conv2d'] = test_torch_conv2d(device)
    test_results['conv3d'] = test_torch_conv3d(device)
    test_results['conv_transpose2d'] = test_torch_conv_transpose2d(device)
    test_results['max_pool2d'] = test_torch_max_pool2d(device)
    test_results['avg_pool2d'] = test_torch_avg_pool2d(device)
    
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

if __name__ == "__main__":
    main()
