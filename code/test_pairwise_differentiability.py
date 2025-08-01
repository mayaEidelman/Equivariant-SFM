#!/usr/bin/env python3
"""
Test script to verify the differentiability of the pairwise consistency loss
and its handling of missing pairwise data.
"""

import torch
import numpy as np
from pyhocon import ConfigFactory
import os
import sys

# Add the code directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loss_functions import PairwiseConsistencyLoss, CombinedLoss, ESFMLoss
from datasets.SceneData import create_scene_data


def test_differentiability():
    """Test that the pairwise loss is differentiable."""
    print("Testing differentiability of pairwise loss...")
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Pairwise.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    # Create dummy predictions that require gradients
    n_cameras = data.y.shape[0]
    device = data.y.device
    
    # Create trainable camera matrices
    Ps_pred = torch.eye(4, device=device, requires_grad=True).unsqueeze(0).expand(n_cameras, 4, 4)
    Ps_pred[:, :3, :3] = torch.eye(3, device=device, requires_grad=True).unsqueeze(0).expand(n_cameras, 3, 3)
    Ps_pred[:, :3, 3] = torch.randn(n_cameras, 3, device=device, requires_grad=True) * 0.1
    
    # Create dummy 3D points
    pts3D = torch.randn(4, data.M.shape[1], device=device, requires_grad=True)
    pts3D[3, :] = 1.0
    
    pred_cam = {
        "Ps": Ps_pred,
        "Ps_norm": Ps_pred,
        "pts3D": pts3D
    }
    
    # Test pairwise loss
    pairwise_loss = PairwiseConsistencyLoss(conf)
    loss_value = pairwise_loss(pred_cam, data)
    
    print(f"Loss value: {loss_value.item():.6f}")
    print(f"Loss requires grad: {loss_value.requires_grad}")
    
    # Test backward pass
    try:
        loss_value.backward()
        print("✓ Backward pass successful - loss is differentiable!")
        
        # Check gradients
        grad_norm = Ps_pred.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.6f}")
        
        if grad_norm > 0:
            print("✓ Gradients are non-zero - loss is properly differentiable!")
        else:
            print("⚠ Warning: Gradients are zero - this might indicate an issue")
            
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return False
    
    return True


def test_missing_pairwise_data():
    """Test handling of missing pairwise data."""
    print("\nTesting handling of missing pairwise data...")
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Pairwise.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data without pairwise computation
    conf["dataset"]["compute_pairwise"] = False
    data = create_scene_data(conf)
    
    # Create dummy predictions
    n_cameras = data.y.shape[0]
    device = data.y.device
    
    Ps_pred = torch.eye(4, device=device, requires_grad=True).unsqueeze(0).expand(n_cameras, 4, 4)
    Ps_pred[:, :3, :3] = torch.eye(3, device=device, requires_grad=True).unsqueeze(0).expand(n_cameras, 3, 3)
    Ps_pred[:, :3, 3] = torch.randn(n_cameras, 3, device=device, requires_grad=True) * 0.1
    
    pts3D = torch.randn(4, data.M.shape[1], device=device, requires_grad=True)
    pts3D[3, :] = 1.0
    
    pred_cam = {
        "Ps": Ps_pred,
        "Ps_norm": Ps_pred,
        "pts3D": pts3D
    }
    
    # Check if pairwise data is missing
    has_relative_poses = hasattr(data, 'relative_poses') and len(data.relative_poses) > 0
    has_matches = hasattr(data, 'matches') and len(data.matches) > 0
    
    print(f"Has relative poses: {has_relative_poses}")
    print(f"Has matches: {has_matches}")
    
    # Test pairwise loss with missing data
    pairwise_loss = PairwiseConsistencyLoss(conf)
    loss_value = pairwise_loss(pred_cam, data)
    
    print(f"Loss value with missing data: {loss_value.item():.6f}")
    print(f"Loss requires grad: {loss_value.requires_grad}")
    
    # Test backward pass
    try:
        loss_value.backward()
        print("✓ Backward pass successful with missing data!")
        return True
    except Exception as e:
        print(f"✗ Backward pass failed with missing data: {e}")
        return False


def test_combined_loss():
    """Test the combined loss function."""
    print("\nTesting combined loss function...")
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Pairwise.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Test with pairwise data
    conf["dataset"]["compute_pairwise"] = True
    data = create_scene_data(conf)
    
    n_cameras = data.y.shape[0]
    device = data.y.device
    
    Ps_pred = torch.eye(4, device=device, requires_grad=True).unsqueeze(0).expand(n_cameras, 4, 4)
    Ps_pred[:, :3, :3] = torch.eye(3, device=device, requires_grad=True).unsqueeze(0).expand(n_cameras, 3, 3)
    Ps_pred[:, :3, 3] = torch.randn(n_cameras, 3, device=device, requires_grad=True) * 0.1
    
    pts3D = torch.randn(4, data.M.shape[1], device=device, requires_grad=True)
    pts3D[3, :] = 1.0
    
    pred_cam = {
        "Ps": Ps_pred,
        "Ps_norm": Ps_pred,
        "pts3D": pts3D
    }
    
    # Test individual losses
    esfm_loss = ESFMLoss(conf)
    pairwise_loss = PairwiseConsistencyLoss(conf)
    combined_loss = CombinedLoss(conf)
    
    esfm_value = esfm_loss(pred_cam, data)
    pairwise_value = pairwise_loss(pred_cam, data)
    combined_value = combined_loss(pred_cam, data)
    
    print(f"ESFM Loss: {esfm_value.item():.6f}")
    print(f"Pairwise Loss: {pairwise_value.item():.6f}")
    print(f"Combined Loss: {combined_value.item():.6f}")
    
    # Test backward pass
    try:
        combined_value.backward()
        print("✓ Combined loss backward pass successful!")
        return True
    except Exception as e:
        print(f"✗ Combined loss backward pass failed: {e}")
        return False


def test_weight_handling():
    """Test that weights are properly handled when pairwise data is missing."""
    print("\nTesting weight handling...")
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Pairwise.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Test with different pairwise weight settings
    test_weights = [0.0, 0.5, 1.0]
    
    for weight in test_weights:
        conf["loss"]["pairwise_weight"] = weight
        conf["dataset"]["compute_pairwise"] = False  # No pairwise data
        
        data = create_scene_data(conf)
        
        n_cameras = data.y.shape[0]
        device = data.y.device
        
        Ps_pred = torch.eye(4, device=device, requires_grad=True).unsqueeze(0).expand(n_cameras, 4, 4)
        Ps_pred[:, :3, :3] = torch.eye(3, device=device, requires_grad=True).unsqueeze(0).expand(n_cameras, 3, 3)
        Ps_pred[:, :3, 3] = torch.randn(n_cameras, 3, device=device, requires_grad=True) * 0.1
        
        pts3D = torch.randn(4, data.M.shape[1], device=device, requires_grad=True)
        pts3D[3, :] = 1.0
        
        pred_cam = {
            "Ps": Ps_pred,
            "Ps_norm": Ps_pred,
            "pts3D": pts3D
        }
        
        combined_loss = CombinedLoss(conf)
        loss_value = combined_loss(pred_cam, data)
        
        print(f"Pairwise weight: {weight}, Combined loss: {loss_value.item():.6f}")
        
        # The loss should be the same regardless of pairwise weight when no pairwise data is available
        if weight == 0.0:
            reference_loss = loss_value.item()
        else:
            assert abs(loss_value.item() - reference_loss) < 1e-6, f"Loss should be same for weight {weight}"
    
    print("✓ Weight handling test passed!")


if __name__ == "__main__":
    print("Testing Pairwise Loss Differentiability and Robustness")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Differentiability
    if test_differentiability():
        tests_passed += 1
    
    # Test 2: Missing pairwise data
    if test_missing_pairwise_data():
        tests_passed += 1
    
    # Test 3: Combined loss
    if test_combined_loss():
        tests_passed += 1
    
    # Test 4: Weight handling
    try:
        test_weight_handling()
        tests_passed += 1
    except Exception as e:
        print(f"✗ Weight handling test failed: {e}")
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! The pairwise loss is working correctly.")
    else:
        print("✗ Some tests failed. Please check the implementation.") 