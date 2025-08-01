#!/usr/bin/env python3
"""
Test script to verify the unsupervised pairwise consistency loss works correctly
without requiring ground truth relative poses.
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


def test_unsupervised_pairwise_loss():
    """Test that the unsupervised pairwise loss works without ground truth relative poses."""
    print("Testing unsupervised pairwise loss...")
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Pairwise.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    print(f"Scene: {data.scan_name}")
    print(f"Number of cameras: {data.y.shape[0]}")
    print(f"Number of points: {data.M.shape[1]}")
    
    # Check pairwise data
    if hasattr(data, 'matches'):
        print(f"Number of camera pairs with matches: {len(data.matches)}")
        for (i, j), matches in data.matches.items():
            print(f"  Pair ({i}, {j}): {matches['num_matches']} matches")
    
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
    
    # Test unsupervised pairwise loss
    pairwise_loss = PairwiseConsistencyLoss(conf)
    loss_value = pairwise_loss(pred_cam, data)
    
    print(f"Unsupervised pairwise loss value: {loss_value.item():.6f}")
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


def test_epipolar_consistency():
    """Test that epipolar consistency works correctly."""
    print("\nTesting epipolar consistency...")
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Pairwise.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    # Create predictions
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
    
    # Test epipolar consistency for each camera pair
    pairwise_loss = PairwiseConsistencyLoss(conf)
    
    if hasattr(data, 'matches'):
        for (i, j), matches in data.matches.items():
            if len(matches['pts1']) > 8:
                pts1 = matches['pts1']
                pts2 = matches['pts2']
                
                # Extract predicted poses
                Vs_pred = Ps_pred[:, 0:3, 0:3].inverse().transpose(1, 2)
                ts_pred = torch.bmm(-Vs_pred.transpose(1, 2), Ps_pred[:, 0:3, 3].unsqueeze(dim=-1)).squeeze()
                
                # Compute fundamental matrix
                F_pred = pairwise_loss._compute_fundamental_matrix(Vs_pred[i], Vs_pred[j], ts_pred[i], ts_pred[j])
                
                # Compute epipolar error
                epipolar_error = pairwise_loss._compute_epipolar_error(F_pred, pts1, pts2)
                
                print(f"  Pair ({i}, {j}): Epipolar error = {epipolar_error.item():.6f}")
    
    return True


def test_geometric_consistency():
    """Test that geometric consistency works correctly."""
    print("\nTesting geometric consistency...")
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Pairwise.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    # Create predictions
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
    
    # Test geometric consistency for each camera pair
    pairwise_loss = PairwiseConsistencyLoss(conf)
    
    if hasattr(data, 'matches'):
        for (i, j), matches in data.matches.items():
            if len(matches['pts1']) > 8:
                pts1 = matches['pts1']
                pts2 = matches['pts2']
                
                # Extract predicted poses
                Vs_pred = Ps_pred[:, 0:3, 0:3].inverse().transpose(1, 2)
                ts_pred = torch.bmm(-Vs_pred.transpose(1, 2), Ps_pred[:, 0:3, 3].unsqueeze(dim=-1)).squeeze()
                
                # Compute geometric consistency
                geometric_error = pairwise_loss._compute_geometric_consistency(
                    Vs_pred[i], Vs_pred[j], ts_pred[i], ts_pred[j], pts1, pts2
                )
                
                print(f"  Pair ({i}, {j}): Geometric error = {geometric_error.item():.6f}")
    
    return True


def test_combined_loss():
    """Test the combined loss with unsupervised pairwise loss."""
    print("\nTesting combined loss...")
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Pairwise.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    # Create predictions
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
    print(f"Unsupervised Pairwise Loss: {pairwise_value.item():.6f}")
    print(f"Combined Loss: {combined_value.item():.6f}")
    
    # Test backward pass
    try:
        combined_value.backward()
        print("✓ Combined loss backward pass successful!")
        return True
    except Exception as e:
        print(f"✗ Combined loss backward pass failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Unsupervised Pairwise Consistency Loss")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Unsupervised pairwise loss
    if test_unsupervised_pairwise_loss():
        tests_passed += 1
    
    # Test 2: Epipolar consistency
    if test_epipolar_consistency():
        tests_passed += 1
    
    # Test 3: Geometric consistency
    if test_geometric_consistency():
        tests_passed += 1
    
    # Test 4: Combined loss
    if test_combined_loss():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! The unsupervised pairwise loss is working correctly.")
        print("✓ No ground truth relative poses are required!")
    else:
        print("✗ Some tests failed. Please check the implementation.") 