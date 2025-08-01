#!/usr/bin/env python3
"""
Example script demonstrating the use of pairwise consistency loss for ESfM.

This script shows how to:
1. Load scene data with pairwise information
2. Use the combined loss function with pairwise consistency
3. Evaluate pairwise consistency metrics
"""

import torch
import numpy as np
from pyhocon import ConfigFactory
import os
import sys

# Add the code directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.SceneData import create_scene_data
from loss_functions import CombinedLoss, PairwiseConsistencyLoss
from utils import pairwise_utils
from models.SetOfSet import SetOfSetNet


def load_config(config_path):
    """Load configuration from file."""
    return ConfigFactory.parse_file(config_path)


def create_model(conf):
    """Create the ESfM model."""
    model_type = conf.get_string('model.type')
    num_features = conf.get_int('model.num_features')
    num_blocks = conf.get_int('model.num_blocks')
    block_size = conf.get_int('model.block_size')
    use_skip = conf.get_bool('model.use_skip')
    multires = conf.get_int('model.multires')
    
    model = SetOfSetNet(
        num_features=num_features,
        num_blocks=num_blocks,
        block_size=block_size,
        use_skip=use_skip,
        multires=multires
    )
    
    return model


def test_pairwise_loss():
    """Test the pairwise consistency loss on a sample scene."""
    
    # Load configuration
    conf = load_config('confs/Learning_Euc_Pairwise.conf')
    
    # Create a sample scene
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]  # Use first training scene
    data = create_scene_data(conf)
    
    print(f"Scene: {data.scan_name}")
    print(f"Number of cameras: {data.y.shape[0]}")
    print(f"Number of points: {data.M.shape[1]}")
    
    # Check pairwise data
    if hasattr(data, 'matches'):
        print(f"Number of camera pairs with matches: {len(data.matches)}")
        for (i, j), matches in data.matches.items():
            print(f"  Pair ({i}, {j}): {matches['num_matches']} matches")
    
    if hasattr(data, 'relative_poses'):
        print(f"Number of valid relative poses: {len(data.relative_poses)}")
        for (i, j), rel_pose in data.relative_poses.items():
            print(f"  Pair ({i}, {j}): error={rel_pose['error']:.4f}")
    
    # Create model and generate dummy predictions
    model = create_model(conf)
    model.eval()
    
    with torch.no_grad():
        # Generate dummy predictions (in practice, this would come from the model)
        n_cameras = data.y.shape[0]
        device = data.y.device
        
        # Create dummy camera matrices
        Ps_pred = torch.eye(4, device=device).unsqueeze(0).expand(n_cameras, 4, 4)
        Ps_pred[:, :3, :3] = torch.eye(3, device=device).unsqueeze(0).expand(n_cameras, 3, 3)
        Ps_pred[:, :3, 3] = torch.randn(n_cameras, 3, device=device) * 0.1
        
        # Create dummy 3D points
        pts3D = torch.randn(4, data.M.shape[1], device=device)
        pts3D[3, :] = 1.0  # Homogeneous coordinates
        
        pred_cam = {
            "Ps": Ps_pred,
            "Ps_norm": Ps_pred,
            "pts3D": pts3D
        }
    
    # Test pairwise consistency loss
    print("\nTesting PairwiseConsistencyLoss...")
    pairwise_loss = PairwiseConsistencyLoss(conf)
    loss_value = pairwise_loss(pred_cam, data)
    print(f"Pairwise consistency loss: {loss_value:.6f}")
    
    # Test combined loss
    print("\nTesting CombinedLoss...")
    combined_loss = CombinedLoss(conf)
    total_loss = combined_loss(pred_cam, data)
    print(f"Combined loss: {total_loss:.6f}")
    
    # Evaluate pairwise consistency metrics
    print("\nEvaluating pairwise consistency metrics...")
    metrics = pairwise_utils.evaluate_pairwise_consistency(data, pred_cam)
    print(f"Mean epipolar error: {metrics['mean_epipolar_error']:.6f}")
    print(f"Mean rotation error: {metrics['mean_rotation_error']:.6f}")
    print(f"Mean translation error: {metrics['mean_translation_error']:.6f}")
    print(f"Number of pairs: {metrics['num_pairs']}")
    
    return metrics


def compare_losses():
    """Compare different loss functions on the same data."""
    
    conf = load_config('confs/Learning_Euc_Pairwise.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    data = create_scene_data(conf)
    
    # Create dummy predictions
    n_cameras = data.y.shape[0]
    device = data.y.device
    
    Ps_pred = torch.eye(4, device=device).unsqueeze(0).expand(n_cameras, 4, 4)
    Ps_pred[:, :3, :3] = torch.eye(3, device=device).unsqueeze(0).expand(n_cameras, 3, 3)
    Ps_pred[:, :3, 3] = torch.randn(n_cameras, 3, device=device) * 0.1
    
    pts3D = torch.randn(4, data.M.shape[1], device=device)
    pts3D[3, :] = 1.0
    
    pred_cam = {
        "Ps": Ps_pred,
        "Ps_norm": Ps_pred,
        "pts3D": pts3D
    }
    
    print("Comparing different loss functions:")
    print("=" * 50)
    
    # Test original ESFM loss
    from loss_functions import ESFMLoss
    esfm_loss = ESFMLoss(conf)
    esfm_loss_value = esfm_loss(pred_cam, data)
    print(f"ESFM Loss: {esfm_loss_value:.6f}")
    
    # Test pairwise consistency loss
    pairwise_loss = PairwiseConsistencyLoss(conf)
    pairwise_loss_value = pairwise_loss(pred_cam, data)
    print(f"Pairwise Loss: {pairwise_loss_value:.6f}")
    
    # Test combined loss
    combined_loss = CombinedLoss(conf)
    combined_loss_value = combined_loss(pred_cam, data)
    print(f"Combined Loss: {combined_loss_value:.6f}")
    
    print("=" * 50)


if __name__ == "__main__":
    print("Testing Pairwise Consistency Loss for ESfM")
    print("=" * 50)
    
    try:
        # Test pairwise loss
        metrics = test_pairwise_loss()
        
        print("\n" + "=" * 50)
        
        # Compare different losses
        compare_losses()
        
        print("\nPairwise consistency loss implementation completed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc() 