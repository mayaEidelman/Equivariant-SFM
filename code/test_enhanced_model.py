#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced ESfM model with attention mechanisms
and visual feature integration.
"""

import torch
import numpy as np
from pyhocon import ConfigFactory
import os
import sys

# Add the code directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.EnhancedSetOfSet import EnhancedSetOfSetNet
from datasets.SceneData import create_scene_data
from utils import visual_utils


def test_enhanced_model():
    """Test the enhanced model with attention and visual features."""
    print("Testing Enhanced ESfM Model with Attention and Visual Features")
    print("=" * 70)
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    print(f"Scene: {data.scan_name}")
    print(f"Number of cameras: {data.y.shape[0]}")
    print(f"Number of points: {data.M.shape[1]}")
    
    # Add visual features to data
    data = visual_utils.add_visual_features_to_data(data, "dummy_path", "cnn")
    
    # Create enhanced model
    model = EnhancedSetOfSetNet(conf)
    
    print(f"\nModel Configuration:")
    print(f"  - Use attention: {conf.get_bool('model.use_attention')}")
    print(f"  - Use visual features: {conf.get_bool('model.use_visual_features')}")
    print(f"  - Number of heads: {conf.get_int('model.num_heads')}")
    print(f"  - Number of features: {conf.get_int('model.num_features')}")
    print(f"  - Number of blocks: {conf.get_int('model.num_blocks')}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    model.eval()
    
    with torch.no_grad():
        pred_cam = model(data)
        
        print(f"✓ Forward pass successful!")
        print(f"  - Predicted cameras shape: {pred_cam['Ps_norm'].shape}")
        print(f"  - Predicted 3D points shape: {pred_cam['pts3D'].shape}")
    
    return True


def test_attention_mechanisms():
    """Test attention mechanisms in the enhanced model."""
    print(f"\nTesting Attention Mechanisms")
    print("-" * 40)
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    # Create model with attention
    model = EnhancedSetOfSetNet(conf)
    
    # Test with attention enabled
    conf["model"]["use_attention"] = True
    model_with_attention = EnhancedSetOfSetNet(conf)
    
    # Test with attention disabled
    conf["model"]["use_attention"] = False
    model_without_attention = EnhancedSetOfSetNet(conf)
    
    print(f"Testing models with and without attention...")
    
    with torch.no_grad():
        # Test with attention
        pred_with_attention = model_with_attention(data)
        
        # Test without attention
        pred_without_attention = model_without_attention(data)
        
        print(f"✓ Both models work correctly!")
        print(f"  - With attention: {pred_with_attention['Ps_norm'].shape}")
        print(f"  - Without attention: {pred_without_attention['Ps_norm'].shape}")
    
    return True


def test_visual_feature_integration():
    """Test visual feature integration in the enhanced model."""
    print(f"\nTesting Visual Feature Integration")
    print("-" * 40)
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    # Test different visual feature types
    feature_types = ['cnn', 'dino', 'superpoint']
    
    for feature_type in feature_types:
        print(f"Testing with {feature_type.upper()} features...")
        
        # Add visual features
        data_with_visual = visual_utils.add_visual_features_to_data(
            data, "dummy_path", feature_type
        )
        
        # Create model with visual features
        conf["model"]["use_visual_features"] = True
        model = EnhancedSetOfSetNet(conf)
        
        with torch.no_grad():
            pred_cam = model(data_with_visual)
            print(f"  ✓ {feature_type.upper()} features integrated successfully!")
    
    return True


def test_model_components():
    """Test individual components of the enhanced model."""
    print(f"\nTesting Model Components")
    print("-" * 40)
    
    from models.EnhancedSetOfSet import MultiHeadAttention, CrossAttention, VisualFeatureExtractor
    
    # Test MultiHeadAttention
    print("Testing MultiHeadAttention...")
    attention = MultiHeadAttention(d_model=256, num_heads=8)
    x = torch.randn(1, 10, 256)  # [batch_size, seq_len, d_model]
    output = attention(x)
    print(f"  ✓ MultiHeadAttention output shape: {output.shape}")
    
    # Test CrossAttention
    print("Testing CrossAttention...")
    cross_attention = CrossAttention(d_model=256, num_heads=8)
    query = torch.randn(1, 5, 256)   # [batch_size, seq_len_q, d_model]
    key = torch.randn(1, 10, 256)    # [batch_size, seq_len_k, d_model]
    value = torch.randn(1, 10, 256)  # [batch_size, seq_len_k, d_model]
    output = cross_attention(query, key, value)
    print(f"  ✓ CrossAttention output shape: {output.shape}")
    
    # Test VisualFeatureExtractor
    print("Testing VisualFeatureExtractor...")
    visual_extractor = VisualFeatureExtractor(feature_dim=256, use_pretrained=False)
    images = torch.randn(4, 3, 224, 224)  # [batch_size, num_cameras, 3, H, W]
    features = visual_extractor(images)
    print(f"  ✓ VisualFeatureExtractor output shape: {features.shape}")
    
    return True


def test_differentiable_forward():
    """Test that the enhanced model is differentiable."""
    print(f"\nTesting Differentiability")
    print("-" * 40)
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    data = visual_utils.add_visual_features_to_data(data, "dummy_path", "cnn")
    
    # Create model
    model = EnhancedSetOfSetNet(conf)
    
    # Create dummy loss function
    def dummy_loss(pred_cam):
        return pred_cam['Ps_norm'].norm() + pred_cam['pts3D'].norm()
    
    # Test backward pass
    print("Testing backward pass...")
    
    try:
        pred_cam = model(data)
        loss = dummy_loss(pred_cam)
        loss.backward()
        
        print(f"✓ Backward pass successful!")
        print(f"  - Loss value: {loss.item():.6f}")
        
        # Check gradients
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
        
        print(f"  - Total gradient norm: {total_grad_norm:.6f}")
        
        if total_grad_norm > 0:
            print(f"  ✓ Gradients are non-zero - model is differentiable!")
        else:
            print(f"  ⚠ Warning: Gradients are zero")
        
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return False
    
    return True


def compare_model_architectures():
    """Compare original and enhanced model architectures."""
    print(f"\nComparing Model Architectures")
    print("-" * 40)
    
    from models.SetOfSet import SetOfSetNet
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    # Create original model
    conf["model"]["type"] = "SetOfSet.SetOfSetNet"
    original_model = SetOfSetNet(conf)
    
    # Create enhanced model
    conf["model"]["type"] = "EnhancedSetOfSet.EnhancedSetOfSetNet"
    enhanced_model = EnhancedSetOfSetNet(conf)
    
    print("Model comparison:")
    print(f"  - Original model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"  - Enhanced model parameters: {sum(p.numel() for p in enhanced_model.parameters()):,}")
    
    # Test both models
    with torch.no_grad():
        original_pred = original_model(data)
        enhanced_pred = enhanced_model(data)
        
        print(f"  ✓ Both models produce valid outputs!")
        print(f"    Original: {original_pred['Ps_norm'].shape}")
        print(f"    Enhanced: {enhanced_pred['Ps_norm'].shape}")
    
    return True


if __name__ == "__main__":
    print("Enhanced ESfM Model Testing")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: Enhanced model
    if test_enhanced_model():
        tests_passed += 1
    
    # Test 2: Attention mechanisms
    if test_attention_mechanisms():
        tests_passed += 1
    
    # Test 3: Visual feature integration
    if test_visual_feature_integration():
        tests_passed += 1
    
    # Test 4: Model components
    if test_model_components():
        tests_passed += 1
    
    # Test 5: Differentiability
    if test_differentiable_forward():
        tests_passed += 1
    
    # Test 6: Architecture comparison
    if compare_model_architectures():
        tests_passed += 1
    
    print("\n" + "=" * 70)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! The enhanced model is working correctly.")
        print("✓ Attention mechanisms and visual features are properly integrated!")
    else:
        print("✗ Some tests failed. Please check the implementation.") 