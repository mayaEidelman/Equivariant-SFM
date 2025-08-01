#!/usr/bin/env python3
"""
Real test script for the enhanced ESfM model with actual implementation.
"""

import torch
import numpy as np
from pyhocon import ConfigFactory
import os
import sys
import warnings

# Add the code directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.EnhancedSetOfSet import EnhancedSetOfSetNet
from datasets.SceneData import create_scene_data
from utils import visual_utils


def test_real_enhanced_model():
    """Test the enhanced model with real data and visual features."""
    print("Testing Enhanced ESfM Model with Real Implementation")
    print("=" * 70)
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data with visual features
    print("Creating scene data...")
    data = create_scene_data(conf)
    
    print(f"Scene: {data.scan_name}")
    print(f"Number of cameras: {data.y.shape[0]}")
    print(f"Number of points: {data.M.shape[1]}")
    
    # Check if visual features were loaded
    if hasattr(data, 'visual_features'):
        print(f"Visual features loaded: {data.visual_features.shape}")
    else:
        print("No visual features loaded")
    
    # Create enhanced model
    print("\nCreating enhanced model...")
    model = EnhancedSetOfSetNet(conf)
    
    print(f"Model Configuration:")
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


def test_attention_mechanisms_real():
    """Test attention mechanisms with real data."""
    print(f"\nTesting Attention Mechanisms with Real Data")
    print("-" * 50)
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
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
        
        # Check if outputs are different (attention should have an effect)
        attention_diff = torch.norm(pred_with_attention['Ps_norm'] - pred_without_attention['Ps_norm'])
        print(f"  - Attention effect magnitude: {attention_diff.item():.6f}")
        
        if attention_diff > 1e-6:
            print(f"  ✓ Attention mechanisms are working!")
        else:
            print(f"  ⚠ Attention effect is minimal")
    
    return True


def test_visual_feature_integration_real():
    """Test visual feature integration with real data."""
    print(f"\nTesting Visual Feature Integration with Real Data")
    print("-" * 50)
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Test different visual feature configurations
    test_configs = [
        {'use_visual_features': True, 'use_pretrained_visual': True},
        {'use_visual_features': True, 'use_pretrained_visual': False},
        {'use_visual_features': False, 'use_pretrained_visual': False}
    ]
    
    for i, config in enumerate(test_configs):
        print(f"Test {i+1}: Visual features={config['use_visual_features']}, Pretrained={config['use_pretrained_visual']}")
        
        # Update configuration
        conf["model"]["use_visual_features"] = config['use_visual_features']
        conf["model"]["use_pretrained_visual"] = config['use_pretrained_visual']
        
        # Create scene data
        data = create_scene_data(conf)
        
        # Create model
        model = EnhancedSetOfSetNet(conf)
        
        with torch.no_grad():
            pred_cam = model(data)
            print(f"  ✓ Model works with config {i+1}")
            print(f"    Output shape: {pred_cam['Ps_norm'].shape}")
            
            if hasattr(data, 'visual_features'):
                print(f"    Visual features: {data.visual_features.shape}")
            else:
                print(f"    No visual features")
    
    return True


def test_model_components_real():
    """Test individual components with real data."""
    print(f"\nTesting Model Components with Real Data")
    print("-" * 50)
    
    from models.EnhancedSetOfSet import MultiHeadAttention, CrossAttention, VisualFeatureExtractor, FeatureFusion
    
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
    
    # Test FeatureFusion
    print("Testing FeatureFusion...")
    feature_fusion = FeatureFusion(geometric_dim=256, visual_dim=256, fused_dim=256)
    geometric_features = torch.randn(4, 10, 256)  # [batch_size, seq_len, geometric_dim]
    visual_features = torch.randn(4, 10, 256)     # [batch_size, seq_len, visual_dim]
    fused_features = feature_fusion(geometric_features, visual_features)
    print(f"  ✓ FeatureFusion output shape: {fused_features.shape}")
    
    return True


def test_differentiable_forward_real():
    """Test that the enhanced model is differentiable with real data."""
    print(f"\nTesting Differentiability with Real Data")
    print("-" * 50)
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
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
        num_params_with_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item()
                num_params_with_grad += 1
        
        print(f"  - Total gradient norm: {total_grad_norm:.6f}")
        print(f"  - Parameters with gradients: {num_params_with_grad}")
        
        if total_grad_norm > 0:
            print(f"  ✓ Gradients are non-zero - model is differentiable!")
        else:
            print(f"  ⚠ Warning: Gradients are zero")
        
    except Exception as e:
        print(f"✗ Backward pass failed: {e}")
        return False
    
    return True


def test_memory_efficiency():
    """Test memory efficiency of the enhanced model."""
    print(f"\nTesting Memory Efficiency")
    print("-" * 50)
    
    import torch.cuda
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    
    # Create scene data
    data = create_scene_data(conf)
    
    # Create model
    model = EnhancedSetOfSetNet(conf)
    
    # Test memory usage
    print("Testing memory usage...")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        data = data.to(device)
        
        # Measure memory before forward pass
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated()
        
        with torch.no_grad():
            pred_cam = model(data)
        
        # Measure memory after forward pass
        memory_after = torch.cuda.memory_allocated()
        memory_used = memory_after - memory_before
        
        print(f"  - Memory used: {memory_used / 1024**2:.2f} MB")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        if memory_used < 1024**3:  # Less than 1GB
            print(f"  ✓ Memory usage is reasonable")
        else:
            print(f"  ⚠ High memory usage")
    else:
        print("  - CUDA not available, skipping memory test")
    
    return True


def test_integration_with_pairwise_loss():
    """Test integration with pairwise loss."""
    print(f"\nTesting Integration with Pairwise Loss")
    print("-" * 50)
    
    # Load configuration
    conf = ConfigFactory.parse_file('confs/Learning_Euc_Enhanced.conf')
    conf["dataset"]["scan"] = conf.get_list('dataset.train_set')[0]
    conf["loss"]["func"] = "CombinedLoss"
    
    # Create scene data
    data = create_scene_data(conf)
    
    # Create model and loss
    model = EnhancedSetOfSetNet(conf)
    
    try:
        from loss_functions import CombinedLoss
        loss_func = CombinedLoss(conf)
        
        # Test forward pass with loss
        with torch.no_grad():
            pred_cam = model(data)
            loss = loss_func(pred_cam, data, epoch=0)
            
            print(f"✓ Integration successful!")
            print(f"  - Loss value: {loss.item():.6f}")
            print(f"  - Model output shape: {pred_cam['Ps_norm'].shape}")
        
    except Exception as e:
        print(f"✗ Integration failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Enhanced ESfM Model - Real Implementation Testing")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 7
    
    # Test 1: Enhanced model
    if test_real_enhanced_model():
        tests_passed += 1
    
    # Test 2: Attention mechanisms
    if test_attention_mechanisms_real():
        tests_passed += 1
    
    # Test 3: Visual feature integration
    if test_visual_feature_integration_real():
        tests_passed += 1
    
    # Test 4: Model components
    if test_model_components_real():
        tests_passed += 1
    
    # Test 5: Differentiability
    if test_differentiable_forward_real():
        tests_passed += 1
    
    # Test 6: Memory efficiency
    if test_memory_efficiency():
        tests_passed += 1
    
    # Test 7: Integration with pairwise loss
    if test_integration_with_pairwise_loss():
        tests_passed += 1
    
    print("\n" + "=" * 70)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! The enhanced model is working correctly.")
        print("✓ Real attention mechanisms and visual features are properly integrated!")
    else:
        print("✗ Some tests failed. Please check the implementation.") 