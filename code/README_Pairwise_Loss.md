# Unsupervised Pairwise Consistency Loss for ESfM

This implementation adds **unsupervised** pairwise consistency loss to the ESfM (Deep Permutation Equivariant Structure from Motion) framework. The pairwise consistency loss enforces geometric consistency between predicted camera poses and pairwise matches without requiring ground truth relative poses.

## Overview

The unsupervised pairwise consistency loss addresses the challenge of leveraging pairwise image matches to guide the learning process in an unsupervised setting. This is implemented using two complementary geometric constraints:

1. **Epipolar Consistency**: Uses fundamental matrices to enforce epipolar geometry constraints
2. **Geometric Consistency**: Uses triangulation and reprojection to enforce geometric consistency

## Key Components

### 1. PairwiseConsistencyLoss Class

Located in `loss_functions.py`, this class implements the unsupervised pairwise consistency loss with two main components:

- **Epipolar Consistency**: Enforces epipolar geometry using fundamental matrices computed from predicted camera poses
- **Geometric Consistency**: Uses triangulation and reprojection to ensure predicted poses are consistent with observed matches

### 2. CombinedLoss Class

Combines the original ESFM loss with the new unsupervised pairwise consistency loss, allowing for weighted combination of both losses.

### 3. Pairwise Utilities

Located in `utils/pairwise_utils.py`, these utilities provide:

- `extract_pairwise_matches_from_scene()`: Extracts pairwise matches from scene data
- `compute_relative_poses_for_scene()`: Computes relative poses from pairwise matches (for validation only)
- `evaluate_pairwise_consistency()`: Evaluates pairwise consistency metrics

### 4. Enhanced SceneData Class

The `SceneData` class has been enhanced to automatically compute and include pairwise information:

- Pairwise matches between camera pairs
- Automatic extraction of matches from the M matrix

## Usage

### 1. Basic Usage

```python
from loss_functions import CombinedLoss
from datasets.SceneData import create_scene_data

# Load configuration with pairwise loss
conf = ConfigFactory.parse_file('confs/Learning_Euc_Pairwise.conf')

# Create scene data (automatically includes pairwise information)
data = create_scene_data(conf)

# Create combined loss function
loss_func = CombinedLoss(conf)

# Use in training loop
loss = loss_func(pred_cam, data, epoch)
```

### 2. Configuration

The unsupervised pairwise loss can be configured in the configuration file:

```hocon
loss
{
    func = CombinedLoss
    infinity_pts_margin = 1e-4
    normalize_grad = True
    hinge_loss = True
    hinge_loss_weight = 1
    
    # Unsupervised pairwise consistency loss weights
    esfm_weight = 1.0
    pairwise_weight = 0.5
    epipolar_weight = 1.0
    geometric_weight = 0.5
}
```

### 3. Evaluation

```python
from utils import pairwise_utils

# Evaluate pairwise consistency metrics
metrics = pairwise_utils.evaluate_pairwise_consistency(data, pred_cam)

print(f"Mean epipolar error: {metrics['mean_epipolar_error']:.6f}")
print(f"Mean geometric error: {metrics['mean_geometric_error']:.6f}")
```

## Implementation Details

### Epipolar Consistency

The epipolar consistency loss enforces the epipolar constraint:

1. **Compute Fundamental Matrix**: From predicted camera poses using essential matrix decomposition
2. **Compute Epipolar Error**: Using symmetric epipolar distance
3. **Loss**: Mean epipolar error across all matched points

### Geometric Consistency

The geometric consistency loss ensures geometric consistency:

1. **Triangulate 3D Points**: Using predicted camera poses and matched points
2. **Reproject Points**: Back to both cameras using predicted poses
3. **Compute Reprojection Error**: Between original and reprojected points
4. **Depth Penalty**: Penalize points behind cameras
5. **Loss**: Combined reprojection error and depth penalty

### Integration with ESfM

The unsupervised pairwise consistency loss is designed to work seamlessly with the existing ESfM framework:

- Automatically extracts pairwise information from scene data
- Compatible with both calibrated and uncalibrated scenarios
- Maintains permutation equivariance of the original framework
- Can be used alongside or instead of the original ESFM loss

## Example Script

Run the example script to test the implementation:

```bash
cd code
python example_pairwise_loss.py
```

This script demonstrates:
- Loading scene data with pairwise information
- Using the combined loss function
- Evaluating pairwise consistency metrics
- Comparing different loss functions

## Benefits

1. **Unsupervised Learning**: Works without ground truth relative poses
2. **Geometric Consistency**: Enforces proper geometric relationships
3. **Robustness**: Additional geometric constraints improve robustness to noise
4. **Flexibility**: Can be weighted and combined with existing losses
5. **No Supervision Required**: Only uses observed pairwise matches

## Requirements

- OpenCV (for essential matrix computation)
- PyTorch
- NumPy
- PyHocon (for configuration)

## Notes

- The implementation automatically handles both calibrated and uncalibrated scenarios
- All computations are performed on the same device as the input tensors
- The loss is designed to be robust to outliers in pairwise matches
- No ground truth relative poses are required

## Future Improvements

1. **Advanced Matching**: Integration with learned feature matching
2. **Multi-view Constraints**: Extension to multi-view geometric constraints
3. **Adaptive Weighting**: Dynamic weighting based on match quality
4. **Uncertainty Modeling**: Incorporating uncertainty in geometric constraints 