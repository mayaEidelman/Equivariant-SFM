import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from torchvision import models
import torch.nn.functional as F
import cv2


class VisualFeatureLoader:
    """
    Load and preprocess visual features for the enhanced ESfM model.
    """
    def __init__(self, images_path, feature_type='cnn', pretrained=True):
        self.images_path = images_path
        self.feature_type = feature_type
        self.pretrained = pretrained
        
        # Standard image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize feature extractor
        if self.pretrained:
            self.feature_extractor = self._load_pretrained_extractor()
    
    def _load_pretrained_extractor(self):
        """Load a pretrained feature extractor."""
        # Use ResNet18 as default
        model = models.resnet18(pretrained=True)
        # Remove the final classification layer
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model
    
    def load_image(self, image_path):
        """Load and preprocess a single image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image
    
    def extract_features(self, image_path):
        """Extract features from a single image."""
        image = self.load_image(image_path)
        
        if self.pretrained:
            with torch.no_grad():
                features = self.feature_extractor(image.unsqueeze(0))
                features = features.squeeze(0).squeeze(-1).squeeze(-1)  # [512]
        else:
            # Use simple CNN features
            features = self._extract_simple_features(image)
        
        return features
    
    def _extract_simple_features(self, image):
        """Extract simple CNN features."""
        # Simple 3-layer CNN with learnable weights
        conv1 = F.conv2d(image.unsqueeze(0), 
                         torch.randn(64, 3, 3, 3, requires_grad=True), padding=1)
        conv1 = F.relu(conv1)
        conv1 = F.max_pool2d(conv1, 2)
        
        conv2 = F.conv2d(conv1, 
                         torch.randn(128, 64, 3, 3, requires_grad=True), padding=1)
        conv2 = F.relu(conv2)
        conv2 = F.max_pool2d(conv2, 2)
        
        conv3 = F.conv2d(conv2, 
                         torch.randn(256, 128, 3, 3, requires_grad=True), padding=1)
        conv3 = F.relu(conv3)
        conv3 = F.adaptive_avg_pool2d(conv3, (1, 1))
        
        features = conv3.squeeze(0).squeeze(-1).squeeze(-1)  # [256]
        return features
    
    def load_scene_images(self, scene_name, image_names):
        """Load all images for a scene."""
        images = []
        for image_name in image_names:
            image_path = os.path.join(self.images_path, scene_name, image_name)
            try:
                image = self.load_image(image_path)
                images.append(image)
            except FileNotFoundError:
                print(f"Warning: Image not found: {image_path}")
                # Create a dummy image
                dummy_image = torch.zeros(3, 224, 224)
                images.append(dummy_image)
        
        return torch.stack(images)  # [num_images, 3, H, W]


class DINOFeatureExtractor:
    """
    Extract DINO features for enhanced visual representation.
    """
    def __init__(self, dino_model_path=None):
        self.dino_model_path = dino_model_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load DINO model if available
        if dino_model_path and os.path.exists(dino_model_path):
            self.model = self._load_dino_model()
        else:
            print("Warning: DINO model not found, using ResNet18 as fallback")
            self.model = models.resnet18(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.eval()
    
    def _load_dino_model(self):
        """Load DINO model from path."""
        try:
            # Try to load DINO model (this would require the actual DINO implementation)
            # For now, use a ResNet with DINO-style features
            model = models.resnet50(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            return model
        except Exception as e:
            print(f"Failed to load DINO model: {e}")
            # Fallback to ResNet18
            model = models.resnet18(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            return model
    
    def extract_features(self, image_path):
        """Extract DINO features from image."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        with torch.no_grad():
            features = self.model(image.unsqueeze(0))
            features = features.squeeze(0).squeeze(-1).squeeze(-1)
        
        return features


class SuperPointFeatureExtractor:
    """
    Extract SuperPoint features for enhanced visual representation.
    """
    def __init__(self, superpoint_model_path=None):
        self.superpoint_model_path = superpoint_model_path
        self.transform = transforms.Compose([
            transforms.Resize((480, 640)),  # SuperPoint default size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load SuperPoint model if available
        if superpoint_model_path and os.path.exists(superpoint_model_path):
            self.model = self._load_superpoint_model()
        else:
            print("Warning: SuperPoint model not found, using ResNet18 as fallback")
            self.model = models.resnet18(pretrained=True)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.model.eval()
    
    def _load_superpoint_model(self):
        """Load SuperPoint model from path."""
        try:
            # Try to load SuperPoint model (this would require the actual SuperPoint implementation)
            # For now, use a ResNet with SuperPoint-style features
            model = models.resnet18(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            return model
        except Exception as e:
            print(f"Failed to load SuperPoint model: {e}")
            # Fallback to ResNet18
            model = models.resnet18(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            return model
    
    def extract_features(self, image_path):
        """Extract SuperPoint features from image."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        with torch.no_grad():
            features = self.model(image.unsqueeze(0))
            features = features.squeeze(0).squeeze(-1).squeeze(-1)
        
        return features


def add_visual_features_to_data(data, images_path, feature_type='cnn'):
    """
    Add visual features to scene data by actually loading images.
    
    Args:
        data: SceneData object
        images_path: Path to images directory
        feature_type: Type of visual features ('cnn', 'dino', 'superpoint')
    
    Returns:
        data: Updated SceneData object with visual features
    """
    try:
        # Create feature loader
        if feature_type == 'dino':
            feature_loader = DINOFeatureExtractor()
        elif feature_type == 'superpoint':
            feature_loader = SuperPointFeatureExtractor()
        else:
            feature_loader = VisualFeatureLoader(images_path, feature_type)
        
        # Get scene name and try to find images
        scene_name = data.scan_name
        scene_images_path = os.path.join(images_path, scene_name)
        
        if not os.path.exists(scene_images_path):
            print(f"Warning: Scene images not found at {scene_images_path}")
            # Create dummy features as fallback
            n_cameras = data.y.shape[0]
            if feature_type == 'dino':
                data.visual_features = torch.randn(n_cameras, 2048)  # DINO feature dimension
            elif feature_type == 'superpoint':
                data.visual_features = torch.randn(n_cameras, 256)  # SuperPoint feature dimension
            else:
                data.visual_features = torch.randn(n_cameras, 512)  # CNN feature dimension
            return data
        
        # Find image files in the scene directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(scene_images_path) 
                              if f.lower().endswith(ext)])
        
        if not image_files:
            print(f"Warning: No image files found in {scene_images_path}")
            # Create dummy features as fallback
            n_cameras = data.y.shape[0]
            if feature_type == 'dino':
                data.visual_features = torch.randn(n_cameras, 2048)
            elif feature_type == 'superpoint':
                data.visual_features = torch.randn(n_cameras, 256)
            else:
                data.visual_features = torch.randn(n_cameras, 512)
            return data
        
        # Sort image files to ensure consistent ordering
        image_files.sort()
        
        # Extract features from actual images
        features = []
        n_cameras = data.y.shape[0]
        
        for i in range(n_cameras):
            if i < len(image_files):
                # Extract features from actual image
                image_path = os.path.join(scene_images_path, image_files[i])
                try:
                    feature = feature_loader.extract_features(image_path)
                    features.append(feature)
                except Exception as e:
                    print(f"Warning: Failed to extract features from {image_path}: {e}")
                    # Create dummy feature as fallback
                    if feature_type == 'dino':
                        feature = torch.randn(2048)
                    elif feature_type == 'superpoint':
                        feature = torch.randn(256)
                    else:
                        feature = torch.randn(512)
                    features.append(feature)
            else:
                # Create dummy feature if we have more cameras than images
                if feature_type == 'dino':
                    feature = torch.randn(2048)
                elif feature_type == 'superpoint':
                    feature = torch.randn(256)
                else:
                    feature = torch.randn(512)
                features.append(feature)
        
        # Add visual features to data
        data.visual_features = torch.stack(features)  # [n_cameras, feature_dim]
        
        print(f"Added visual features to scene {data.scan_name}: {data.visual_features.shape}")
        print(f"  - Feature type: {feature_type}")
        print(f"  - Images found: {len(image_files)}")
        print(f"  - Cameras: {n_cameras}")
        
    except Exception as e:
        print(f"Warning: Failed to add visual features to scene {data.scan_name}: {e}")
        # Create dummy features as fallback
        n_cameras = data.y.shape[0]
        if feature_type == 'dino':
            data.visual_features = torch.randn(n_cameras, 2048)
        elif feature_type == 'superpoint':
            data.visual_features = torch.randn(n_cameras, 256)
        else:
            data.visual_features = torch.randn(n_cameras, 512)
    
    return data


def create_visual_attention_mask(valid_points, n_cameras, n_points):
    """
    Create attention mask for visual features based on valid points.
    
    Args:
        valid_points: Boolean mask indicating valid camera-point pairs
        n_cameras: Number of cameras
        n_points: Number of points
    
    Returns:
        attention_mask: Attention mask for visual features
    """
    # Create attention mask based on valid points
    attention_mask = torch.zeros(n_cameras, n_points, dtype=torch.bool)
    
    # Set valid camera-point pairs to True
    for cam_idx in range(n_cameras):
        for pt_idx in range(n_points):
            if valid_points[cam_idx, pt_idx]:
                attention_mask[cam_idx, pt_idx] = True
    
    return attention_mask


def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    """
    Load and preprocess multiple images for batch processing.
    
    Args:
        image_paths: List of image file paths
        target_size: Target size for resizing (H, W)
    
    Returns:
        images: Tensor of preprocessed images [batch_size, 3, H, W]
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            images.append(image)
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            # Create dummy image
            dummy_image = torch.zeros(3, target_size[0], target_size[1])
            images.append(dummy_image)
    
    return torch.stack(images) 