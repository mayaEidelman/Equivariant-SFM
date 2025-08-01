import torch
from torch import nn
from torch.nn import functional as F
from models.baseNet import BaseNet
from models.layers import *
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for set-of-sets architecture.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.w_o(context)
        output = self.layer_norm(output + x)  # Residual connection
        
        return output


class CrossAttention(nn.Module):
    """
    Cross-attention between camera and point features.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        # query: [batch_size, seq_len_q, d_model]
        # key, value: [batch_size, seq_len_k, d_model]
        batch_size, seq_len_q, d_model = query.shape
        _, seq_len_k, _ = key.shape
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, d_model)
        
        # Output projection
        output = self.w_o(context)
        output = self.layer_norm(output + query)  # Residual connection
        
        return output


class VisualFeatureExtractor(nn.Module):
    """
    Extract visual features from images using a lightweight CNN or pre-trained features.
    """
    def __init__(self, feature_dim=256, use_pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            # Use a lightweight pre-trained model (e.g., ResNet18)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                # ResNet-like blocks
                self._make_layer(64, 64, 2),
                self._make_layer(64, 128, 2, stride=2),
                self._make_layer(128, 256, 2, stride=2),
                
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, feature_dim)
            )
        else:
            # Simple CNN for feature extraction
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, feature_dim)
            )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, images):
        # images: [batch_size, num_cameras, 3, H, W]
        batch_size, num_cameras, _, H, W = images.shape
        images = images.view(-1, 3, H, W)
        features = self.backbone(images)
        features = features.view(batch_size, num_cameras, self.feature_dim)
        return features


class EnhancedSetOfSetBlock(nn.Module):
    """
    Enhanced SetOfSet block with attention mechanisms.
    """
    def __init__(self, d_in, d_out, conf):
        super().__init__()
        self.block_size = conf.get_int("model.block_size")
        self.use_skip = conf.get_bool("model.use_skip")
        self.use_attention = conf.get_bool("model.use_attention", default=True)
        self.num_heads = conf.get_int("model.num_heads", default=8)
        
        # Standard SetOfSet layers
        modules = []
        modules.extend([SetOfSetLayer(d_in, d_out), NormalizationLayer()])
        for i in range(1, self.block_size):
            modules.extend([ActivationLayer(), SetOfSetLayer(d_out, d_out), NormalizationLayer()])
        self.layers = nn.Sequential(*modules)
        
        # Attention mechanisms
        if self.use_attention:
            self.self_attention = MultiHeadAttention(d_out, self.num_heads)
            self.cross_attention = CrossAttention(d_out, self.num_heads)
        
        self.final_act = ActivationLayer()
        
        if self.use_skip:
            if d_in == d_out:
                self.skip = IdentityLayer()
            else:
                self.skip = nn.Sequential(ProjLayer(d_in, d_out), NormalizationLayer())
    
    def forward(self, x):
        # x is [m,n,d] sparse matrix
        xl = self.layers(x)
        
        if self.use_attention:
            # Convert sparse matrix to dense for attention
            dense_x = self._sparse_to_dense(xl)
            
            # Self-attention on camera dimension
            camera_features = dense_x.mean(dim=1)  # [m, d_out]
            camera_features = camera_features.unsqueeze(0)  # [1, m, d_out]
            camera_features = self.self_attention(camera_features)
            camera_features = camera_features.squeeze(0)  # [m, d_out]
            
            # Cross-attention between cameras and points
            point_features = dense_x.mean(dim=0)  # [n, d_out]
            point_features = point_features.unsqueeze(0)  # [1, n, d_out]
            camera_features = camera_features.unsqueeze(0)  # [1, m, d_out]
            
            # Camera attends to points
            enhanced_cameras = self.cross_attention(camera_features, point_features, point_features)
            enhanced_cameras = enhanced_cameras.squeeze(0)  # [m, d_out]
            
            # Points attend to cameras
            enhanced_points = self.cross_attention(point_features, camera_features, camera_features)
            enhanced_points = enhanced_points.squeeze(0)  # [n, d_out]
            
            # Update sparse matrix with enhanced features
            enhanced_features = xl.values.clone()
            for i, (cam_idx, pt_idx) in enumerate(zip(xl.indices[0], xl.indices[1])):
                enhanced_features[i] += (enhanced_cameras[cam_idx] + enhanced_points[pt_idx]) / 2
            
            xl = SparseMat(enhanced_features, xl.indices, xl.cam_per_pts, xl.pts_per_cam, xl.shape)
        
        if self.use_skip:
            xl = self.skip(x) + xl
        
        out = self.final_act(xl)
        return out
    
    def _sparse_to_dense(self, sparse_mat):
        """Convert sparse matrix to dense for attention computation."""
        dense = torch.zeros(sparse_mat.shape, device=sparse_mat.values.device)
        dense[sparse_mat.indices[0], sparse_mat.indices[1]] = sparse_mat.values
        return dense


class FeatureFusion(nn.Module):
    """
    Fuse geometric and visual features.
    """
    def __init__(self, geometric_dim, visual_dim, fused_dim):
        super().__init__()
        self.geometric_dim = geometric_dim
        self.visual_dim = visual_dim
        self.fused_dim = fused_dim
        
        self.geometric_proj = nn.Linear(geometric_dim, fused_dim)
        self.visual_proj = nn.Linear(visual_dim, fused_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(geometric_dim + visual_dim, fused_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(fused_dim, fused_dim)
        
    def forward(self, geometric_features, visual_features):
        # geometric_features: [batch_size, seq_len, geometric_dim]
        # visual_features: [batch_size, seq_len, visual_dim]
        
        geometric_proj = self.geometric_proj(geometric_features)
        visual_proj = self.visual_proj(visual_features)
        
        # Gated fusion
        combined = torch.cat([geometric_features, visual_features], dim=-1)
        gate = self.fusion_gate(combined)
        
        fused = gate * geometric_proj + (1 - gate) * visual_proj
        output = self.output_proj(fused)
        
        return output


class EnhancedSetOfSetNet(BaseNet):
    """
    Enhanced SetOfSet network with attention mechanisms and visual feature integration.
    """
    def __init__(self, conf):
        super().__init__(conf)
        
        num_blocks = conf.get_int('model.num_blocks')
        num_feats = conf.get_int('model.num_features')
        multires = conf.get_int('model.multires')
        use_visual_features = conf.get_bool('model.use_visual_features', default=True)
        use_attention = conf.get_bool('model.use_attention', default=True)
        
        n_d_out = 3
        m_d_out = self.out_channels
        d_in = 2
        
        # Geometric feature embedding
        self.embed = EmbeddingLayer(multires, d_in)
        
        # Visual feature extractor
        if use_visual_features:
            self.visual_extractor = VisualFeatureExtractor(
                feature_dim=num_feats,
                use_pretrained=conf.get_bool('model.use_pretrained_visual', default=True)
            )
            self.feature_fusion = FeatureFusion(num_feats, num_feats, num_feats)
        else:
            self.visual_extractor = None
            self.feature_fusion = None
        
        # Enhanced equivariant blocks
        self.equivariant_blocks = torch.nn.ModuleList([
            EnhancedSetOfSetBlock(self.embed.d_out, num_feats, conf)
        ])
        for i in range(num_blocks - 1):
            self.equivariant_blocks.append(EnhancedSetOfSetBlock(num_feats, num_feats, conf))
        
        # Output networks
        self.m_net = get_linear_layers([num_feats] * 2 + [m_d_out], final_layer=True, batchnorm=False)
        self.n_net = get_linear_layers([num_feats] * 2 + [n_d_out], final_layer=True, batchnorm=False)
        
        # Additional attention for final predictions
        if use_attention:
            self.final_attention = MultiHeadAttention(num_feats, 8)
    
    def forward(self, data):
        x = data.x  # x is [m,n,d] sparse matrix
        
        # Geometric feature extraction
        x = self.embed(x)
        for eq_block in self.equivariant_blocks:
            x = eq_block(x)  # [m,n,d_in] -> [m,n,d_out]
        
        # Visual feature integration (if available)
        if self.visual_extractor is not None and hasattr(data, 'visual_features'):
            # Use pre-extracted visual features
            visual_features = data.visual_features  # [m, visual_dim]
            
            # Convert sparse geometric features to dense for fusion
            dense_geometric = self._sparse_to_dense(x)
            
            # Expand visual features to match geometric features
            # [m, visual_dim] -> [m, n, visual_dim]
            visual_features_expanded = visual_features.unsqueeze(1).expand(-1, dense_geometric.shape[1], -1)
            
            # Fuse geometric and visual features
            fused_features = self.feature_fusion(dense_geometric, visual_features_expanded)
            
            # Convert back to sparse format
            x = self._dense_to_sparse(fused_features, x)
        
        # Final attention (if enabled)
        if hasattr(self, 'final_attention'):
            dense_x = self._sparse_to_dense(x)
            dense_x = dense_x.unsqueeze(0)  # Add batch dimension
            dense_x = self.final_attention(dense_x)
            dense_x = dense_x.squeeze(0)  # Remove batch dimension
            x = self._dense_to_sparse(dense_x, x)
        
        # Cameras predictions
        m_input = x.mean(dim=1)  # [m,d_out]
        m_out = self.m_net(m_input)  # [m, d_m]
        
        # Points predictions
        n_input = x.mean(dim=0)  # [n,d_out]
        n_out = self.n_net(n_input).T  # [n, d_n] -> [d_n, n]
        
        pred_cam = self.extract_model_outputs(m_out, n_out, data)
        
        return pred_cam
    
    def _sparse_to_dense(self, sparse_mat):
        """Convert sparse matrix to dense."""
        dense = torch.zeros(sparse_mat.shape, device=sparse_mat.values.device)
        dense[sparse_mat.indices[0], sparse_mat.indices[1]] = sparse_mat.values
        return dense
    
    def _dense_to_sparse(self, dense, original_sparse):
        """Convert dense matrix back to sparse format."""
        values = dense[original_sparse.indices[0], original_sparse.indices[1]]
        return SparseMat(values, original_sparse.indices, original_sparse.cam_per_pts, 
                        original_sparse.pts_per_cam, original_sparse.shape)


class IdentityLayer(nn.Module):
    """Identity layer for skip connections."""
    def forward(self, x):
        return x 