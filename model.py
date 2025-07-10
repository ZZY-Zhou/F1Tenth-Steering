"""neural network architecture
"""

import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F

from math import pi

from fusion_latent import EveLidFus_latent

class EfficientNetB0WithDropout(nn.Module):
    def __init__(self, in_channels=3):
        super(EfficientNetB0WithDropout, self).__init__()
        
        self.efficientnet_b0 = models.efficientnet_b0(pretrained=True)	# Load a pretrained ResNet50 model
        
        # Modify the first convolutional layer to accept the specified number of input channels
        if in_channels != 3:
            self.efficientnet_b0.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        ## with Aggressive Dropout
        self.dropout_prob = 0.75
        
        # Add dropout layers after the original layers
        self.dropout = nn.Dropout(p=self.dropout_prob)
    
    def forward(self, x):
        
        # Stem: Conv2d, BN, SiLU
        x = self.efficientnet_b0.features[0](x)
        
        x = self.dropout(x)	# Dropout after initial layers
        
        # Apply dropout after each block of EfficientNet
        for i in range(1, len(self.efficientnet_b0.features)):
            x = self.efficientnet_b0.features[i](x)
            x = self.dropout(x)
        
        # Average Pooling
        x = self.efficientnet_b0.avgpool(x)
        
        
        # Dropout before the final fully connected layer
        x = self.dropout(x)
        
        
        return x


class MLPDecoder(nn.Module):
    
    def __init__(self, hidden_dim=512, output_dim=1):
        super(MLPDecoder, self).__init__()
        
        # Global Average Pooling (GAP) layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        # Apply GAP to get the vectorized descriptor
        x = self.gap(x)	# (B, input_dim, 1, 1)
        x = x.view(x.size(0), -1)	# Flatten to (B, input_dim)
        
        # Pass through the fully connected layers
        x = self.relu(self.fc1(x))	# (B, hidden_dim)
        
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EveLidNet(nn.Module):
    
    def __init__(self, hidden_dim=1280):	## efficient net
    
        super(EveLidNet, self).__init__()
        
        
        self.encoder_event = EfficientNetB0WithDropout(in_channels=3)
        self.encoder_lidar = EfficientNetB0WithDropout(in_channels=2)
        
        
        self.decoder = MLPDecoder(hidden_dim=hidden_dim, output_dim=1)
        
        ## self.fusion = EveLidFus_latent(in_channels1=hidden_dim, in_channels2=hidden_dim, latent_channels=8, out_channels=hidden_dim)
        self.fusion = EveLidFus_latent(in_channels1=hidden_dim, in_channels2=hidden_dim, latent_channels=16, out_channels=hidden_dim)
        
        
        ## with Aggressive Dropout
        self.dropout_prob = 0.75
        
        # Add dropout layers after the original layers
        self.dropout = nn.Dropout(p=self.dropout_prob)
        
    def forward(self, event_frames, lidar_depth_frames):
        
        
        features_event = self.encoder_event(event_frames)	# output shape: (B, 512, 7, 7)
        
        features_lidar = self.encoder_lidar(lidar_depth_frames)	# output shape: (B, 512, 7, 7)
        
        
        fused_features = self.fusion(features_event, features_lidar)
        
        
        ### KL Loss ###
        # Convert the features to log-probabilities and probabilities for KL divergence
        fused_features_log_prob = F.log_softmax(fused_features, dim=1)	# log-softmax of fused features
        features_event_prob = F.softmax(features_event, dim=1)
        features_lidar_prob = F.softmax(features_lidar, dim=1)
        
        fused_features_prob = F.softmax(fused_features, dim=1)
        features_event_log_prob = F.log_softmax(features_event, dim=1)
        features_lidar_log_prob = F.log_softmax(features_lidar, dim=1)
        
        # KL divergence loss between fused features and features_event
        
        kl_loss_event = F.kl_div(fused_features_log_prob, features_event_prob, reduction='batchmean') + F.kl_div(features_event_log_prob, fused_features_prob, reduction='batchmean')
        
        # KL divergence loss between fused features and features_event
        
        kl_loss_lidar = F.kl_div(fused_features_log_prob, features_lidar_prob, reduction='batchmean') + F.kl_div(features_lidar_log_prob, fused_features_prob, reduction='batchmean')
        
        
        ## with Aggressive Dropout
        fused_features = self.dropout(fused_features)
        
        steering_angle = self.decoder(fused_features)	# output shape: (B, 1)
        
        
        ### KL Loss ###
        return steering_angle, kl_loss_event, kl_loss_lidar

