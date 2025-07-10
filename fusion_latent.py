import torch
import torch.nn as nn

class EveLidFus_latent(nn.Module):
    def __init__(self, in_channels1, in_channels2, latent_channels, out_channels):
        super(EveLidFus_latent, self).__init__()
        self.proj1 = nn.Conv2d(in_channels1, latent_channels, kernel_size=1, stride=1, padding=0)
        self.proj2 = nn.Conv2d(in_channels2, latent_channels, kernel_size=1, stride=1, padding=0)
        
        self.conv = nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=1, stride=1, padding=0)
        
        self.reproj = nn.Conv2d(latent_channels * 2, out_channels, kernel_size=1, stride=1, padding=0)
        
        self.gelu = nn.GELU()
        
    def forward(self, feature1, feature2):
        
        feature1 = self.proj1(feature1)
        feature2 = self.proj2(feature2)
        
        # Concatenate along the channel dimension
        features = torch.cat([feature1, feature2], dim=1)
        
        features = self.conv(features)
        
        features = self.gelu(features)
        
        feature1 = torch.mul(features, feature1)
        feature2 = torch.mul(features, feature2)
        
        out = torch.cat([feature1, feature2], dim=1)
        
        out = self.reproj(out)
        
        return out

