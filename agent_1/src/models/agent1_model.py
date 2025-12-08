"""
Agent 1 Model Architecture - Direct Coordinate Regression

Uses EfficientNet-B0 as backbone with a regression head to predict
normalized latitude and longitude coordinates.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class GeoLocationRegressor(nn.Module):
    """
    CNN model for predicting geographic coordinates from street view images.
    
    Architecture:
    - Backbone: EfficientNet-B0 (pretrained on ImageNet)
    - Regression Head: Features → [lat, lon] (normalized to [-1, 1])
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        """
        Initialize the GeoLocation Regressor model.
        
        Args:
            backbone_name: Name of the backbone model (e.g., "efficientnet_b0", "resnet50")
            pretrained: Whether to use pretrained ImageNet weights
            dropout: Dropout probability for regression head
            freeze_backbone: Whether to freeze backbone weights (for fine-tuning)
        """
        super().__init__()
        
        # Load backbone
        if backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=pretrained)
            # Remove the classifier
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            # EfficientNet-B0 feature size: 1280
            feature_size = 1280
        elif backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            # Remove the final fully connected layer
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            # ResNet50 feature size: 2048
            feature_size = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Regression head: maps features to [lat, lon]
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),  # Output: [lat, lon] in normalized [-1, 1] range
            # No activation - linear output for regression
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images, shape [batch_size, 3, height, width]
        
        Returns:
            Normalized coordinates [lat, lon], shape [batch_size, 2] in [-1, 1] range
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Predict coordinates using regression head
        coords = self.regressor(features)
        
        # Clamp to [-1, 1] to ensure valid normalized range
        # (though model should learn this naturally)
        coords = torch.clamp(coords, -1.0, 1.0)
        
        return coords
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from backbone (useful for visualization/debugging).
        
        Args:
            x: Input images, shape [batch_size, 3, height, width]
        
        Returns:
            Features from backbone before regression head
        """
        return self.backbone(x)


def create_model(
    backbone: str = "efficientnet_b0",
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    device: str = None
) -> GeoLocationRegressor:
    """
    Convenience function to create and initialize the model.
    
    Args:
        backbone: Backbone architecture name
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability
        freeze_backbone: Whether to freeze backbone
        device: Device to move model to (e.g., "cuda", "mps", "cpu")
    
    Returns:
        Initialized model
    """
    model = GeoLocationRegressor(
        backbone_name=backbone,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )
    
    if device is not None:
        model = model.to(device)
    
    return model

