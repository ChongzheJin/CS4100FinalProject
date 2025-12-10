"""
Interactive test script for combined Agent 1 and Agent 2
Visualizes predictions on map and shows how the fusion works
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
import yaml
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import json
import random

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from data.datasets import GeoCSVDataset

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent_1" / "src"))

# Import custom modules
from agent_2.src.grid_mapper import GridMapper
from models.agent1_model import create_model as create_agent1_model
from utils.coordinates import compute_normalization_params

# UNCOMMENT TO DOWNLOAD MODEL
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Initialize grid mapper
GRID_MAPPER = GridMapper(rows=7, cols=7)


class CombinedAgentTester:
    def __init__(self, agent1_checkpoint, agent2_checkpoint, data_config_path, train_config_path):
        """Initialize the combined agent tester"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  "mps" if torch.backends.mps.is_available() else "cpu")
        print(f'Using device: {self.device.type}')
        
        # Load configs
        with open(data_config_path, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        with open(train_config_path, 'r') as f:
            self.train_config = yaml.safe_load(f)
        
        self.img_size = self.data_config['img_size']
        
        # Setup transforms
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        self.denormalize = transforms.Compose([
            transforms.Normalize(mean=[-m/s for m, s in zip(self.mean, self.std)],
                               std=[1/s for s in self.std]),
            transforms.ToPILImage()
        ])
        
        # Compute normalization parameters for Agent 1
        print("\nComputing coordinate normalization parameters...")
        train_csv = self.data_config['train_csv']
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = compute_normalization_params(train_csv)
        print(f"Latitude range: [{self.lat_min:.6f}, {self.lat_max:.6f}]")
        print(f"Longitude range: [{self.lon_min:.6f}, {self.lon_max:.6f}]")
        
        # Load models
        print("\nLoading Agent 1 (Coordinate Regression)...")
        self.agent1 = self.load_agent1(agent1_checkpoint)
        
        print("Loading Agent 2 (Grid Classification)...")
        self.agent2 = self.load_agent2(agent2_checkpoint)
        
        # Load map if exists
        self.map_img = None
        map_path = "data/NA_map.png"
        if os.path.exists(map_path):
            self.map_img = Image.open(map_path)
            print(f"Map loaded from {map_path}")
        else:
            print(f"Warning: Map not found at {map_path}")
        
        # Initialize datasets (lazy loading)
        self.test_dataset = None
        self.test_loader = None
        self.train_dataset = None
        
        # Create output directory
        os.makedirs("combined_test_results", exist_ok=True)
        
        print("\nCombined Agent Tester initialized successfully!")
    
    def load_agent1(self, checkpoint_path):
        """Load trained Agent 1 model"""
        model_config = self.train_config['model']
        
        model = create_agent1_model(
            backbone=model_config['backbone'],
            pretrained=False,
            dropout=model_config['dropout'],
            freeze_backbone=False,
            device=self.device
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def load_agent2(self, checkpoint_path):
        """Load trained Agent 2 model"""
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 49),
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def haversine_km(self, lat1, lon1, lat2, lon2):
        """Compute distance between two lat/lon points in kilometers"""
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c
    
    def compute_likelihood_terms(self, lat1, lon1, sigma_km=800.0):
        """Compute likelihood terms for each grid based on distance from Agent 1 prediction"""
        num_grids = 49
        likelihoods = np.zeros(num_grids)
        
        for g in range(num_grids):
            lat_g, lon_g = GRID_MAPPER.get_grid_center(g)
            dist = self.haversine_km(lat1, lon1, lat_g, lon_g)
            likelihoods[g] = np.exp(-(dist**2) / (2 * sigma_km**2))
        
        return likelihoods
    
    def combine_agents(self, img_tensor, return_details=True, weight=0.7):
        """Combine outputs from both agents"""
        img_tensor = img_tensor.to(self.device)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Agent 1 prediction
        with torch.no_grad():
            agent1_output = self.agent1(img_tensor)
            # Agent 1 outputs in [-1, 1] range, need to denormalize
            norm_lat, norm_lon = agent1_output[0].cpu().numpy()
            # Denormalize from [-1, 1] to actual coordinates
            lat1 = (norm_lat + 1.0) * 0.5 * (self.lat_max - self.lat_min) + self.lat_min
            lon1 = (norm_lon + 1.0) * 0.5 * (self.lon_max - self.lon_min) + self.lon_min
        
        # Convert Agent 1 coords to likelihood terms
        likelihoods = self.compute_likelihood_terms(lat1, lon1)
        
        # Agent 2 prediction
        with torch.no_grad():
            logits = self.agent2(img_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()


        #likelihood term normalization
        likelihoods_norm = likelihoods / likelihoods.sum()

        # Combine scores
        scores = w * likelihoods_norm + (1 - w) * probs
        
        if scores.sum() == 0:
            scores = probs
        else:
            scores = scores / scores.sum()
        
        # Pick best grid
        best_grid = int(np.argmax(scores))
        final_lat, final_lon = GRID_MAPPER.get_grid_center(best_grid)
        
        if return_details:
            return final_lat, final_lon, {
                "agent1_coords": (lat1, lon1),
                "agent1_grid": GRID_MAPPER.latlon_to_grid(lat1, lon1),
                "agent2_probs": probs,
                "agent2_top_grid": int(np.argmax(probs)),
                "likelihoods": likelihoods,
                "combined_scores": scores,
                "final_grid": best_grid,
                "final_coords": (final_lat, final_lon)
            }
        return final_lat, final_lon
    
    def visualize_combined_prediction(self, img_tensor, true_coords=None, save_path=None, original_img=None, img_path=None):
        """Comprehensive visualization of the combined prediction"""
        # Get predictions
        final_lat, final_lon, details = self.combine_agents(img_tensor, return_details=True)
        
        if self.map_img is None:
            print("Map not available. Cannot visualize.")
            return
        
        # Create figure with 4 subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Original Image
        ax1 = plt.subplot(2, 3, 1)
        if original_img is not None:
            # Use the actual original image if provided
            ax1.imshow(original_img)
        elif img_path and os.path.exists(img_path):
            # Try to load from path
            original_img = Image.open(img_path).convert('RGB')
            ax1.imshow(original_img)
        else:
            # Last resort: denormalize the tensor (this is what makes it look filtered)
            img_pil = self.denormalize(img_tensor.cpu().squeeze(0))
            ax1.imshow(img_pil)
        ax1.set_title('Input Image', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # 2. Agent 1 Prediction (Coordinate Regression)
        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(self.map_img)
        
        # Draw grid
        map_height, map_width = self.map_img.size[1], self.map_img.size[0]
        cell_width = map_width / 7
        cell_height = map_height / 7
        
        for i in range(8):
            ax2.axvline(x=i * cell_width, color='gray', linewidth=0.5, alpha=0.3)
            ax2.axhline(y=i * cell_height, color='gray', linewidth=0.5, alpha=0.3)
        
        # Plot Agent 1 prediction
        agent1_lat, agent1_lon = details['agent1_coords']
        # Convert to pixel coordinates using the actual US bounds that match the map
        # The map shows US with these approximate bounds
        map_lon_min, map_lon_max = -125.450687, -67.343249  # From GRID_MAPPER bounds
        map_lat_min, map_lat_max = 24.455005, 49.049081     # From GRID_MAPPER bounds
        
        x_pix = ((agent1_lon - map_lon_min) / (map_lon_max - map_lon_min)) * map_width
        y_pix = ((map_lat_max - agent1_lat) / (map_lat_max - map_lat_min)) * map_height
        
        ax2.scatter(x_pix, y_pix, s=200, c='red', marker='*', edgecolor='white', linewidth=2, 
                   label=f'Agent 1: ({agent1_lat:.2f}, {agent1_lon:.2f})')
        
        # Highlight Agent 1's grid
        a1_grid = details['agent1_grid']
        if a1_grid >= 0:
            a1_y, a1_x = a1_grid // 7, a1_grid % 7
            rect = patches.Rectangle((a1_x * cell_width, a1_y * cell_height),
                                    cell_width, cell_height, linewidth=2,
                                    edgecolor='red', facecolor='red', alpha=0.3)
            ax2.add_patch(rect)
        
        ax2.set_title(f'Agent 1: Coordinate Regression\nGrid {a1_grid}', fontsize=12)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.axis('off')
        
        # 3. Agent 2 Prediction (Grid Classification)
        ax3 = plt.subplot(2, 3, 3)
        ax3.imshow(self.map_img)
        
        for i in range(8):
            ax3.axvline(x=i * cell_width, color='gray', linewidth=0.5, alpha=0.3)
            ax3.axhline(y=i * cell_height, color='gray', linewidth=0.5, alpha=0.3)
        
        # Show Agent 2 probabilities as heatmap
        probs_2d = details['agent2_probs'].reshape(7, 7)
        
        colors = [(1, 1, 1, 0), (0, 0, 1, 0.9)]
        cmap_blue = LinearSegmentedColormap.from_list('agent2', colors)
        
        im2 = ax3.imshow(probs_2d, extent=[0, map_width, map_height, 0], 
                        cmap=cmap_blue, alpha=0.7, vmin=0, vmax=probs_2d.max())
        
        # Highlight top prediction
        a2_top = details['agent2_top_grid']
        a2_y, a2_x = a2_top // 7, a2_top % 7
        rect = patches.Rectangle((a2_x * cell_width, a2_y * cell_height),
                                cell_width, cell_height, linewidth=2,
                                edgecolor='blue', facecolor='none')
        ax3.add_patch(rect)
        
        ax3.set_title(f'Agent 2: Grid Classification\nTop Grid: {a2_top} (Conf: {details["agent2_probs"][a2_top]:.2%})', 
                     fontsize=12)
        ax3.axis('off')
        
        # 4. Likelihood from Agent 1
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(self.map_img)
        
        for i in range(8):
            ax4.axvline(x=i * cell_width, color='gray', linewidth=0.5, alpha=0.3)
            ax4.axhline(y=i * cell_height, color='gray', linewidth=0.5, alpha=0.3)
        
        # Show likelihood heatmap
        likelihood_2d = details['likelihoods'].reshape(7, 7)
        
        colors = [(1, 1, 1, 0), (1, 0.5, 0, 0.9)]
        cmap_orange = LinearSegmentedColormap.from_list('likelihood', colors)
        
        im3 = ax4.imshow(likelihood_2d, extent=[0, map_width, map_height, 0], 
                        cmap=cmap_orange, alpha=0.7, vmin=0, vmax=1)
        
        ax4.set_title('Likelihood Terms from Agent 1\n(Distance-based weights)', fontsize=12)
        ax4.axis('off')
        
        # 5. Combined Scores
        ax5 = plt.subplot(2, 3, 5)
        ax5.imshow(self.map_img)
        
        for i in range(8):
            ax5.axvline(x=i * cell_width, color='gray', linewidth=0.5, alpha=0.3)
            ax5.axhline(y=i * cell_height, color='gray', linewidth=0.5, alpha=0.3)
        
        # Show combined scores
        scores_2d = details['combined_scores'].reshape(7, 7)
        
        colors = [(1, 1, 1, 0), (0.5, 0, 0.5, 0.9)]
        cmap_purple = LinearSegmentedColormap.from_list('combined', colors)
        
        im4 = ax5.imshow(scores_2d, extent=[0, map_width, map_height, 0], 
                        cmap=cmap_purple, alpha=0.7, vmin=0, vmax=scores_2d.max())
        
        ax5.set_title('Combined Scores\n(Agent2 × Likelihood)', fontsize=12)
        ax5.axis('off')
        
        # 6. Final Prediction
        ax6 = plt.subplot(2, 3, 6)
        ax6.imshow(self.map_img)
        
        for i in range(8):
            ax6.axvline(x=i * cell_width, color='gray', linewidth=0.5, alpha=0.3)
            ax6.axhline(y=i * cell_height, color='gray', linewidth=0.5, alpha=0.3)
        
        # Highlight final prediction
        final_grid = details['final_grid']
        f_y, f_x = final_grid // 7, final_grid % 7
        rect = patches.Rectangle((f_x * cell_width, f_y * cell_height),
                                cell_width, cell_height, linewidth=3,
                                edgecolor='purple', facecolor='purple', alpha=0.3)
        ax6.add_patch(rect)
        
        # Plot final coordinate
        map_lon_min, map_lon_max = -125.450687, -67.343249
        map_lat_min, map_lat_max = 24.455005, 49.049081
        
        x_pix_final = ((final_lon - map_lon_min) / (map_lon_max - map_lon_min)) * map_width
        y_pix_final = ((map_lat_max - final_lat) / (map_lat_max - map_lat_min)) * map_height
        
        ax6.scatter(x_pix_final, y_pix_final, s=200, c='purple', marker='*', 
                   edgecolor='white', linewidth=2)
        
        # If true location provided, show it
        if true_coords:
            true_lat, true_lon = true_coords
            x_pix_true = ((true_lon - map_lon_min) / (map_lon_max - map_lon_min)) * map_width
            y_pix_true = ((map_lat_max - true_lat) / (map_lat_max - map_lat_min)) * map_height
            
            ax6.scatter(x_pix_true, y_pix_true, s=200, c='green', marker='o', 
                       edgecolor='white', linewidth=2, label='True Location')
            
            # Draw line between prediction and true
            ax6.plot([x_pix_final, x_pix_true], [y_pix_final, y_pix_true], 
                    'k--', alpha=0.5, linewidth=1)
            
            # Calculate error
            error_km = self.haversine_km(final_lat, final_lon, true_lat, true_lon)
            ax6.set_title(f'Final Combined Prediction\nGrid {final_grid} | Error: {error_km:.1f} km', 
                         fontsize=12, fontweight='bold')
        else:
            ax6.set_title(f'Final Combined Prediction\nGrid {final_grid}', 
                         fontsize=12, fontweight='bold')
        
        if true_coords:
            ax6.legend(loc='upper right', fontsize=8)
        ax6.axis('off')
        
        plt.suptitle('Combined Agent Prediction Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def load_test_data(self):
        """Load test dataset"""
        if self.test_dataset is None:
            print("Loading test dataset...")
            test_csv = self.data_config.get('test_csv', self.data_config['val_csv'])
            
            self.test_dataset = GeoCSVDataset(
                csv_path=test_csv,
                img_size=self.data_config['img_size'],
                train=False,
                lat_min=self.lat_min,
                lat_max=self.lat_max,
                lon_min=self.lon_min,
                lon_max=self.lon_max,
                normalize=False  # Get raw coordinates to understand the format
            )
            
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=4
            )
            print(f"Test set loaded: {len(self.test_dataset)} samples")
            
            # Test one sample to see format
            if len(self.test_dataset) > 0:
                sample_img, sample_coords = self.test_dataset[0]
                if isinstance(sample_coords, torch.Tensor):
                    print(f"Sample coords (raw): {sample_coords[0].item():.4f}, {sample_coords[1].item():.4f}")
                else:
                    print(f"Sample coords (raw): {sample_coords}")
    
    def load_train_data(self):
        """Load training dataset for random sampling"""
        if self.train_dataset is None:
            print("Loading training dataset for sampling...")
    
            self.train_dataset = GeoCSVDataset(
                csv_path=self.data_config['train_csv'],
                img_size=self.data_config['img_size'],
                train=False,  # Don't use augmentation for testing
                lat_min=self.lat_min,
                lat_max=self.lat_max,
                lon_min=self.lon_min,
                lon_max=self.lon_max,
                normalize=False  # Get raw coordinates
            )
            print(f"Training set loaded: {len(self.train_dataset)} samples")
            
            # Test one sample
            if len(self.train_dataset) > 0:
                sample_img, sample_coords = self.train_dataset[0]
                if isinstance(sample_coords, torch.Tensor):
                    print(f"Sample coords (raw): {sample_coords[0].item():.4f}, {sample_coords[1].item():.4f}")
                else:
                    print(f"Sample coords (raw): {sample_coords}")
    
    def test_random_image(self):
        """Test on a random image"""
        self.load_train_data()
        
        idx = random.randint(0, len(self.train_dataset) - 1)
        
        # Get the actual image path if available
        original_img = None
        if hasattr(self.train_dataset, 'df'):
            img_info = self.train_dataset.df.iloc[idx]
            if 'image_path' in img_info:
                img_path = img_info['image_path']
                if os.path.exists(img_path):
                    original_img = Image.open(img_path).convert('RGB')
                    print(f"\nTesting on image: {img_path}")
        
        img_tensor, coords = self.train_dataset[idx]
        
        # Debug: Let's see what we're getting
        print(f"\nDebug - Raw coords from dataset: {coords}")
        
        # Get true coordinates 
        if isinstance(coords, torch.Tensor):
            coord_0 = coords[0].item()
            coord_1 = coords[1].item()
        else:
            coord_0, coord_1 = coords[0], coords[1]
            
        print(f"Debug - Coord values: {coord_0}, {coord_1}")
        print(f"Debug - Training data bounds: lat [{self.lat_min:.2f}, {self.lat_max:.2f}], lon [{self.lon_min:.2f}, {self.lon_max:.2f}]")
        
        # Check if these are normalized or raw coordinates
        # Raw lat should be in range ~[24, 49] and lon in range ~[-124, -67]
        if 20 <= abs(coord_0) <= 60 and 60 <= abs(coord_1) <= 130:
            # These look like raw lat/lon in degrees
            true_lat = coord_0
            true_lon = coord_1
            print(f"Debug - Detected raw coordinates")
        elif -2 <= coord_0 <= 2 and -2 <= coord_1 <= 2:
            # These look like normalized coordinates in [-1, 1] range
            true_lat = (coord_0 + 1.0) * 0.5 * (self.lat_max - self.lat_min) + self.lat_min
            true_lon = (coord_1 + 1.0) * 0.5 * (self.lon_max - self.lon_min) + self.lon_min
            print(f"Debug - Detected normalized coordinates, denormalizing...")
        else:
            # Unclear, assume raw
            true_lat = coord_0
            true_lon = coord_1
            print(f"Debug - Unknown coordinate format, assuming raw")
        
        print(f"\nTesting on random image (index {idx})")
        print(f"True location: ({true_lat:.4f}, {true_lon:.4f})")
        print(f"True grid: {GRID_MAPPER.latlon_to_grid(true_lat, true_lon)}")
        
        # Get predictions
        final_lat, final_lon, details = self.combine_agents(img_tensor, return_details=True)
        
        # Calculate errors
        agent1_lat, agent1_lon = details['agent1_coords']
        agent1_error = self.haversine_km(true_lat, true_lon, agent1_lat, agent1_lon)
        combined_error = self.haversine_km(true_lat, true_lon, final_lat, final_lon)
        
        print(f"\nAgent 1 prediction: ({agent1_lat:.4f}, {agent1_lon:.4f})")
        print(f"  Error: {agent1_error:.1f} km")
        print(f"  Grid: {details['agent1_grid']}")
        
        print(f"\nAgent 2 top prediction: Grid {details['agent2_top_grid']}")
        print(f"  Confidence: {details['agent2_probs'][details['agent2_top_grid']]:.2%}")
        
        print(f"\nCombined prediction: ({final_lat:.4f}, {final_lon:.4f})")
        print(f"  Error: {combined_error:.1f} km")
        print(f"  Final grid: {details['final_grid']}")
        
        improvement = (agent1_error - combined_error)
        print(f"\nImprovement over Agent 1: {improvement:.1f} km ({improvement/agent1_error*100:.1f}%)")
        
        # Visualize
        self.visualize_combined_prediction(
            img_tensor, 
            true_coords=(true_lat, true_lon),
            save_path='combined_test_results/random_prediction.png',
            original_img=original_img
        )

    def test_specific_image(self, path):
        """Test on a specific image"""
        self.load_train_data()
        
        # Check if path exists
        if not os.path.exists(path):
            print(f"Error: Image path '{path}' does not exist")
            return
        
        # Load the specific image
        original_img = Image.open(path).convert('RGB')
        print(f"\nTesting on image: {path}")
        
        # Extract coordinates from filename
        # Format: "dataset/united_states/grid-5-3/29.213949142947,-97.83494812509969.jpg"
        filename = os.path.basename(path)  # "29.213949142947,-97.83494812509969.jpg"
        filename_without_ext = os.path.splitext(filename)[0]  # "29.213949142947,-97.83494812509969"
        
        try:
            # Split by comma to get lat and lon
            lat_str, lon_str = filename_without_ext.split(',')
            true_lat = float(lat_str)
            true_lon = float(lon_str)
            print(f"Extracted coordinates from filename: lat={true_lat}, lon={true_lon}")
            print(f"True grid: {GRID_MAPPER.latlon_to_grid(true_lat, true_lon)}")
            
        except Exception as e:
            print(f"Error parsing coordinates from filename: {e}")
            print("Proceeding without ground truth coordinates...")
            true_lat, true_lon = None, None
        
        # Apply transforms to prepare the image for the model
        img_tensor = self.transform(original_img)
        
        # Get predictions
        final_lat, final_lon, details = self.combine_agents(img_tensor, return_details=True)
        
        # Print results
        agent1_lat, agent1_lon = details['agent1_coords']
        print(f"\nAgent 1 prediction: ({agent1_lat:.4f}, {agent1_lon:.4f})")
        print(f"  Grid: {details['agent1_grid']}")
        
        print(f"\nAgent 2 top prediction: Grid {details['agent2_top_grid']}")
        print(f"  Confidence: {details['agent2_probs'][details['agent2_top_grid']]:.2%}")
        
        print(f"\nCombined prediction: ({final_lat:.4f}, {final_lon:.4f})")
        print(f"  Final grid: {details['final_grid']}")
        
        # Calculate errors if we have ground truth
        if true_lat is not None and true_lon is not None:
            agent1_error = self.haversine_km(true_lat, true_lon, agent1_lat, agent1_lon)
            combined_error = self.haversine_km(true_lat, true_lon, final_lat, final_lon)
            
            print(f"\nAgent 1 error: {agent1_error:.1f} km")
            print(f"Combined error: {combined_error:.1f} km")
            
            improvement = (agent1_error - combined_error)
            if agent1_error > 0:
                print(f"Improvement over Agent 1: {improvement:.1f} km ({improvement/agent1_error*100:.1f}%)")
            
            # Visualize with ground truth
            self.visualize_combined_prediction(
                img_tensor, 
                true_coords=(true_lat, true_lon),
                save_path='combined_test_results/specific_prediction.png',
                original_img=original_img,
                img_path=path
            )
        else:
            # Visualize without ground truth
            self.visualize_combined_prediction(
                img_tensor,
                true_coords=None,
                save_path='combined_test_results/specific_prediction.png',
                original_img=original_img,
                img_path=path
            )
    
    def run_full_test(self):
        """Run full test set evaluation"""
        self.load_test_data()
        
        print("\nRunning full test set evaluation...")
        
        agent1_errors = []
        combined_errors = []
        agent2_correct = []
        
        with torch.no_grad():
            for images, coords in tqdm(self.test_loader, desc="Testing"):
                for i in range(images.size(0)):
                    img = images[i]
                    
                    # Get true coordinates
                    # Check if coordinates are normalized
                    if isinstance(coords, tuple):
                        coord_val = coords[0][i].item()
                        coord_lon = coords[1][i].item()
                    else:
                        coord_val = coords[i, 0].item()
                        coord_lon = coords[i, 1].item()
                    
                    # Check if values are in normalized range [-1, 1] or actual coordinates
                    if -1.5 <= coord_val <= 1.5 and -1.5 <= coord_lon <= 1.5:
                        # Denormalize from [-1, 1]
                        true_lat = (coord_val + 1.0) * 0.5 * (self.lat_max - self.lat_min) + self.lat_min
                        true_lon = (coord_lon + 1.0) * 0.5 * (self.lon_max - self.lon_min) + self.lon_min
                    else:
                        # Already in degrees
                        true_lat = coord_val
                        true_lon = coord_lon
                    
                    true_grid = GRID_MAPPER.latlon_to_grid(true_lat, true_lon)
                    
                    # Get predictions
                    final_lat, final_lon, details = self.combine_agents(img, return_details=True)
                    
                    # Calculate errors
                    agent1_lat, agent1_lon = details['agent1_coords']
                    agent1_error = self.haversine_km(true_lat, true_lon, agent1_lat, agent1_lon)
                    combined_error = self.haversine_km(true_lat, true_lon, final_lat, final_lon)
                    
                    agent1_errors.append(agent1_error)
                    combined_errors.append(combined_error)
                    agent2_correct.append(details['agent2_top_grid'] == true_grid)
        
        # Calculate statistics
        agent1_errors = np.array(agent1_errors)
        combined_errors = np.array(combined_errors)
        agent2_acc = np.mean(agent2_correct) * 100
        
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        
        print("\nAgent 1 (Coordinate Regression):")
        print(f"  Mean error: {np.mean(agent1_errors):.2f} km")
        print(f"  Median error: {np.median(agent1_errors):.2f} km")
        print(f"  Std dev: {np.std(agent1_errors):.2f} km")
        
        print(f"\nAgent 2 (Grid Classification):")
        print(f"  Grid accuracy: {agent2_acc:.1f}%")
        
        print(f"\nCombined Agents:")
        print(f"  Mean error: {np.mean(combined_errors):.2f} km")
        print(f"  Median error: {np.median(combined_errors):.2f} km")
        print(f"  Std dev: {np.std(combined_errors):.2f} km")
        
        improvement = (np.mean(agent1_errors) - np.mean(combined_errors)) / np.mean(agent1_errors) * 100
        print(f"\nImprovement over Agent 1: {improvement:.1f}%")
        
        # Save results
        results = {
            'test_size': len(self.test_dataset),
            'agent1_mean_error_km': float(np.mean(agent1_errors)),
            'agent1_median_error_km': float(np.median(agent1_errors)),
            'agent2_accuracy': float(agent2_acc),
            'combined_mean_error_km': float(np.mean(combined_errors)),
            'combined_median_error_km': float(np.median(combined_errors)),
            'improvement_percent': float(improvement)
        }
        
        with open('combined_test_results/test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create comparison plot
        self.plot_error_comparison(agent1_errors, combined_errors)
        
        print("\nResults saved to combined_test_results/")
    
    def plot_error_comparison(self, agent1_errors, combined_errors):
        """Plot error distribution comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Histogram
        ax = axes[0]
        ax.hist(agent1_errors, bins=50, alpha=0.5, label='Agent 1', color='blue')
        ax.hist(combined_errors, bins=50, alpha=0.5, label='Combined', color='purple')
        ax.set_xlabel('Error (km)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[1]
        box_data = [agent1_errors, combined_errors]
        bp = ax.boxplot(box_data, labels=['Agent 1', 'Combined'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lavender')
        ax.set_ylabel('Error (km)')
        ax.set_title('Error Distribution Comparison')
        ax.grid(True, alpha=0.3)
        
        # CDF
        ax = axes[2]
        sorted_a1 = np.sort(agent1_errors)
        sorted_comb = np.sort(combined_errors)
        cdf = np.arange(1, len(sorted_a1) + 1) / len(sorted_a1)
        
        ax.plot(sorted_a1, cdf, label='Agent 1', linewidth=2, color='blue')
        ax.plot(sorted_comb, cdf, label='Combined', linewidth=2, color='purple')
        ax.set_xlabel('Error (km)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('combined_test_results/error_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()


def print_menu():
    """Print interactive menu"""
    print("\n" + "="*50)
    print("COMBINED AGENT TESTER")
    print("="*50)
    print("1. Test on random image")
    print("2. Test on specific image path")
    print("3. Run full test set evaluation")
    print("4. Exit")
    print("-"*50)


def main():
    """Main function"""
    AGENT1_CHECKPOINT = "agent_1/checkpoints/agent1/best_checkpoint.pt"
    AGENT2_CHECKPOINT = "agent_2/saved_models/latest.pth"
    DATA_CONFIG = "configs/data.yaml"
    TRAIN_CONFIG = "configs/train_agent1.yaml"
    
    print("Initializing Combined Agent Tester...")
    tester = CombinedAgentTester(
        agent1_checkpoint=AGENT1_CHECKPOINT,
        agent2_checkpoint=AGENT2_CHECKPOINT,
        data_config_path=DATA_CONFIG,
        train_config_path=TRAIN_CONFIG
    )
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            tester.test_random_image()
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            tester.test_specific_image(image_path)
        elif choice == '3':
            tester.run_full_test()
        elif choice == '4':
            print("\nExiting... Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-4.")
        
        if choice in ['1', '2', '3']:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
