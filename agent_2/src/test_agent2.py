import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datasets_agent2 import StreetViewGridDataset
from PIL import Image
import json
import random

# -----------------------------
# Configuration
# -----------------------------
CONFIG_PATH = "configs/data.yaml"
MODEL_PATH = "agent_2/saved_models/latest.pth"
MAP_PATH = "data/NA_map.png"
BATCH_SIZE = 64
CORES = 4

class StreetViewTester:
    def __init__(self):
        """Initialize the tester with model and data"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "mps" if torch.backends.mps.is_available() else "cpu")
        print(f'Using device: {self.device.type}')
        
        # Load config
        with open(CONFIG_PATH, "r") as f:
            self.cfg = yaml.safe_load(f)
        
        self.img_size = self.cfg["img_size"]
        
        # Setup transforms
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        
        # Load model
        print(f"Loading model from {MODEL_PATH}...")
        self.model = self.load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load map if exists
        self.map_img = None
        if os.path.exists(MAP_PATH):
            self.map_img = Image.open(MAP_PATH)
            print(f"Map loaded from {MAP_PATH}")
        else:
            print(f"Warning: Map not found at {MAP_PATH}")
        
        # Initialize datasets (lazy loading)
        self.test_dataset = None
        self.test_loader = None
        self.train_dataset = None  # For random sampling
        
        print("\nTester initialized successfully!")
    
    def load_model(self):
        """Load the trained model"""
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 49),
        )
        
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        return model
    
    def load_test_data(self):
        """Load test dataset (lazy loading)"""
        if self.test_dataset is None:
            print("Loading test dataset...")
            test_csv = self.cfg["test_csv"]
            self.test_dataset = StreetViewGridDataset(test_csv, self.img_size, transform=self.transform)
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=CORES,
                persistent_workers=True,
            )
            print(f"Test set loaded: {len(self.test_dataset)} samples")
    
    def load_train_data(self):
        """Load training dataset for random sampling"""
        if self.train_dataset is None:
            print("Loading training dataset for sampling...")
            train_csv = self.cfg["train_csv"]
            self.train_dataset = StreetViewGridDataset(train_csv, self.img_size, transform=self.transform)
            print(f"Training set loaded: {len(self.train_dataset)} samples")
    
    def predict_single_image(self, image_path=None, image_tensor=None):
        """Predict on a single image"""
        if image_path:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        elif image_tensor is not None:
            img_tensor = image_tensor.unsqueeze(0).to(self.device)
        else:
            raise ValueError("Either image_path or image_tensor must be provided")
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Get top-5 predictions
            top5_probs, top5_indices = torch.topk(probs, 5)
            
        return {
            'top1': top5_indices[0, 0].item(),
            'top5': top5_indices[0].cpu().numpy(),
            'top5_probs': top5_probs[0].cpu().numpy(),
            'all_probs': probs[0].cpu().numpy()
        }
    
    def visualize_prediction_on_map(self, prediction_data, true_label=None, save_path=None, original_image=None):
        """Visualize prediction on the map with 7x7 grid overlay"""
        if self.map_img is None:
            print("Map not available. Cannot visualize.")
            return
        
        # Create figure with 2 subplots if we have original image, 1 otherwise
        if original_image is not None:
            fig, (ax_img, ax1) = plt.subplots(1, 2, figsize=(16, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
        
        # Display original image if provided
        if original_image is not None:
            ax_img.imshow(original_image)
            ax_img.set_title('Input Image', fontsize=14)
            ax_img.axis('off')
        
        # Left plot: Map with grid overlay and prediction
        ax1.imshow(self.map_img)
        
        # Get map dimensions
        map_height, map_width = self.map_img.size[1], self.map_img.size[0]
        
        # Calculate grid cell dimensions
        cell_width = map_width / 7
        cell_height = map_height / 7
        
        # Draw grid lines
        for i in range(8):
            ax1.axvline(x=i * cell_width, color='gray', linewidth=0.5, alpha=0.5)
            ax1.axhline(y=i * cell_height, color='gray', linewidth=0.5, alpha=0.5)
        
        # Highlight predictions with heat map
        probs = prediction_data['all_probs'].reshape(7, 7)
        
        # Add grid labels with prediction probabilities
        for i in range(7):
            for j in range(7):
                grid_idx = i * 7 + j
                prob_value = probs[i, j]
                
                # Show grid index and probability percentage
                if prob_value > 0.01:  # Only show if probability > 1%
                    label_text = f"{grid_idx}\n{prob_value*100:.1f}%"
                    fontsize = 10 if grid_idx == prediction_data['top1'] else 8
                    fontweight = 'bold' if grid_idx == prediction_data['top1'] else 'normal'
                else:
                    label_text = str(grid_idx)
                    fontsize = 8
                    fontweight = 'normal'
                
                ax1.text(j * cell_width + cell_width/2, i * cell_height + cell_height/2, 
                        label_text, ha='center', va='center', 
                        fontsize=fontsize, color='white', weight=fontweight,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        # Create custom colormap (transparent to red)
        colors = [(1, 1, 1, 0), (1, 0, 0, 0.9)]  # White (transparent) to Red
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('prediction', colors, N=n_bins)
        
        # Overlay probability heatmap
        im = ax1.imshow(probs, extent=[0, map_width, map_height, 0], 
                       cmap=cmap, alpha=0.7, vmin=0, vmax=probs.max())
        
        # Highlight top-1 prediction
        top1 = prediction_data['top1']
        top1_y, top1_x = top1 // 7, top1 % 7
        rect = patches.Rectangle((top1_x * cell_width, top1_y * cell_height),
                                cell_width, cell_height, linewidth=3,
                                edgecolor='yellow', facecolor='none')
        ax1.add_patch(rect)
        
        # If true label provided, show it
        if true_label is not None:
            true_y, true_x = true_label // 7, true_label % 7
            rect_true = patches.Rectangle((true_x * cell_width, true_y * cell_height),
                                         cell_width, cell_height, linewidth=3,
                                         edgecolor='green', facecolor='none', linestyle='--')
            ax1.add_patch(rect_true)
        
        ax1.set_title('Prediction on Map (Yellow=Predicted, Green=True)', fontsize=14)
        ax1.axis('off')
        
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Prediction Probability')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Prediction map saved to {save_path}")
        
        plt.show()
    
    def visualize_kernels(self, image_path=None):
        """Visualize convolutional kernels/feature maps for an image"""
        # Load image
        if image_path is None:
            # Use random image
            self.load_train_data()
            idx = random.randint(0, len(self.train_dataset) - 1)
            img_tensor, true_label = self.train_dataset[idx]
            img_info = self.train_dataset.df.iloc[idx]
            img_path_str = img_info['image_path'] if 'image_path' in img_info else f"Image {idx}"
            
            # Load original image for display
            if 'image_path' in img_info and os.path.exists(img_info['image_path']):
                original_img = Image.open(img_info['image_path']).convert('RGB')
            else:
                # Convert tensor back to PIL image for display
                denorm = transforms.Compose([
                    transforms.Normalize(mean=[-m/s for m, s in zip(self.mean, self.std)],
                                       std=[1/s for s in self.std]),
                    transforms.ToPILImage()
                ])
                original_img = denorm(img_tensor)
            
            print(f"\nVisualizing kernels for random image: {img_path_str}")
            print(f"True grid label: {true_label} (Grid {true_label//7},{true_label%7})")
        else:
            if not os.path.exists(image_path):
                print(f"Error: Image not found at {image_path}")
                return
            
            original_img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(original_img)
            print(f"\nVisualizing kernels for: {image_path}")
        
        # Prepare input
        img_batch = img_tensor.unsqueeze(0).to(self.device)
        
        # Hook to capture activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for different layers
        layers_to_visualize = {
            'conv1': self.model.conv1,
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4
        }
        
        handles = []
        for name, layer in layers_to_visualize.items():
            handle = layer.register_forward_hook(get_activation(name))
            handles.append(handle)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(img_batch)
            pred_probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(pred_probs, dim=1).item()
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Ensure image is exactly 224x224
        if original_img.size != (224, 224):
            original_img = original_img.resize((224, 224), Image.BILINEAR)
        
        # Show original image
        ax_img = plt.subplot(3, 6, 1)
        ax_img.imshow(original_img)
        ax_img.set_title(f'Input Image (224x224)\nPredicted: Grid {pred_class}', fontsize=10)
        ax_img.axis('off')
        
        # Visualize conv1 (first layer) - these are 112x112 after conv and pool
        conv1_acts = activations['conv1'][0].cpu()  # Remove batch dimension
        n_kernels = min(11, conv1_acts.shape[0])  # Show max 11 kernels to fit in grid
        
        # Get conv1 output size for resizing original image
        conv1_size = conv1_acts.shape[1]  # Height/width of feature maps
        
        for i in range(n_kernels):
            ax = plt.subplot(3, 6, i+2)  # Simply use i+2 to i+12
            act = conv1_acts[i].numpy()
            # Normalize for display
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)
            ax.imshow(act, cmap='viridis')
            ax.set_title(f'Conv1 K{i} ({conv1_size}x{conv1_size})', fontsize=8)
            ax.axis('off')
        
        # Show later layer activations with 224x224 image and native resolution overlays
        layer_positions = {
            'layer1': 13,  # Position 13 in grid
            'layer2': 14,
            'layer3': 15,
            'layer4': 16
        }
        
        for layer_name, pos in layer_positions.items():
            ax = plt.subplot(3, 6, pos)
            acts = activations[layer_name][0].cpu()
            
            # Get the size of this layer's feature maps
            feature_size = acts.shape[1]  # Height/width of feature maps
            
            # Average across channels and normalize
            avg_act = acts.mean(dim=0).numpy()
            avg_act = (avg_act - avg_act.min()) / (avg_act.max() - avg_act.min() + 1e-8)
            
            # Display 224x224 image
            ax.imshow(original_img)
            
            # Overlay the low-res activation with nearest neighbor interpolation (blocky)
            # extent=[left, right, bottom, top] in data coordinates
            ax.imshow(avg_act, cmap='hot', alpha=0.5, 
                     extent=[0, 224, 224, 0],
                     interpolation='nearest')
            
            ax.set_title(f'{layer_name} ({feature_size}x{feature_size})', fontsize=10)
            ax.axis('off')
        
        # Show CAM with 224x224 image
        ax_final = plt.subplot(3, 6, 18)  # Last position
        final_acts = activations['layer4'][0].cpu()
        
        # Get activation for predicted class
        gap_weights = self.model.fc[1].weight[pred_class].cpu()
        
        # Compute class activation map
        cam = torch.zeros(final_acts.shape[1:])
        for i, w in enumerate(gap_weights[:final_acts.shape[0]]):
            cam += w * final_acts[i]
        
        cam = cam.detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Display 224x224 image with blocky CAM overlay
        ax_final.imshow(original_img)
        ax_final.imshow(cam, cmap='jet', alpha=0.5,
                       extent=[0, 224, 224, 0],
                       interpolation='nearest')
        
        ax_final.set_title(f'CAM ({cam.shape[0]}x{cam.shape[0]})\nfor Grid {pred_class}', fontsize=10)
        ax_final.axis('off')
        
        plt.suptitle('Convolutional Feature Visualizations', fontsize=14)
        plt.tight_layout()
        
        # Save
        save_path = 'agent_2/test_results/kernel_visualization.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Kernel visualization saved to {save_path}")
        
        plt.show()
    
    def test_random_image(self):
        """Test on a random image from the dataset"""
        self.load_train_data()  # We'll use training set for random sampling
        
        # Get random sample
        idx = random.randint(0, len(self.train_dataset) - 1)
        img_tensor, true_label = self.train_dataset[idx]
        
        # Get image path for display
        img_info = self.train_dataset.df.iloc[idx]
        img_path = img_info['image_path'] if 'image_path' in img_info else f"Image {idx}"
        
        # Load original image for display
        original_img = None
        if 'image_path' in img_info and os.path.exists(img_info['image_path']):
            original_img = Image.open(img_info['image_path']).convert('RGB')
        
        print(f"\nTesting on random image: {img_path}")
        print(f"True grid label: {true_label} (Grid {true_label//7},{true_label%7})")
        
        # Predict
        prediction = self.predict_single_image(image_tensor=img_tensor)
        
        # Print results
        print(f"\nPrediction Results:")
        print(f"Top-1 Prediction: Grid {prediction['top1']} ({prediction['top1']//7},{prediction['top1']%7})")
        print(f"Confidence: {prediction['top5_probs'][0]:.3f}")
        
        if prediction['top1'] == true_label:
            print("✓ CORRECT!")
        else:
            # Check if in top-5
            if true_label in prediction['top5']:
                rank = np.where(prediction['top5'] == true_label)[0][0] + 1
                print(f"✗ Incorrect, but true label is in top-{rank}")
            else:
                print("✗ Incorrect, true label not in top-5")
        
        # Visualize
        self.visualize_prediction_on_map(prediction, true_label, 
                                        save_path='agent_2/test_results/random_prediction.png',
                                        original_image=original_img)
    
    def test_specific_image(self):
        """Test on a specific image path"""
        image_path = input("Enter image path: ").strip()
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return
        
        print(f"\nTesting on: {image_path}")
        
        # Load original image for display
        original_img = Image.open(image_path).convert('RGB')
        
        # Predict
        prediction = self.predict_single_image(image_path=image_path)
        
        # Print results
        print(f"\nPrediction Results:")
        print(f"Top-1 Prediction: Grid {prediction['top1']} ({prediction['top1']//7},{prediction['top1']%7})")
        print(f"Confidence: {prediction['top5_probs'][0]:.3f}")
        
        print("\nTop-5 Predictions:")
        for i, (idx, prob) in enumerate(zip(prediction['top5'], prediction['top5_probs'])):
            print(f"  {i+1}. Grid {idx} ({idx//7},{idx%7}): {prob:.3f}")
        
        # Visualize
        self.visualize_prediction_on_map(prediction, 
                                        save_path='agent_2/test_results/specific_prediction.png',
                                        original_image=original_img)
    
    def run_full_test(self):
        """Run full test set evaluation"""
        self.load_test_data()
        
        print("\nRunning full test set evaluation...")
        all_predictions = []
        all_labels = []
        top_k_correct = {1: [], 3: [], 5: []}
        
        with torch.no_grad():
            for imgs, labels in tqdm(self.test_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                outputs = self.model(imgs)
                _, predicted = torch.max(outputs, 1)
                
                # Calculate top-k accuracy
                for k in [1, 3, 5]:
                    topk = outputs.topk(k, dim=1).indices
                    correct = topk.eq(labels.view(-1, 1)).any(dim=1)
                    top_k_correct[k].extend(correct.cpu().numpy())
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        print("\n" + "="*50)
        print("TEST SET RESULTS")
        print("="*50)
        
        # Accuracies
        for k in [1, 3, 5]:
            acc = np.mean(top_k_correct[k]) * 100
            print(f"Top-{k} Accuracy: {acc:.2f}%")
        
        # Save confusion matrix
        self.save_confusion_matrix(labels, predictions)
        
        # Save results
        os.makedirs("agent_2/test_results", exist_ok=True)
        results = {
            'test_size': len(self.test_dataset),
            'top1_accuracy': float(np.mean(top_k_correct[1]) * 100),
            'top3_accuracy': float(np.mean(top_k_correct[3]) * 100),
            'top5_accuracy': float(np.mean(top_k_correct[5]) * 100),
        }
        
        with open('agent_2/test_results/test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nResults saved to agent_2/test_results/")
    
    def save_confusion_matrix(self, y_true, y_pred):
        """Save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        
        # Handle division by zero - normalize only rows with non-zero sums
        cm_normalized = np.zeros_like(cm, dtype=float)
        row_sums = cm.sum(axis=1)
        
        # Only normalize rows that have samples
        non_zero_rows = row_sums > 0
        cm_normalized[non_zero_rows] = cm[non_zero_rows].astype('float') / row_sums[non_zero_rows][:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=False, cmap='Blues', cbar=True,
                   square=True, vmin=0, vmax=1)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Grid')
        plt.xlabel('Predicted Grid')
        
        tick_marks = np.arange(0, 49, 5)
        plt.xticks(tick_marks + 0.5, tick_marks, fontsize=8)
        plt.yticks(tick_marks + 0.5, tick_marks, fontsize=8)
        
        plt.tight_layout()
        plt.savefig('agent_2/test_results/confusion_matrix.png', dpi=150)
        plt.close()
        print("Confusion matrix saved")

def print_menu():
    """Print the interactive menu"""
    print("\n" + "="*50)
    print("STREET VIEW AGENT TESTER")
    print("="*50)
    print("1. Test on random image")
    print("2. Test on specific image path")
    print("3. Run full test set evaluation")
    print("4. Visualize kernels/features")
    print("5. Exit")
    print("-"*50)

def main():
    """Main interactive loop"""
    print("Initializing Street View Agent Tester...")
    tester = StreetViewTester()
    
    # Create output directory
    os.makedirs("agent_2/test_results", exist_ok=True)
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            tester.test_random_image()
        elif choice == '2':
            tester.test_specific_image()
        elif choice == '3':
            tester.run_full_test()
        elif choice == '4':
            viz_choice = input("Press Enter for random image or enter image path: ").strip()
            if viz_choice:
                tester.visualize_kernels(viz_choice)
            else:
                tester.visualize_kernels()
        elif choice == '5':
            print("\nExiting... Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1-5.")
        
        if choice in ['1', '2', '3', '4']:
            input("\nPress Enter to continue...")

if __name__ == '__main__':
    main()