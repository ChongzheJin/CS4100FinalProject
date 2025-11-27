import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# GPU Configuration
print("="*60)
print("CONFIGURING GPU")
print("="*60)

# Enable GPU memory growth to avoid allocating all GPU memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Method 1: Memory growth (recommended)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Method 2: Limit memory (optional - use if method 1 causes issues)
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Limit to 4GB
        # )
        
        print(f"✅ Found {len(gpus)} GPU(s) and configured for use:")
        for gpu in gpus:
            print(f"   - {gpu}")
        
        # Force TensorFlow to use GPU
        with tf.device('/GPU:0'):
            test = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            print(f"✅ GPU test successful: {test.device}")
            
    except RuntimeError as e:
        print(f"⚠️ GPU configuration error: {e}")
else:
    print("❌ No GPUs detected. Using CPU.")
    print("\nTo enable GPU:")
    print("  For NVIDIA: pip install tensorflow[and-cuda]")
    print("  For Mac M1/M2/M3: pip install tensorflow-metal")
    
# Set TensorFlow to use less verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class GeoGuessrAI:
    def __init__(self, base_dir="../../dataset/united_states", img_height=224, img_width=224, 
                 batch_size=32, min_images_threshold=200):
        """
        Initialize the GeoGuessr AI system.
        
        Args:
            base_dir: Base directory containing grid folders
            img_height: Height to resize images to
            img_width: Width to resize images to
            batch_size: Batch size for training
            min_images_threshold: Minimum images required for a grid to be included in training
        """
        self.base_dir = base_dir
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.min_images_threshold = min_images_threshold
        self.model = None
        self.class_names = []
        self.num_classes = 0
        self.excluded_grids = []
        self.class_weights = None
        self.grid_dims = "unknown"
        
    def detect_grid_dimensions(self):
        """
        Automatically detect the grid dimensions from the class names.
        Returns a string like '4x5' or '5x5'
        """
        if not self.class_names:
            return "unknown"
        
        max_row = 0
        max_col = 0
        
        for class_name in self.class_names:
            parts = class_name.split('-')
            if len(parts) == 3:
                row = int(parts[1])
                col = int(parts[2])
                max_row = max(max_row, row)
                max_col = max(max_col, col)
        
        # Grid dimensions are max + 1 (since indexing starts at 0)
        rows = max_row + 1
        cols = max_col + 1
        
        self.grid_dims = f"{rows}x{cols}"
        return self.grid_dims
        
    def analyze_grid_coverage(self):
        """
        Analyze the current grid coverage and identify grids to exclude.
        Returns a dictionary of grid counts and exclusion recommendations.
        """
        print("\n" + "="*60)
        print("ANALYZING GRID COVERAGE")
        print("="*60)
        
        grid_counts = {}
        for folder in os.listdir(self.base_dir):
            folder_path = os.path.join(self.base_dir, folder)
            if os.path.isdir(folder_path) and folder.startswith("grid-"):
                jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
                grid_counts[folder] = len(jpg_files)
        
        # Sort grids by name for consistent ordering
        sorted_grids = sorted(grid_counts.items(), 
                            key=lambda x: (int(x[0].split('-')[1]), int(x[0].split('-')[2])))
        
        # Categorize grids
        excluded = []
        included = []
        
        print(f"\nGrid Analysis (threshold: {self.min_images_threshold} images):")
        print("-" * 40)
        
        for grid, count in sorted_grids:
            if count < self.min_images_threshold:
                excluded.append(grid)
                status = "❌ EXCLUDED"
            else:
                included.append(grid)
                status = "✅ INCLUDED"
            
            # Extract grid coordinates for better display
            grid_num = grid.replace("grid-", "")
            print(f"  {grid_num:6} : {count:5} images  {status}")
        
        self.excluded_grids = excluded
        
        print("-" * 40)
        print(f"Total grids: {len(grid_counts)}")
        print(f"Included: {len(included)} grids")
        print(f"Excluded: {len(excluded)} grids")
        
        if excluded:
            print(f"\nExcluded grids: {', '.join([g.replace('grid-', '') for g in excluded])}")
        
        # Calculate total images
        total_included = sum(grid_counts[g] for g in included)
        total_excluded = sum(grid_counts[g] for g in excluded)
        print(f"\nTotal images included: {total_included:,}")
        print(f"Total images excluded: {total_excluded:,}")
        
        return grid_counts, included, excluded
    
    def handle_unfit_images(self):
        """
        Redistribute unfit images to their nearest grids or remove them.
        """
        unfit_dir = os.path.join(self.base_dir, "unfit")
        if not os.path.exists(unfit_dir):
            return
        
        unfit_files = [f for f in os.listdir(unfit_dir) if f.endswith('.jpg')]
        if not unfit_files:
            return
        
        print(f"\nFound {len(unfit_files)} unfit images")
        
        # For now, we'll just report them - you can implement redistribution logic
        # based on your specific geographic boundaries
        print("Note: Unfit images will be excluded from training")
        print("Consider manually reviewing and redistributing these images")
        
    def prepare_train_test_split(self, test_split=0.2, val_split=0.1, seed=42):
        """
        Split the data into training, validation, and test sets while maintaining class balance.
        Automatically excludes grids below the threshold.
        
        Args:
            test_split: Fraction of data for testing
            val_split: Fraction of training data for validation
            seed: Random seed for reproducibility
        """
        print("\n" + "="*60)
        print("PREPARING TRAIN/VAL/TEST SPLIT")
        print("="*60)
        
        # First analyze coverage
        grid_counts, included_grids, excluded_grids = self.analyze_grid_coverage()
        
        # Handle unfit images
        self.handle_unfit_images()
        
        # Create directories
        train_dir = os.path.join(self.base_dir, "train")
        val_dir = os.path.join(self.base_dir, "val")
        test_dir = os.path.join(self.base_dir, "test")
        
        # Clean up existing directories
        for dir_path in [train_dir, val_dir, test_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        
        # Process only included grids
        self.class_names = sorted([g for g in included_grids])
        self.num_classes = len(self.class_names)
        
        # Detect grid dimensions
        grid_dims = self.detect_grid_dimensions()
        print(f"\nDetected grid configuration: {grid_dims}")
        
        print(f"Processing {self.num_classes} grids for training...")
        
        # Calculate class weights based on image counts
        class_counts_list = []
        for grid in self.class_names:
            class_counts_list.append(grid_counts[grid])
        
        # Calculate inverse frequency weights
        total_images = sum(class_counts_list)
        self.class_weights = {}
        for i, count in enumerate(class_counts_list):
            # Inverse frequency weighting with smoothing
            self.class_weights[i] = (total_images / (self.num_classes * count)) ** 0.5
        
        # Process each grid
        total_train = 0
        total_val = 0
        total_test = 0
        
        print("\nSplitting data...")
        
        # Use tqdm for progress tracking
        for grid in tqdm(self.class_names, desc="Processing grids", unit="grid"):
            grid_path = os.path.join(self.base_dir, grid)
            
            # Create subdirectories
            train_grid_dir = os.path.join(train_dir, grid)
            val_grid_dir = os.path.join(val_dir, grid)
            test_grid_dir = os.path.join(test_dir, grid)
            os.makedirs(train_grid_dir)
            os.makedirs(val_grid_dir)
            os.makedirs(test_grid_dir)
            
            # Get all jpg files
            jpg_files = [f for f in os.listdir(grid_path) if f.endswith('.jpg')]
            
            # First split: separate test set
            train_val_files, test_files = train_test_split(
                jpg_files, test_size=test_split, random_state=seed
            )
            
            # Second split: separate validation from training
            train_files, val_files = train_test_split(
                train_val_files, test_size=val_split/(1-test_split), random_state=seed
            )
            
            # Copy files with progress bar for each grid
            grid_num = grid.replace("grid-", "")
            
            # Copy train files
            for file in tqdm(train_files, desc=f"  Copying {grid_num} train", leave=False, unit="file"):
                shutil.copy2(os.path.join(grid_path, file), 
                           os.path.join(train_grid_dir, file))
            
            # Copy validation files
            for file in tqdm(val_files, desc=f"  Copying {grid_num} val", leave=False, unit="file"):
                shutil.copy2(os.path.join(grid_path, file), 
                           os.path.join(val_grid_dir, file))
            
            # Copy test files
            for file in tqdm(test_files, desc=f"  Copying {grid_num} test", leave=False, unit="file"):
                shutil.copy2(os.path.join(grid_path, file), 
                           os.path.join(test_grid_dir, file))
            
            total_train += len(train_files)
            total_val += len(val_files)
            total_test += len(test_files)
            
            tqdm.write(f"  {grid_num}: {len(train_files):4} train, {len(val_files):3} val, {len(test_files):3} test")
        
        print(f"\nTotal images: {total_train:,} train, {total_val:,} val, {total_test:,} test")
        print(f"Split ratio: {100*(1-test_split-val_split):.0f}%/{100*val_split:.0f}%/{100*test_split:.0f}%")
        
        # Save metadata with grid dimensions
        metadata = {
            'grid_dimensions': grid_dims,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'excluded_grids': self.excluded_grids,
            'min_images_threshold': self.min_images_threshold,
            'total_train': total_train,
            'total_val': total_val,
            'total_test': total_test,
            'class_weights': {str(k): float(v) for k, v in self.class_weights.items()},
            'grid_counts': {g: grid_counts[g] for g in self.class_names}
        }
        
        with open(os.path.join(self.base_dir, f'training_metadata_{grid_dims}.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return train_dir, val_dir, test_dir
    
    def create_model(self, model_type='efficient'):
        """
        Create the neural network model.
        
        Args:
            model_type: Type of model to create ('simple', 'vgg', 'efficient', 'resnet')
        """
        print(f"\nCreating {model_type} model for {self.num_classes} classes...")
        
        # Explicitly use GPU if available
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        print(f"Using device: {device}")
        
        with tf.device(device):
            if model_type == 'simple':
                # Simple CNN model with batch normalization
                self.model = models.Sequential([
                    layers.Rescaling(1./255, input_shape=(self.img_height, self.img_width, 3)),
                    
                    layers.Conv2D(32, 3, padding='same'),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.MaxPooling2D(),
                    
                    layers.Conv2D(64, 3, padding='same'),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.MaxPooling2D(),
                    
                    layers.Conv2D(128, 3, padding='same'),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.MaxPooling2D(),
                    
                    layers.Conv2D(256, 3, padding='same'),
                    layers.BatchNormalization(),
                    layers.Activation('relu'),
                    layers.MaxPooling2D(),
                    
                    layers.GlobalAveragePooling2D(),
                    layers.Dropout(0.5),
                    layers.Dense(512, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(self.num_classes, activation='softmax')
                ])
                
            elif model_type == 'efficient':
                # Use transfer learning with EfficientNetB0
                base_model = tf.keras.applications.EfficientNetB0(
                    input_shape=(self.img_height, self.img_width, 3),
                    include_top=False,
                    weights='imagenet'
                )
                base_model.trainable = False  # Freeze base model initially
                
                inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
                x = tf.keras.applications.efficientnet.preprocess_input(inputs)
                x = base_model(x, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.Dropout(0.3)(x)
                outputs = layers.Dense(self.num_classes, activation='softmax')(x)
                
                self.model = keras.Model(inputs, outputs)
                self.base_model = base_model  # Store for fine-tuning later
                
            elif model_type == 'resnet':
                # Use ResNet50 for potentially better accuracy
                base_model = tf.keras.applications.ResNet50(
                    input_shape=(self.img_height, self.img_width, 3),
                    include_top=False,
                    weights='imagenet'
                )
                base_model.trainable = False
                
                inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
                x = tf.keras.applications.resnet.preprocess_input(inputs)
                x = base_model(x, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.Dropout(0.3)(x)
                outputs = layers.Dense(self.num_classes, activation='softmax')(x)
                
                self.model = keras.Model(inputs, outputs)
                self.base_model = base_model
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')
            ]
        )
        
        print(f"Model created with {self.model.count_params():,} parameters")
        print(f"Model will run on: {device}")
        
        return self.model
    
    def train(self, train_dir, val_dir, epochs=30, use_augmentation=True, use_class_weights=True):
        """
        Train the model with class balancing.
        
        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            epochs: Number of epochs to train
            use_augmentation: Whether to use data augmentation
            use_class_weights: Whether to apply class weights for balancing
        """
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Detect grid dimensions
        grid_dims = self.detect_grid_dimensions()
        print(f"Training {grid_dims} grid model")
        
        # Create data generators
        if use_augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255 if not hasattr(self, 'base_model') else None,
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                horizontal_flip=True,
                zoom_range=0.15,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255 if not hasattr(self, 'base_model') else None
            )
        
        val_datagen = ImageDataGenerator(
            rescale=1./255 if not hasattr(self, 'base_model') else None
        )
        
        # Create data generators with progress
        print("\nPreparing data generators...")
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Print class distribution
        print("\nClass distribution in training:")
        for cls_idx, cls_name in train_generator.class_indices.items():
            count = np.sum(train_generator.classes == cls_name)
            weight = self.class_weights[cls_name] if use_class_weights else 1.0
            print(f"  {cls_idx}: {count:4} images, weight: {weight:.2f}")
        
        # Custom TQDM callback for better progress display
        class TqdmProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, epochs, steps_per_epoch, validation_steps=None):
                super().__init__()
                self.epochs = epochs
                self.steps_per_epoch = steps_per_epoch
                self.validation_steps = validation_steps
                self.epoch_bar = None
                self.batch_bar = None
                
            def on_train_begin(self, logs=None):
                self.epoch_bar = tqdm(total=self.epochs, desc='Epochs', unit='epoch', position=0)
                
            def on_epoch_begin(self, epoch, logs=None):
                self.batch_bar = tqdm(total=self.steps_per_epoch, 
                                    desc=f'Epoch {epoch+1}/{self.epochs}',
                                    unit='batch', 
                                    position=1,
                                    leave=False)
                
            def on_batch_end(self, batch, logs=None):
                self.batch_bar.update(1)
                if logs:
                    self.batch_bar.set_postfix({
                        'loss': f'{logs.get("loss", 0):.4f}',
                        'acc': f'{logs.get("accuracy", 0):.4f}'
                    })
                    
            def on_epoch_end(self, epoch, logs=None):
                self.batch_bar.close()
                self.epoch_bar.update(1)
                if logs:
                    self.epoch_bar.set_postfix({
                        'loss': f'{logs.get("loss", 0):.4f}',
                        'acc': f'{logs.get("accuracy", 0):.4f}',
                        'val_loss': f'{logs.get("val_loss", 0):.4f}',
                        'val_acc': f'{logs.get("val_accuracy", 0):.4f}'
                    })
                    
            def on_train_end(self, logs=None):
                self.epoch_bar.close()
        
        # Calculate steps
        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)
        
        # Callbacks with grid dimensions in filenames
        callbacks = [
            TqdmProgressCallback(epochs, steps_per_epoch, validation_steps),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.base_dir, f'best_model_{grid_dims}.keras'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=0  # Reduced verbosity since we have tqdm
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Prepare class weights for training
        training_class_weights = self.class_weights if use_class_weights else None
        
        print(f"\nTraining with{'out' if not use_class_weights else ''} class weights")
        print(f"Data augmentation: {'ON' if use_augmentation else 'OFF'}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        print("-" * 60)
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            class_weight=training_class_weights,
            verbose=0  # Disable default Keras progress since we're using tqdm
        )
        
        # Save the final model with grid dimensions
        print("\nSaving model...")
        self.model.save(os.path.join(self.base_dir, f'final_model_{grid_dims}.keras'))
        
        # Save training history with grid dimensions
        with open(os.path.join(self.base_dir, f'training_history_{grid_dims}.json'), 'w') as f:
            # Convert history to serializable format
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        print(f"✅ Training complete for {grid_dims} grid!")
        
        return history
    
    def fine_tune(self, train_dir, val_dir, epochs=10, unfreeze_layers=30):
        """
        Fine-tune the model (for transfer learning models).
        
        Args:
            train_dir: Directory containing training data
            val_dir: Directory containing validation data
            epochs: Number of epochs to fine-tune
            unfreeze_layers: Number of layers to unfreeze from the top
        """
        if not hasattr(self, 'base_model'):
            print("Fine-tuning is only available for transfer learning models")
            return None
        
        print("\n" + "="*60)
        print("FINE-TUNING MODEL")
        print("="*60)
        
        # Unfreeze the top layers of the base model
        self.base_model.trainable = True
        
        # Freeze all layers except the top `unfreeze_layers`
        for layer in self.base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_acc')
            ]
        )
        
        print(f"Fine-tuning top {unfreeze_layers} layers")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        # Train with same setup as before
        return self.train(train_dir, val_dir, epochs=epochs, use_augmentation=True, use_class_weights=True)
    
    def evaluate(self, test_dir, model_path=None):
        """
        Evaluate the model on test data and generate detailed metrics.
        
        Args:
            test_dir: Directory containing test data
            model_path: Path to saved model (if None, uses current model)
        """
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)
        
        # Load model if path provided
        if model_path:
            print(f"Loading model from {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            
            # Extract grid dimensions from model path if possible
            import re
            match = re.search(r'(\d+x\d+)', model_path)
            if match:
                self.grid_dims = match.group(1)
            
            # Load metadata
            metadata_path = os.path.join(self.base_dir, f'training_metadata_{self.grid_dims}.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.class_names = metadata['class_names']
                    self.num_classes = metadata['num_classes']
                    self.excluded_grids = metadata.get('excluded_grids', [])
        
        # Detect grid dimensions if not already set
        if self.grid_dims == "unknown":
            self.grid_dims = self.detect_grid_dimensions()
        
        # Create test data generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        # Evaluate overall metrics
        results = self.model.evaluate(test_generator, verbose=1)
        
        print("\n" + "="*40)
        print(f"TEST RESULTS ({self.grid_dims} Grid)")
        print("="*40)
        for name, value in zip(self.model.metrics_names, results):
            print(f"  {name:15}: {value:.4f} ({value*100:.2f}%)")
        
        # Generate predictions for detailed analysis
        print("\nGenerating detailed predictions...")
        
        # Create custom progress bar for predictions
        total_batches = len(test_generator)
        predictions = []
        
        with tqdm(total=total_batches, desc="Predicting", unit="batch") as pbar:
            for i in range(total_batches):
                batch_predictions = self.model.predict(
                    test_generator[i][0], 
                    verbose=0
                )
                predictions.append(batch_predictions)
                pbar.update(1)
        
        predictions = np.vstack(predictions)
        
        # Calculate per-class metrics
        y_true = test_generator.classes
        y_pred = np.argmax(predictions, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Classification report
        print("\n" + "="*40)
        print(f"PER-GRID CLASSIFICATION REPORT ({self.grid_dims})")
        print("="*40)
        
        # Create cleaner class names for report
        clean_names = [g.replace("grid-", "") for g in self.class_names]
        
        report = classification_report(y_true, y_pred, 
                                      target_names=clean_names, 
                                      output_dict=True)
        
        # Print formatted report
        print(f"{'Grid':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 46)
        
        for grid in clean_names:
            if grid in report:
                metrics = report[grid]
                print(f"{grid:<6} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} "
                      f"{metrics['f1-score']:<10.2f} {int(metrics['support']):<10}")
        
        print("-" * 46)
        print(f"{'Avg':<6} {report['weighted avg']['precision']:<10.2f} "
              f"{report['weighted avg']['recall']:<10.2f} "
              f"{report['weighted avg']['f1-score']:<10.2f} {int(report['weighted avg']['support']):<10}")
        
        # Save evaluation results with grid dimensions
        eval_results = {
            'grid_dimensions': self.grid_dims,
            'test_loss': float(results[0]),
            'test_accuracy': float(results[1]),
            'test_top3_accuracy': float(results[2]) if len(results) > 2 else None,
            'test_top5_accuracy': float(results[3]) if len(results) > 3 else None,
            'excluded_grids': self.excluded_grids,
            'timestamp': datetime.now().isoformat(),
            'classification_report': report
        }
        
        with open(os.path.join(self.base_dir, f'evaluation_results_{self.grid_dims}.json'), 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"\nResults saved to {self.base_dir}/evaluation_results_{self.grid_dims}.json")
        
        return results, predictions
    
    def predict_single_image(self, image_path, top_k=5, return_all=False, show_image=False):
        """
        Predict the grid probabilities for a single image.
        
        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return
            return_all: If True, returns full probability distribution
            show_image: If True, displays the image with predictions
            
        Returns:
            Dictionary mapping grid names to probabilities
        """
        # Load and preprocess image
        img = tf.keras.preprocessing.image.load_img(
            image_path, target_size=(self.img_height, self.img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch
        
        # Generate predictions
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Create probability distribution dictionary
        prob_dist = {}
        for i, class_name in enumerate(self.class_names):
            # Clean up grid name for output
            clean_name = class_name.replace("grid-", "")
            prob_dist[clean_name] = float(predictions[i])
        
        # Sort by probability
        sorted_probs = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)
        
        # Print top predictions
        print(f"\nPredictions for {os.path.basename(image_path)}:")
        print("-" * 40)
        
        cumulative_prob = 0
        for i, (grid, prob) in enumerate(sorted_probs[:top_k]):
            cumulative_prob += prob
            print(f"  {i+1}. Grid {grid:5}: {prob:6.2%} (cumulative: {cumulative_prob:6.2%})")
        
        # Optionally show the image with predictions
        if show_image:
            self._visualize_prediction(image_path, sorted_probs[:top_k])
        
        if return_all:
            return dict(sorted_probs)
        else:
            return dict(sorted_probs[:top_k])
    
    def _visualize_prediction(self, image_path, top_predictions):
        """Helper method to visualize image with predictions"""
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display the image
        img = plt.imread(image_path)
        ax1.imshow(img)
        ax1.set_title(f"Image: {os.path.basename(image_path)}", fontweight='bold')
        ax1.axis('off')
        
        # Display prediction bar chart
        grids = [g for g, _ in top_predictions]
        probs = [p for _, p in top_predictions]
        colors = ['green' if i == 0 else 'skyblue' for i in range(len(grids))]
        
        bars = ax2.barh(range(len(grids)), probs, color=colors)
        ax2.set_yticks(range(len(grids)))
        ax2.set_yticklabels([f"Grid {g}" for g in grids])
        ax2.set_xlabel('Probability')
        ax2.set_title('Top Predictions', fontweight='bold')
        ax2.set_xlim(0, 1)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, probs):
            width = bar.get_width()
            ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1%}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_test_predictions(self, test_dir, num_samples=6, random_selection=True):
        """
        Visualize predictions on multiple test images in a grid layout.
        
        Args:
            test_dir: Directory containing test data
            num_samples: Number of images to visualize
            random_selection: If True, randomly select images; if False, take first num_samples
        """
        import matplotlib.pyplot as plt
        
        print("\n" + "="*60)
        print("VISUALIZING TEST PREDICTIONS")
        print("="*60)
        
        # Collect test images
        test_images = []
        for grid_folder in os.listdir(test_dir):
            if grid_folder.startswith('grid-'):
                grid_path = os.path.join(test_dir, grid_folder)
                for img_file in os.listdir(grid_path):
                    if img_file.endswith('.jpg'):
                        test_images.append({
                            'path': os.path.join(grid_path, img_file),
                            'true_grid': grid_folder.replace('grid-', '')
                        })
        
        # Select images to visualize
        if random_selection:
            sample_images = random.sample(test_images, min(num_samples, len(test_images)))
        else:
            sample_images = test_images[:num_samples]
        
        # Create visualization grid
        cols = 3
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for idx, img_data in enumerate(sample_images):
            ax = axes[idx]
            
            # Load and display image
            img = plt.imread(img_data['path'])
            ax.imshow(img)
            
            # Get predictions
            predictions = self.predict_single_image(img_data['path'], top_k=3, show_image=False)
            
            # Format title with true label and top predictions
            top_pred = list(predictions.keys())[0]
            top_prob = list(predictions.values())[0]
            
            # Color code based on correctness
            title_color = 'green' if top_pred == img_data['true_grid'] else 'red'
            
            title = f"True: Grid {img_data['true_grid']}\n"
            title += f"Predicted: Grid {top_pred} ({top_prob:.1%})"
            
            ax.set_title(title, color=title_color, fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Model Predictions on Test Images', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Print summary
        correct = sum(1 for img_data in sample_images 
                     if list(self.predict_single_image(img_data['path'], top_k=1, show_image=False).keys())[0] == img_data['true_grid'])
        print(f"\nAccuracy on this sample: {correct}/{num_samples} ({100*correct/num_samples:.1f}%)")
    
    def analyze_misclassifications(self, test_dir, num_examples=5):
        """
        Find and visualize the most confident misclassifications.
        
        Args:
            test_dir: Directory containing test data
            num_examples: Number of misclassified examples to show
        """
        import matplotlib.pyplot as plt
        
        print("\n" + "="*60)
        print("ANALYZING MISCLASSIFICATIONS")
        print("="*60)
        
        misclassifications = []
        
        # Check predictions for all test images
        for grid_folder in os.listdir(test_dir):
            if grid_folder.startswith('grid-'):
                grid_path = os.path.join(test_dir, grid_folder)
                true_grid = grid_folder.replace('grid-', '')
                
                for img_file in os.listdir(grid_path):
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(grid_path, img_file)
                        
                        # Get prediction
                        predictions = self.predict_single_image(img_path, top_k=1, show_image=False)
                        pred_grid = list(predictions.keys())[0]
                        confidence = list(predictions.values())[0]
                        
                        # Store misclassifications
                        if pred_grid != true_grid:
                            misclassifications.append({
                                'path': img_path,
                                'true': true_grid,
                                'pred': pred_grid,
                                'confidence': confidence
                            })
        
        # Sort by confidence (most confident mistakes first)
        misclassifications.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"Found {len(misclassifications)} misclassifications")
        
        # Visualize top misclassifications
        if misclassifications:
            num_to_show = min(num_examples, len(misclassifications))
            
            fig, axes = plt.subplots(1, num_to_show, figsize=(4*num_to_show, 4))
            axes = axes if num_to_show > 1 else [axes]
            
            for idx, mis in enumerate(misclassifications[:num_to_show]):
                ax = axes[idx]
                img = plt.imread(mis['path'])
                ax.imshow(img)
                
                title = f"True: {mis['true']}\nPred: {mis['pred']} ({mis['confidence']:.1%})"
                ax.set_title(title, color='red', fontsize=10)
                ax.axis('off')
            
            plt.suptitle('Most Confident Misclassifications', fontsize=14, fontweight='bold', color='red')
            plt.tight_layout()
            plt.show()
            
            # Print details
            print("\nMost confident mistakes:")
            for i, mis in enumerate(misclassifications[:5], 1):
                print(f"{i}. True: Grid {mis['true']} → Predicted: Grid {mis['pred']} ({mis['confidence']:.1%})")
        else:
            print("No misclassifications found - perfect accuracy!")
    
    def plot_training_history(self, history):
        """
        Plot comprehensive training history.
        
        Args:
            history: Training history object
        """
        # Detect grid dimensions
        grid_dims = self.detect_grid_dimensions()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot Top-3 accuracy
        if 'top3_acc' in history.history:
            axes[1, 0].plot(history.history['top3_acc'], label='Train Top-3', linewidth=2)
            axes[1, 0].plot(history.history['val_top3_acc'], label='Val Top-3', linewidth=2)
            axes[1, 0].set_title('Top-3 Accuracy', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Top-3 Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot Top-5 accuracy
        if 'top5_acc' in history.history:
            axes[1, 1].plot(history.history['top5_acc'], label='Train Top-5', linewidth=2)
            axes[1, 1].plot(history.history['val_top5_acc'], label='Val Top-5', linewidth=2)
            axes[1, 1].set_title('Top-5 Accuracy', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Top-5 Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Training History - {grid_dims} Grid', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, f'training_history_{grid_dims}.png'), dpi=100, bbox_inches='tight')
        plt.show()
        
        # Print final metrics
        print(f"\nFinal Training Metrics ({grid_dims} Grid):")
        print("-" * 40)
        final_epoch = len(history.history['accuracy'])
        print(f"  Final Train Accuracy:      {history.history['accuracy'][-1]:.4f}")
        print(f"  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        if 'top3_acc' in history.history:
            print(f"  Final Val Top-3 Accuracy:  {history.history['val_top3_acc'][-1]:.4f}")
        if 'top5_acc' in history.history:
            print(f"  Final Val Top-5 Accuracy:  {history.history['val_top5_acc'][-1]:.4f}")

    def render_recover_options():
        print("\n" + "="*70)
        print("RECOVERING TRAINED MODEL")
        print("="*70)
        
        # Try to find models with different grid dimensions
        model_files = glob.glob(os.path.join(base_dir, 'best_model_*.keras'))
        
        if model_files:
            if len(model_files) > 1:
                print("Found multiple trained models:")
                for i, model_file in enumerate(model_files):
                    grid_dim = os.path.basename(model_file).replace('best_model_', '').replace('.keras', '')
                    
                    # Load history to get accuracy
                    history_path = os.path.join(base_dir, f'training_history_{grid_dim}.json')
                    
                    acc_info = ""
                    if os.path.exists(history_path):
                        with open(history_path, 'r') as f:
                            hist = json.load(f)
                            if 'val_accuracy' in hist:
                                final_acc = hist['val_accuracy'][-1]
                                acc_info = f" (Acc: {final_acc:.2%})"
                    
                    print(f"{i+1}. {grid_dim} grid{acc_info}")
                
                choice = input(f"\nSelect model to load (1-{len(model_files)}): ").strip()
                try:
                    model_idx = int(choice) - 1
                    best_model_path = model_files[model_idx]
                except:
                    best_model_path = model_files[0]  # Default to first
            else:
                best_model_path = model_files[0]
            
            # Extract grid dimensions from filename
            grid_dims = os.path.basename(best_model_path).replace('best_model_', '').replace('.keras', '')
            
            print(f"\n✅ Loading {grid_dims} grid model")
            ai.model = tf.keras.models.load_model(best_model_path)
            ai.grid_dims = grid_dims
            
            # Load corresponding metadata and history
            metadata_path = os.path.join(base_dir, f'training_metadata_{grid_dims}.json')
            history_path = os.path.join(base_dir, f'training_history_{grid_dims}.json')
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    ai.class_names = metadata['class_names']
                    ai.num_classes = metadata['num_classes']
                    ai.excluded_grids = metadata.get('excluded_grids', [])
                    ai.class_weights = {int(k): v for k, v in metadata.get('class_weights', {}).items()}
                    print(f"✅ Loaded metadata: {ai.num_classes} classes in {grid_dims} configuration")
            
            # Load and display training history
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history_data = json.load(f)
                    final_accuracy = history_data['val_accuracy'][-1] if 'val_accuracy' in history_data else None
                    final_loss = history_data['val_loss'][-1] if 'val_loss' in history_data else None
                    epochs_trained = len(history_data.get('accuracy', []))
                    
                    print(f"\n📊 Training History:")
                    print(f"   Grid configuration: {grid_dims}")
                    print(f"   Epochs trained: {epochs_trained}")
                    if final_accuracy:
                        print(f"   Final validation accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
                    if final_loss:
                        print(f"   Final validation loss: {final_loss:.4f}")
                    
                    # Check for top-k accuracies
                    if 'val_top3_acc' in history_data:
                        print(f"   Final top-3 accuracy: {history_data['val_top3_acc'][-1]:.4f}")
                    if 'val_top5_acc' in history_data:
                        print(f"   Final top-5 accuracy: {history_data['val_top5_acc'][-1]:.4f}")
            
            print("\n✅ Model recovered successfully!")
            
            # INTERACTIVE MENU
            while True:
                print("\n" + "="*70)
                print("WHAT WOULD YOU LIKE TO DO?")
                print("="*70)
                print("1. Evaluate on test set")
                print("2. Visualize sample predictions")
                print("3. Analyze misclassifications")
                print("4. Single image comprehensive analysis")
                print("5. Predict on custom image")
                print("6. Show all visualizations")
                print("0. Exit")
                
                choice = input("\nEnter your choice (0-6): ")
                
                test_dir = os.path.join(base_dir, "test")
                
                if choice == "1":
                    # Evaluate on test set
                    if os.path.exists(test_dir):
                        print("\n" + "="*70)
                        print("EVALUATING MODEL")
                        print("="*70)
                        results, predictions = ai.evaluate(test_dir)
                    else:
                        print("❌ Test directory not found!")
                        
                elif choice == "2":
                    # Visualize predictions
                    if os.path.exists(test_dir):
                        num = input("How many samples to visualize? (default 6): ")
                        num_samples = int(num) if num else 6
                        ai.visualize_test_predictions(test_dir, num_samples=num_samples, random_selection=True)
                    else:
                        print("❌ Test directory not found!")
                        
                elif choice == "3":
                    # Analyze misclassifications
                    if os.path.exists(test_dir):
                        num = input("How many misclassifications to show? (default 5): ")
                        num_examples = int(num) if num else 5
                        ai.analyze_misclassifications(test_dir, num_examples=num_examples)
                    else:
                        print("❌ Test directory not found!")
                        
                elif choice == "4":
                    # Single image comprehensive analysis
                    if os.path.exists(test_dir):
                        use_random = input("Use random image? (y/n, default y): ").lower()
                        
                        sample_image = None
                        true_grid = None
                        
                        if use_random != 'n':
                            # Find random test image
                            test_images = []
                            for grid_folder in os.listdir(test_dir):
                                if grid_folder.startswith('grid-'):
                                    grid_path = os.path.join(test_dir, grid_folder)
                                    for img_file in os.listdir(grid_path):
                                        if img_file.endswith('.jpg'):
                                            test_images.append((os.path.join(grid_path, img_file), 
                                                              grid_folder.replace('grid-', '')))
                            
                            if test_images:
                                sample_image, true_grid = random.choice(test_images)
                        else:
                            path = input("Enter image path: ").strip()
                            if os.path.exists(path):
                                sample_image = path
                                # Try to extract true grid from path
                                for grid_folder in os.listdir(test_dir):
                                    if grid_folder in path:
                                        true_grid = grid_folder.replace('grid-', '')
                                        break
                                if not true_grid:
                                    true_grid = "unknown"
                        
                        if sample_image and os.path.exists(sample_image):
                            print(f"\nAnalyzing: {os.path.basename(sample_image)}")
                            print(f"True grid: {true_grid}")
                            
                            # Get predictions
                            predictions = ai.predict_single_image(sample_image, return_all=True, show_image=False)
                            
                            # Create figure with multiple subplots
                            fig = plt.figure(figsize=(20, 12))
                            
                            # 1. Original Image (top left)
                            ax1 = plt.subplot(2, 3, 1)
                            img = plt.imread(sample_image)
                            ax1.imshow(img)
                            ax1.set_title(f"Input Image\n{os.path.basename(sample_image)}", fontsize=12, fontweight='bold')
                            ax1.axis('off')
                            
                            # 2. Top-5 Predictions Bar Chart (top middle)
                            ax2 = plt.subplot(2, 3, 2)
                            top_5 = list(predictions.items())[:5]
                            grids = [g for g, _ in top_5]
                            probs = [p for _, p in top_5]
                            colors = ['green' if grids[0] == true_grid else 'red'] + ['skyblue']*4
                            
                            bars = ax2.barh(range(len(grids)), probs, color=colors)
                            ax2.set_yticks(range(len(grids)))
                            ax2.set_yticklabels([f"Grid {g}" for g in grids])
                            ax2.set_xlabel('Probability')
                            ax2.set_title('Top 5 Predictions', fontsize=12, fontweight='bold')
                            ax2.set_xlim(0, max(probs)*1.2 if probs else 1)
                            
                            for bar, prob in zip(bars, probs):
                                ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                        f'{prob:.1%}', ha='left', va='center')
                            
                            # 3. Probability Distribution Histogram (top right)
                            ax3 = plt.subplot(2, 3, 3)
                            all_probs = sorted(predictions.values(), reverse=True)
                            ax3.bar(range(len(all_probs)), all_probs, color='steelblue', alpha=0.7)
                            ax3.set_xlabel('Grid Rank')
                            ax3.set_ylabel('Probability')
                            ax3.set_title('Probability Distribution Across All Grids', fontsize=12, fontweight='bold')
                            ax3.axhline(y=1.0/len(predictions), color='red', linestyle='--', 
                                       label=f'Random Chance ({1.0/len(predictions):.1%})')
                            ax3.legend()
                            
                            # 4. Geographic Heat Map on NA_map.png (bottom, spanning all columns)
                            ax4 = plt.subplot(2, 1, 2)
                            
                            # Load the base map
                            map_path = os.path.join(base_dir, "..", "NA_map.png")
                            if os.path.exists(map_path):
                                base_map = plt.imread(map_path)
                                ax4.imshow(base_map)
                                ax4.set_title('Geographic Probability Heat Map', fontsize=14, fontweight='bold')
                                ax4.axis('off')
                                
                                # Calculate grid dimensions
                                map_height, map_width = base_map.shape[:2]
                                max_row = max(int(name.split('-')[1]) for name in ai.class_names)
                                max_col = max(int(name.split('-')[2]) for name in ai.class_names)
                                rows = max_row + 1
                                cols = max_col + 1
                                
                                cell_height = map_height / rows
                                cell_width = map_width / cols
                                
                                # Create probability overlay
                                for class_name in ai.class_names:
                                    parts = class_name.split('-')
                                    row = int(parts[1])
                                    col = int(parts[2])
                                    
                                    grid_key = class_name.replace("grid-", "")
                                    prob = predictions.get(grid_key, 0)
                                    
                                    x = col * cell_width
                                    y = row * cell_height
                                    
                                    # Color intensity based on probability
                                    rect = Rectangle((x, y), cell_width, cell_height,
                                                   linewidth=1, edgecolor='black',
                                                   facecolor='red', alpha=min(prob * 2, 0.9))
                                    ax4.add_patch(rect)
                                    
                                    # Add text for significant probabilities
                                    if prob > 0.05:
                                        cx = x + cell_width / 2
                                        cy = y + cell_height / 2
                                        ax4.text(cx, cy, f'{prob:.0%}', 
                                                color='white', fontsize=10, fontweight='bold',
                                                ha='center', va='center',
                                                path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
                                    
                                    # Highlight true grid with green border
                                    if grid_key == true_grid:
                                        rect_true = Rectangle((x, y), cell_width, cell_height,
                                                             linewidth=3, edgecolor='lime',
                                                             facecolor='none')
                                        ax4.add_patch(rect_true)
                                
                                # Add legend
                                legend_elements = [
                                    mpatches.Patch(color='red', alpha=0.8, label='High Probability'),
                                    mpatches.Patch(color='red', alpha=0.4, label='Medium Probability'),
                                    mpatches.Patch(color='red', alpha=0.1, label='Low Probability'),
                                    mpatches.Rectangle((0, 0), 1, 1, linewidth=3, 
                                                      edgecolor='lime', facecolor='none', 
                                                      label='True Location')
                                ]
                                ax4.legend(handles=legend_elements, loc='lower right', fontsize=10)
                            else:
                                ax4.text(0.5, 0.5, f'NA_map.png not found at {map_path}', 
                                        ha='center', va='center', transform=ax4.transAxes)
                                ax4.axis('off')
                            
                            # Add summary statistics
                            top_pred = list(predictions.keys())[0]
                            top_prob = list(predictions.values())[0]
                            correct = "✅ CORRECT" if top_pred == true_grid else "❌ INCORRECT"
                            
                            # Calculate confidence metrics
                            entropy = -sum(p * np.log(p + 1e-10) for p in predictions.values())
                            max_entropy = np.log(len(predictions))
                            confidence = 1 - (entropy / max_entropy)
                            
                            fig.suptitle(f'Comprehensive Analysis | Prediction: Grid {top_pred} ({top_prob:.1%}) | {correct}\n'
                                        f'Model Confidence: {confidence:.1%} | Entropy: {entropy:.2f}', 
                                        fontsize=16, fontweight='bold')
                            
                            plt.tight_layout()
                            plt.show()
                            
                            # Print detailed statistics
                            print("\n" + "="*40)
                            print("PREDICTION DETAILS")
                            print("="*40)
                            print(f"True Grid:        {true_grid}")
                            print(f"Predicted Grid:   {top_pred}")
                            print(f"Confidence:       {top_prob:.2%}")
                            print(f"Result:           {correct}")
                            print(f"Model Certainty:  {confidence:.2%}")
                            print(f"Top-3 Accuracy:   {'Yes' if true_grid in [g for g, _ in list(predictions.items())[:3]] else 'No'}")
                            print(f"Top-5 Accuracy:   {'Yes' if true_grid in [g for g, _ in list(predictions.items())[:5]] else 'No'}")
                            
                        else:
                            print("❌ Image not found!")
                    else:
                        print("❌ Test directory not found!")
                        
                elif choice == "5":
                    # Predict on custom image
                    path = input("Enter image path: ").strip()
                    if os.path.exists(path):
                        # top_k = input("How many top predictions to show? (default 5): ").strip()
                        # top_k = int(top_k) if top_k else 5
                        # ai.predict_single_image(path, top_k=top_k, show_image=True)
                        # Get predictions
                        predictions = ai.predict_single_image(path, return_all=True, show_image=False)
                        sample_image = path
                        true_grid = "unknown"
                        
                        # Create figure with multiple subplots
                        fig = plt.figure(figsize=(20, 12))
                        
                        # 1. Original Image (top left)
                        ax1 = plt.subplot(2, 3, 1)
                        img = plt.imread(sample_image)
                        ax1.imshow(img)
                        ax1.set_title(f"Input Image\n{os.path.basename(sample_image)}", fontsize=12, fontweight='bold')
                        ax1.axis('off')
                        
                        # 2. Top-5 Predictions Bar Chart (top middle)
                        ax2 = plt.subplot(2, 3, 2)
                        top_5 = list(predictions.items())[:5]
                        grids = [g for g, _ in top_5]
                        probs = [p for _, p in top_5]
                        colors = ['green' if grids[0] == true_grid else 'red'] + ['skyblue']*4
                        
                        bars = ax2.barh(range(len(grids)), probs, color=colors)
                        ax2.set_yticks(range(len(grids)))
                        ax2.set_yticklabels([f"Grid {g}" for g in grids])
                        ax2.set_xlabel('Probability')
                        ax2.set_title('Top 5 Predictions', fontsize=12, fontweight='bold')
                        ax2.set_xlim(0, max(probs)*1.2 if probs else 1)
                        
                        for bar, prob in zip(bars, probs):
                            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                    f'{prob:.1%}', ha='left', va='center')
                        
                        # 3. Probability Distribution Histogram (top right)
                        ax3 = plt.subplot(2, 3, 3)
                        all_probs = sorted(predictions.values(), reverse=True)
                        ax3.bar(range(len(all_probs)), all_probs, color='steelblue', alpha=0.7)
                        ax3.set_xlabel('Grid Rank')
                        ax3.set_ylabel('Probability')
                        ax3.set_title('Probability Distribution Across All Grids', fontsize=12, fontweight='bold')
                        ax3.axhline(y=1.0/len(predictions), color='red', linestyle='--', 
                                    label=f'Random Chance ({1.0/len(predictions):.1%})')
                        ax3.legend()
                        
                        # 4. Geographic Heat Map on NA_map.png (bottom, spanning all columns)
                        ax4 = plt.subplot(2, 1, 2)
                        
                        # Load the base map
                        map_path = os.path.join(base_dir, "..", "NA_map.png")
                        if os.path.exists(map_path):
                            base_map = plt.imread(map_path)
                            ax4.imshow(base_map)
                            ax4.set_title('Geographic Probability Heat Map', fontsize=14, fontweight='bold')
                            ax4.axis('off')
                            
                            # Calculate grid dimensions
                            map_height, map_width = base_map.shape[:2]
                            max_row = max(int(name.split('-')[1]) for name in ai.class_names)
                            max_col = max(int(name.split('-')[2]) for name in ai.class_names)
                            rows = max_row + 1
                            cols = max_col + 1
                            
                            cell_height = map_height / rows
                            cell_width = map_width / cols
                            
                            # Create probability overlay
                            for class_name in ai.class_names:
                                parts = class_name.split('-')
                                row = int(parts[1])
                                col = int(parts[2])
                                
                                grid_key = class_name.replace("grid-", "")
                                prob = predictions.get(grid_key, 0)
                                
                                x = col * cell_width
                                y = row * cell_height
                                
                                # Color intensity based on probability
                                rect = Rectangle((x, y), cell_width, cell_height,
                                                linewidth=1, edgecolor='black',
                                                facecolor='red', alpha=min(prob * 2, 0.9))
                                ax4.add_patch(rect)
                                
                                # Add text for significant probabilities
                                if prob > 0.05:
                                    cx = x + cell_width / 2
                                    cy = y + cell_height / 2
                                    ax4.text(cx, cy, f'{prob:.0%}', 
                                            color='white', fontsize=10, fontweight='bold',
                                            ha='center', va='center',
                                            path_effects=[patheffects.withStroke(linewidth=2, foreground='black')])
                                
                                # Highlight true grid with green border
                                if grid_key == true_grid:
                                    rect_true = Rectangle((x, y), cell_width, cell_height,
                                                            linewidth=3, edgecolor='lime',
                                                            facecolor='none')
                                    ax4.add_patch(rect_true)
                            
                            # Add legend
                            legend_elements = [
                                mpatches.Patch(color='red', alpha=0.8, label='High Probability'),
                                mpatches.Patch(color='red', alpha=0.4, label='Medium Probability'),
                                mpatches.Patch(color='red', alpha=0.1, label='Low Probability'),
                                mpatches.Rectangle((0, 0), 1, 1, linewidth=3, 
                                                    edgecolor='lime', facecolor='none', 
                                                    label='True Location')
                            ]
                            ax4.legend(handles=legend_elements, loc='lower right', fontsize=10)
                        else:
                            ax4.text(0.5, 0.5, f'NA_map.png not found at {map_path}', 
                                    ha='center', va='center', transform=ax4.transAxes)
                            ax4.axis('off')
                        
                        # Add summary statistics
                        top_pred = list(predictions.keys())[0]
                        top_prob = list(predictions.values())[0]
                        correct = "✅ CORRECT" if top_pred == true_grid else "❌ INCORRECT"
                        
                        # Calculate confidence metrics
                        entropy = -sum(p * np.log(p + 1e-10) for p in predictions.values())
                        max_entropy = np.log(len(predictions))
                        confidence = 1 - (entropy / max_entropy)
                        
                        fig.suptitle(f'Comprehensive Analysis | Prediction: Grid {top_pred} ({top_prob:.1%}) | {correct}\n'
                                    f'Model Confidence: {confidence:.1%} | Entropy: {entropy:.2f}', 
                                    fontsize=16, fontweight='bold')
                        
                        plt.tight_layout()
                        plt.show()
                        
                        # Print detailed statistics
                        print("\n" + "="*40)
                        print("PREDICTION DETAILS")
                        print("="*40)
                        print(f"True Grid:        {true_grid}")
                        print(f"Predicted Grid:   {top_pred}")
                        print(f"Confidence:       {top_prob:.2%}")
                        print(f"Result:           {correct}")
                        print(f"Model Certainty:  {confidence:.2%}")
                        print(f"Top-3 Accuracy:   {'Yes' if true_grid in [g for g, _ in list(predictions.items())[:3]] else 'No'}")
                        print(f"Top-5 Accuracy:   {'Yes' if true_grid in [g for g, _ in list(predictions.items())[:5]] else 'No'}")
                        
                    else:
                        print("❌ Test directory not found!")
                        
                elif choice == "6":
                    # Show all visualizations
                    if os.path.exists(test_dir):
                        print("\nRunning all visualizations...")
                        results, predictions = ai.evaluate(test_dir)
                        ai.visualize_test_predictions(test_dir, num_samples=6)
                        ai.analyze_misclassifications(test_dir, num_examples=5)
                    else:
                        print("❌ Test directory not found!")
                        
                elif choice == "0":
                    print("\nExiting...")
                    break
                    
                else:
                    print("Invalid choice! Please try again.")
                
        else:
            print(f"❌ No trained models found in {base_dir}")
            print("Switching to training mode...")
            MODE = "train"
    
    def render_train_options():
        # Step 1: Prepare train/validation/test split
        print("\n" + "="*70)
        print("STEP 1: DATA PREPARATION")
        print("="*70)
        train_dir, val_dir, test_dir = ai.prepare_train_test_split(
            test_split=0.2,  # 20% for testing
            val_split=0.1    # 10% for validation
        )
        
        # Step 2: Create and train model
        print("\n" + "="*70)
        print("STEP 2: MODEL CREATION AND TRAINING")
        print("="*70)
        
        # Choose model type: 'simple', 'efficient', or 'resnet'
        ai.create_model(model_type='efficient')
        
        # Train the model with class weighting
        history = ai.train(
            train_dir, 
            val_dir, 
            epochs=25,  # Adjust based on your needs
            use_augmentation=True,
            use_class_weights=True  # Important for imbalanced grids
        )
        
        # Optional: Fine-tune if using transfer learning
        if hasattr(ai, 'base_model'):
            print("\n" + "="*70)
            print("STEP 3: FINE-TUNING")
            print("="*70)
            history_ft = ai.fine_tune(
                train_dir, 
                val_dir, 
                epochs=10,
                unfreeze_layers=30
            )
        
        # Step 4: Evaluate the model
        print("\n" + "="*70)
        print("STEP 4: MODEL EVALUATION")
        print("="*70)
        results, predictions = ai.evaluate(test_dir)
        
        # Plot training history
        ai.plot_training_history(history)
        
        # Visualize predictions on test set
        ai.visualize_test_predictions(test_dir, num_samples=6, random_selection=True)
        
        # Analyze misclassifications
        ai.analyze_misclassifications(test_dir, num_examples=5)

    def render_evaluate_options():
        print("\n" + "="*70)
        print("EVALUATION MODE")
        print("="*70)
        
        # Find available models
        model_files = glob.glob(os.path.join(base_dir, 'best_model_*.keras'))
        
        if model_files:
            if len(model_files) > 1:
                print("Found multiple models to evaluate:")
                for i, model_file in enumerate(model_files):
                    grid_dim = os.path.basename(model_file).replace('best_model_', '').replace('.keras', '')
                    print(f"{i+1}. {grid_dim} grid")
                
                choice = input(f"Select model to evaluate (1-{len(model_files)}): ").strip()
                try:
                    model_idx = int(choice) - 1
                    model_path = model_files[model_idx]
                except:
                    model_path = model_files[0]
            else:
                model_path = model_files[0]
            
            test_dir = os.path.join(base_dir, "test")
            
            if os.path.exists(test_dir):
                results, predictions = ai.evaluate(test_dir, model_path=model_path)
                
                # Visualizations
                ai.visualize_test_predictions(test_dir, num_samples=9, random_selection=True)
                ai.analyze_misclassifications(test_dir, num_examples=5)
            else:
                print("❌ Test directory not found!")
        else:
            print("❌ No models found to evaluate!")

# Main execution
if __name__ == "__main__":

    base_dir = "../../dataset/united_states"
    
    # ========================================
    # CONFIGURATION
    # ========================================

    print("========== WELCOME ==========")
    print("To get started, choose an action for the model")
    print("1. Recover a model to test")
    print("2. Train a new model")
    print("3. Evaluate an existing model")
    mode_input = input("Select an (default 1): ")
    if mode_input == "1":
        MODE = "recover"
    elif mode_input == "2":
        MODE = "train"
    elif mode_input == "3":
        MODE = "evaluate"
    else:
        MODE = "recover"

    print()
    print()

    threshold = 600

    if MODE == "train":
        threshold = input("Enter a minimum number of training images to qualify a grid for training (default 600): ")
        if threshold != "":
            threshold = int(threshold)
   
    print()
    
    # Initialize the AI system with minimum threshold
    ai = GeoGuessrAI(
        base_dir=base_dir,
        min_images_threshold=threshold
    )
    
    # ========================================
    # MODE: RECOVER - Load existing model
    # ========================================
    if MODE == "recover":
        GeoGuessrAI.render_recover_options()
    
    # ========================================
    # MODE: TRAIN - Train new model
    # ========================================
    if MODE == "train":
        GeoGuessrAI.render_train_options()
    
    # ========================================
    # MODE: EVALUATE - Just evaluate existing model
    # ========================================
    elif MODE == "evaluate":
        GeoGuessrAI.render_evaluate_options()
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "="*70)
    print("EXECUTION COMPLETE!")
    print("="*70)
        
    if MODE == "train":
        print(f"\nModels and results saved in {ai.base_dir}:")
        print(f"  📁 best_model_{ai.grid_dims}.keras       - Best model during training")
        print(f"  📁 final_model_{ai.grid_dims}.keras      - Final model after all epochs")
        print(f"  📄 training_metadata_{ai.grid_dims}.json - Grid information and settings")
        print(f"  📄 training_history_{ai.grid_dims}.json  - Training metrics per epoch")
        print(f"  📄 evaluation_results_{ai.grid_dims}.json - Test evaluation metrics")
        print(f"  📊 training_history_{ai.grid_dims}.png   - Training curves visualization")
        
        if ai.excluded_grids:
            print(f"\n⚠️  Excluded grids (< {ai.min_images_threshold} images):")
            for grid in ai.excluded_grids:
                print(f"    - {grid}")