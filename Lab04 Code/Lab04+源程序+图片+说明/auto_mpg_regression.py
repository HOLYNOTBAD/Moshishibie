"""
Project 2: Predicting Auto MPG with PyTorch (Regression Problem)
Corresponding to Chapter 13: Going Deeper – The Mechanics of PyTorch
Main techniques: Custom dataset, feature engineering, DNN regression, autograd, PyTorch mechanics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. Data Preparation ====================
print("=" * 60)
print("Project 2: Auto MPG Prediction (Regression)")
print("Corresponding to Chapter 13: Predicting fuel efficiency")
print("=" * 60)

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin', 'Car Name']

# Read data
try:
    df = pd.read_csv(url, sep='\s+', header=None, names=column_names, na_values='?')
    print("✓ Successfully loaded dataset from web")
except Exception as e:
    print(f"⚠ Web loading failed: {e}")
    print("Please check internet connection or use local file")
    # Fallback: create dummy data for demonstration
    print("Creating synthetic data for demonstration...")
    np.random.seed(42)
    n_samples = 200
    df = pd.DataFrame({
        'MPG': np.random.uniform(10, 40, n_samples),
        'Cylinders': np.random.choice([4, 6, 8], n_samples),
        'Displacement': np.random.uniform(100, 400, n_samples),
        'Horsepower': np.random.uniform(80, 200, n_samples),
        'Weight': np.random.uniform(2000, 5000, n_samples),
        'Acceleration': np.random.uniform(8, 20, n_samples),
        'Model Year': np.random.randint(70, 85, n_samples),
        'Origin': np.random.choice([1, 2, 3], n_samples),
        'Car Name': [f'Car_{i}' for i in range(n_samples)]
    })

print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows of data:")
print(df.head())

# ==================== 2. Data Preprocessing ====================
print("\n" + "-" * 60)
print("Step 1: Data Cleaning and Preprocessing")
print("-" * 60)

# Handle missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Remove rows with missing values (simple approach as in textbook)
df_clean = df.dropna().reset_index(drop=True)
print(f"\nCleaned dataset shape: {df_clean.shape}")

# Convert Origin to categorical feature
df_clean['Origin'] = df_clean['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

# Create dummy variables (one-hot encoding)
df_clean = pd.get_dummies(df_clean, columns=['Origin'], prefix='', prefix_sep='')

# Remove car name column (not useful for prediction)
df_clean = df_clean.drop('Car Name', axis=1)

print(f"\nFeatures after preprocessing: {df_clean.columns.tolist()}")
print(f"Final dataset shape: {df_clean.shape}")

# ==================== 3. Separate Features and Target ====================
print("\n" + "-" * 60)
print("Step 2: Prepare Features and Target Variable")
print("-" * 60)

# Target variable: MPG (Miles Per Gallon)
target = 'MPG'
y = df_clean[target].values
X = df_clean.drop(target, axis=1).values

print(f"Feature matrix shape: {X.shape}")
print(f"Target variable shape: {y.shape}")
print(f"MPG range: [{y.min():.1f}, {y.max():.1f}], Mean: {y.mean():.2f}")

# ==================== 4. Feature Standardization ====================
print("\n" + "-" * 60)
print("Step 3: Feature Standardization")
print("-" * 60)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# ==================== 5. Split Dataset ====================
print("\n" + "-" * 60)
print("Step 4: Split into Train, Validation, and Test Sets")
print("-" * 60)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y_scaled)

# Create dataset
full_dataset = TensorDataset(X_tensor, y_tensor)

# Split ratios: 70% train, 15% validation, 15% test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== 6. Build Deep Neural Network ====================
print("\n" + "-" * 60)
print("Step 5: Build Deep Neural Network (DNN) Regression Model")
print("Corresponding to textbook: 'Training a DNN regression model'")
print("-" * 60)

class DNNRegressor(nn.Module):
    """
    Deep Neural Network Regressor
    Architecture: Input -> Hidden1 -> Hidden2 -> Hidden3 -> Output
    Uses ReLU activation and Dropout regularization
    """
    def __init__(self, input_dim):
        super(DNNRegressor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
        # Initialize weights (Kaiming initialization for ReLU)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# Initialize model
input_dim = X.shape[1]
model = DNNRegressor(input_dim)
print(model)
print(f"Input dimension: {input_dim}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# ==================== 7. Training Configuration ====================
print("\n" + "-" * 60)
print("Step 6: Configure Training Parameters")
print("-" * 60)

criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

# ==================== 8. Training Loop ====================
print("\n" + "-" * 60)
print("Step 7: Train the Model")
print("=" * 60)

num_epochs = 100
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience_counter = 0
patience = 20  # Early stopping patience

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_train_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X).squeeze()
        loss = criterion(predictions, batch_y)
        loss.backward()
        
        # Gradient clipping (prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_train_loss += loss.item() * batch_X.size(0)
    
    train_loss = epoch_train_loss / len(train_dataset)
    train_losses.append(train_loss)
    
    # Validation phase
    model.eval()
    epoch_val_loss = 0
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            epoch_val_loss += loss.item() * batch_X.size(0)
    
    val_loss = epoch_val_loss / len(val_dataset)
    val_losses.append(val_loss)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_dnn_model.pth')
    else:
        patience_counter += 1
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1:3d}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Early stopping
    if patience_counter >= patience:
        print(f'\n⚠ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
        break

print("✓ Training completed!")

# ==================== 9. Visualize Training Results ====================
print("\n" + "-" * 60)
print("Step 8: Visualize Training Results")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Training and Validation Loss
axes[0, 0].plot(train_losses, label='Training Loss', color='blue', alpha=0.8)
axes[0, 0].plot(val_losses, label='Validation Loss', color='red', alpha=0.8)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Load best model
model.load_state_dict(torch.load('best_dnn_model.pth'))
model.eval()

# 3. Predictions Visualization
with torch.no_grad():
    # Make predictions on test set
    test_predictions = []
    test_targets = []
    
    for batch_X, batch_y in test_loader:
        preds = model(batch_X).squeeze()
        test_predictions.extend(preds.numpy())
        test_targets.extend(batch_y.numpy())
    
    # Inverse transform to original scale
    test_predictions_orig = y_scaler.inverse_transform(
        np.array(test_predictions).reshape(-1, 1)
    ).flatten()
    
    test_targets_orig = y_scaler.inverse_transform(
        np.array(test_targets).reshape(-1, 1)
    ).flatten()

# Predictions vs Actual Values scatter plot
axes[0, 1].scatter(test_targets_orig, test_predictions_orig, alpha=0.6, edgecolors='k')
max_val = max(test_targets_orig.max(), test_predictions_orig.max())
min_val = min(test_targets_orig.min(), test_predictions_orig.min())
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual MPG')
axes[0, 1].set_ylabel('Predicted MPG')
axes[0, 1].set_title('Predictions vs Actual Values')
axes[0, 1].grid(True, alpha=0.3)

# Calculate performance metrics
mae = np.mean(np.abs(test_predictions_orig - test_targets_orig))
rmse = np.sqrt(np.mean((test_predictions_orig - test_targets_orig) ** 2))
r2 = 1 - np.sum((test_targets_orig - test_predictions_orig) ** 2) / np.sum((test_targets_orig - np.mean(test_targets_orig)) ** 2)

# 4. Error Distribution Histogram
errors = test_predictions_orig - test_targets_orig
axes[1, 0].hist(errors, bins=20, edgecolor='black', alpha=0.7)
axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Prediction Error (MPG)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title(f'Error Distribution (Mean: {errors.mean():.2f})')
axes[1, 0].grid(True, alpha=0.3)

# 5. Display Performance Metrics
axes[1, 1].axis('off')
metrics_text = (
    f'Model Performance Metrics:\n\n'
    f'• MAE: {mae:.2f} MPG\n'
    f'• RMSE: {rmse:.2f} MPG\n'
    f'• R² Score: {r2:.3f}\n'
    f'• Best Val Loss: {best_val_loss:.4f}\n'
    f'• Test Set Size: {len(test_dataset)}\n'
    f'• Number of Features: {input_dim}'
)
axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Auto MPG Prediction - DNN Regression Results', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('auto_mpg_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== 10. Feature Importance Analysis ====================
print("\n" + "-" * 60)
print("Step 9: Feature Importance Analysis")
print("-" * 60)

# Estimate feature importance using gradient-based method
model.eval()
feature_names = df_clean.drop('MPG', axis=1).columns.tolist()

# Calculate average absolute gradient for each feature
X_tensor.requires_grad_(True)
predictions = model(X_tensor)
predictions.sum().backward()

feature_importance = torch.abs(X_tensor.grad).mean(dim=0).numpy()

# Sort feature importance
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("Feature Importance Ranking:")
print(importance_df.to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(importance_df)), importance_df['Importance'][::-1])
plt.yticks(range(len(importance_df)), importance_df['Feature'][::-1])
plt.xlabel('Feature Importance (Mean Absolute Gradient)')
plt.title('Feature Importance Analysis')
plt.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, importance_df['Importance'][::-1])):
    plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# ==================== 11. Prediction Examples ====================
print("\n" + "-" * 60)
print("Step 10: New Sample Prediction Examples")
print("-" * 60)

# Create new car samples (based on dataset statistics)
new_car_features = {
    'Cylinders': 6,
    'Displacement': 250.0,
    'Horsepower': 100.0,
    'Weight': 3500.0,
    'Acceleration': 15.0,
    'Model Year': 80,
    'USA': 1,
    'Europe': 0,
    'Japan': 0
}

# Convert to DataFrame and ensure column order
new_car_df = pd.DataFrame([new_car_features])
new_car_df = new_car_df[feature_names]  # Ensure feature order consistency

# Standardize and predict
new_car_scaled = X_scaler.transform(new_car_df.values)
new_car_tensor = torch.FloatTensor(new_car_scaled)

with torch.no_grad():
    predicted_mpg_scaled = model(new_car_tensor).item()
    predicted_mpg = y_scaler.inverse_transform(
        np.array([[predicted_mpg_scaled]])
    )[0, 0]

print(f"\nNew Car Feature Prediction:")
for feature, value in new_car_features.items():
    print(f"  {feature}: {value}")
print(f"\nPredicted Fuel Efficiency: {predicted_mpg:.1f} MPG")

print("\n" + "=" * 60)
print("Project 2 Completed Successfully!")
print("=" * 60)