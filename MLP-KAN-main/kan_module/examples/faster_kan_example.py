"""
FasterKAN Example
=================

This example demonstrates how to use the optimized FasterKAN implementation
for a classification task on synthetic data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from kan_module import FasterKAN

def generate_classification_data():
    """Generate synthetic classification data"""
    X, y = make_classification(
        n_samples=2000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def train_faster_kan_classifier():
    """Train a FasterKAN model for classification"""
    print("Generating classification data...")
    X_train, X_test, y_train, y_test = generate_classification_data()
    
    print("Creating FasterKAN model...")
    # Create FasterKAN model: 2 inputs -> 10 hidden -> 2 outputs
    model = FasterKAN([2, 10, 2])
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training model...")
    train_losses = []
    train_accuracies = []
    
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_train).float().mean()
        
        train_losses.append(loss.item())
        train_accuracies.append(accuracy.item())
        
        if (epoch + 1) % 40 == 0:
            print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}')
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = (test_predicted == y_test).float().mean()
        
        print(f'Test Loss: {test_loss.item():.4f}')
        print(f'Test Accuracy: {test_accuracy.item():.4f}')
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot training curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot decision boundary
    plt.subplot(1, 3, 3)
    
    # Create a mesh for decision boundary
    h = 0.02
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    model.eval()
    with torch.no_grad():
        mesh_outputs = model(mesh_points)
        _, mesh_predicted = torch.max(mesh_outputs, 1)
        mesh_predicted = mesh_predicted.reshape(xx.shape)
    
    plt.contourf(xx, yy, mesh_predicted.numpy(), alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot test points
    colors = ['red', 'blue']
    for i in range(2):
        mask = y_test == i
        plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=colors[i], label=f'Class {i}', alpha=0.7)
    
    plt.title('FasterKAN Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('kan_module/examples/faster_kan_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model

def compare_with_mlp():
    """Compare FasterKAN with traditional MLP"""
    print("\n=== Comparing FasterKAN with MLP ===")
    
    X_train, X_test, y_train, y_test = generate_classification_data()
    
    # FasterKAN model
    kan_model = FasterKAN([2, 10, 2])
    kan_optimizer = optim.Adam(kan_model.parameters(), lr=0.01)
    
    # Traditional MLP
    mlp_model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)
    
    criterion = nn.CrossEntropyLoss()
    
    kan_accuracies = []
    mlp_accuracies = []
    
    for epoch in range(100):
        # Train KAN
        kan_model.train()
        kan_optimizer.zero_grad()
        kan_outputs = kan_model(X_train)
        kan_loss = criterion(kan_outputs, y_train)
        kan_loss.backward()
        kan_optimizer.step()
        
        # Train MLP
        mlp_model.train()
        mlp_optimizer.zero_grad()
        mlp_outputs = mlp_model(X_train)
        mlp_loss = criterion(mlp_outputs, y_train)
        mlp_loss.backward()
        mlp_optimizer.step()
        
        # Evaluate both models
        if epoch % 10 == 0:
            kan_model.eval()
            mlp_model.eval()
            
            with torch.no_grad():
                # KAN accuracy
                kan_test_outputs = kan_model(X_test)
                _, kan_predicted = torch.max(kan_test_outputs, 1)
                kan_acc = (kan_predicted == y_test).float().mean().item()
                kan_accuracies.append(kan_acc)
                
                # MLP accuracy
                mlp_test_outputs = mlp_model(X_test)
                _, mlp_predicted = torch.max(mlp_test_outputs, 1)
                mlp_acc = (mlp_predicted == y_test).float().mean().item()
                mlp_accuracies.append(mlp_acc)
                
                print(f'Epoch {epoch}: KAN Acc: {kan_acc:.4f}, MLP Acc: {mlp_acc:.4f}')
    
    print(f'Final - KAN: {kan_accuracies[-1]:.4f}, MLP: {mlp_accuracies[-1]:.4f}')

if __name__ == "__main__":
    print("=== FasterKAN Example ===")
    model = train_faster_kan_classifier()
    compare_with_mlp()
    print("Training completed! Results saved to 'faster_kan_results.png'")