"""
Basic KAN Example
=================

This example demonstrates how to use the classic KAN implementation
for a simple regression task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from kan_module import KAN

def generate_data(n_samples=1000):
    """Generate synthetic data for regression"""
    x = torch.linspace(-2, 2, n_samples).unsqueeze(1)
    # Target function: y = x^3 + 0.5*sin(5*x)
    y = x**3 + 0.5 * torch.sin(5 * x) + 0.1 * torch.randn_like(x)
    return x, y

def train_kan_model():
    """Train a KAN model on synthetic data"""
    print("Generating synthetic data...")
    x_train, y_train = generate_data(800)
    x_test, y_test = generate_data(200)
    
    print("Creating KAN model...")
    # Create KAN model: 1 input -> 5 hidden -> 1 output
    model = KAN([1, 5, 1], grid_size=5, spline_order=3)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("Training model...")
    losses = []
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        # Add regularization loss
        reg_loss = 0
        for layer in model.layers:
            reg_loss += layer.regularization_loss(regularize_activation=1.0, regularize_entropy=1.0)
        
        total_loss = loss + 0.01 * reg_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {total_loss.item():.6f}')
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test)
        test_loss = criterion(y_pred_test, y_test)
        print(f'Test Loss: {test_loss.item():.6f}')
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    x_plot = torch.linspace(-2, 2, 200).unsqueeze(1)
    with torch.no_grad():
        y_plot = model(x_plot)
    
    plt.scatter(x_test.numpy(), y_test.numpy(), alpha=0.5, label='True', s=10)
    plt.plot(x_plot.numpy(), y_plot.numpy(), 'r-', label='KAN Prediction', linewidth=2)
    plt.title('KAN Regression Results')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('kan_module/examples/basic_kan_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model

if __name__ == "__main__":
    print("=== Basic KAN Example ===")
    model = train_kan_model()
    print("Training completed! Results saved to 'basic_kan_results.png'")