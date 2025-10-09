"""
Vision KAN Example
==================

This example demonstrates how to use Vision KAN models for image classification.
It shows how to create different Vision KAN architectures and use them for training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

try:
    from kan_module import create_kan, MoE_KAN_MLP, kanBlock
    VISION_KAN_AVAILABLE = True
except ImportError as e:
    print(f"Vision KAN not available: {e}")
    print("Please install timm: pip install timm>=0.3.2")
    VISION_KAN_AVAILABLE = False

def get_cifar10_data(batch_size=32):
    """Load CIFAR-10 dataset"""
    transform_train = transforms.Compose([
        transforms.Resize(224),  # Vision KAN expects 224x224 images
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

def train_vision_kan_model():
    """Train a Vision KAN model on CIFAR-10"""
    if not VISION_KAN_AVAILABLE:
        print("Vision KAN is not available. Please install required dependencies.")
        return None
    
    print("Loading CIFAR-10 dataset...")
    trainloader, testloader = get_cifar10_data(batch_size=16)  # Smaller batch size for demo
    
    print("Creating Vision KAN model...")
    # Create a tiny Vision KAN model for demonstration
    model = create_kan('deit_tiny_patch16_224_KAN', pretrained=False, num_classes=10)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop (shortened for demo)
    print("Training model...")
    train_losses = []
    train_accuracies = []
    
    num_epochs = 3  # Reduced for demo purposes
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'Accuracy: {100*correct/total:.2f}%')
                running_loss = 0.0
        
        epoch_accuracy = 100 * correct / total
        train_accuracies.append(epoch_accuracy)
        scheduler.step()
    
    # Test the model
    print("Testing model...")
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Loss: {test_loss/len(testloader):.4f}')
    
    return model

def demonstrate_moe_kan_mlp():
    """Demonstrate MoE KAN MLP usage"""
    if not VISION_KAN_AVAILABLE:
        print("Vision KAN is not available. Please install required dependencies.")
        return
    
    print("\n=== MoE KAN MLP Demonstration ===")
    
    # Create MoE KAN MLP
    moe_kan = MoE_KAN_MLP(
        hidden_dim=768,
        ffn_dim=3072,
        num_experts=8,
        top_k=2
    )
    
    # Example input (batch_size=4, seq_len=197, hidden_dim=768)
    # This simulates Vision Transformer patches + class token
    x = torch.randn(4, 197, 768)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = moe_kan(x)
    print(f"Output shape: {output.shape}")
    
    # Show expert utilization
    print("MoE KAN MLP created successfully!")
    print(f"Number of experts: {moe_kan.num_experts}")
    print(f"Top-k experts used: {moe_kan.top_k}")

def demonstrate_kan_block():
    """Demonstrate kanBlock usage"""
    if not VISION_KAN_AVAILABLE:
        print("Vision KAN is not available. Please install required dependencies.")
        return
    
    print("\n=== KAN Block Demonstration ===")
    
    # Create kanBlock (similar to Transformer block but with KAN)
    kan_block = kanBlock(
        dim=768,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.1
    )
    
    # Example input
    x = torch.randn(4, 197, 768)  # [batch, seq_len, dim]
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = kan_block(x)
    print(f"Output shape: {output.shape}")
    print("KAN Block created and tested successfully!")

def visualize_model_comparison():
    """Create a visualization comparing different architectures"""
    print("\n=== Model Architecture Comparison ===")
    
    # Model parameters comparison
    models_info = {
        'Traditional MLP': {'params': '~1M', 'activation': 'ReLU/GELU'},
        'KAN': {'params': '~1.2M', 'activation': 'B-spline basis'},
        'FasterKAN': {'params': '~1.1M', 'activation': 'Optimized splines'},
        'Vision KAN': {'params': '~5.7M', 'activation': 'MoE + KAN'},
    }
    
    print("Model Comparison:")
    print("-" * 50)
    for model, info in models_info.items():
        print(f"{model:15} | Params: {info['params']:8} | Activation: {info['activation']}")
    
    # Create a simple visualization
    plt.figure(figsize=(10, 6))
    
    # Simulated performance data (for illustration)
    models = list(models_info.keys())
    accuracy = [85.2, 87.1, 86.8, 89.3]  # Example accuracies
    training_time = [1.0, 1.8, 1.4, 2.5]  # Relative training time
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots()
    
    # Accuracy bars
    color = 'tab:blue'
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Accuracy (%)', color=color)
    bars1 = ax1.bar(x - width/2, accuracy, width, label='Accuracy', color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(80, 95)
    
    # Training time bars
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Relative Training Time', color=color)
    bars2 = ax2.bar(x + width/2, training_time, width, label='Training Time', color=color, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Labels and formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, accuracy):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}%', ha='center', va='bottom')
    
    for bar, val in zip(bars2, training_time):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.1f}x', ha='center', va='bottom')
    
    plt.title('Model Performance Comparison\n(Simulated Data for Illustration)')
    plt.tight_layout()
    plt.savefig('kan_module/examples/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=== Vision KAN Example ===")
    
    if VISION_KAN_AVAILABLE:
        # Train a small Vision KAN model
        model = train_vision_kan_model()
        
        # Demonstrate individual components
        demonstrate_moe_kan_mlp()
        demonstrate_kan_block()
    else:
        print("Vision KAN components are not available.")
        print("To use Vision KAN, please install: pip install timm>=0.3.2")
    
    # Show model comparison
    visualize_model_comparison()
    
    print("Vision KAN example completed!")