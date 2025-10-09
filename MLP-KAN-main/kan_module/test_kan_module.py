"""
Test script for KAN Module
===========================

This script tests the basic functionality of the extracted KAN module
to ensure it works independently.
"""

import sys
import os
import torch
import traceback

# Add the current directory to Python path for testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports"""
    print("Testing basic imports...")
    try:
        from ekan import KAN
        from fasterkan import FasterKAN
        print("✓ Basic KAN imports successful")
        return True
    except Exception as e:
        print(f"✗ Basic imports failed: {e}")
        traceback.print_exc()
        return False

def test_vision_imports():
    """Test vision-related imports"""
    print("Testing vision imports...")
    try:
        from vision_kan import MoE_KAN_MLP, kanBlock, create_kan
        print("✓ Vision KAN imports successful")
        return True
    except Exception as e:
        print(f"✗ Vision imports failed: {e}")
        print("Note: This might be expected if timm is not installed")
        return False

def test_kan_functionality():
    """Test basic KAN functionality"""
    print("Testing KAN functionality...")
    try:
        from ekan import KAN
        
        # Create a simple KAN model
        model = KAN([2, 5, 1])
        
        # Test forward pass
        x = torch.randn(10, 2)
        y = model(x)
        
        assert y.shape == (10, 1), f"Expected shape (10, 1), got {y.shape}"
        print("✓ KAN forward pass successful")
        
        # Test regularization
        reg_loss = 0
        for layer in model.layers:
            reg_loss += layer.regularization_loss(regularize_activation=1.0, regularize_entropy=1.0)
        
        assert isinstance(reg_loss, torch.Tensor), "Regularization loss should be a tensor"
        print("✓ KAN regularization successful")
        
        return True
    except Exception as e:
        print(f"✗ KAN functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_faster_kan_functionality():
    """Test FasterKAN functionality"""
    print("Testing FasterKAN functionality...")
    try:
        from fasterkan import FasterKAN
        
        # Create a FasterKAN model
        model = FasterKAN([2, 5, 1])
        
        # Test forward pass
        x = torch.randn(10, 2)
        y = model(x)
        
        assert y.shape == (10, 1), f"Expected shape (10, 1), got {y.shape}"
        print("✓ FasterKAN forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ FasterKAN functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_moe_kan_functionality():
    """Test MoE KAN functionality"""
    print("Testing MoE KAN functionality...")
    try:
        from vision_kan import MoE_KAN_MLP
        
        # Create MoE KAN MLP
        moe_kan = MoE_KAN_MLP(
            hidden_dim=64,
            ffn_dim=256,
            num_experts=4,
            top_k=2
        )
        
        # Test forward pass
        x = torch.randn(8, 64)  # [seq_len, hidden_dim]
        y = moe_kan(x)
        
        assert y.shape == (8, 64), f"Expected shape (8, 64), got {y.shape}"
        print("✓ MoE KAN MLP forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ MoE KAN functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_kan_block_functionality():
    """Test kanBlock functionality"""
    print("Testing kanBlock functionality...")
    try:
        from vision_kan import kanBlock
        
        # Create kanBlock
        block = kanBlock(
            dim=64,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.1,
            attn_drop=0.1,
            drop_path=0.1
        )
        
        # Test forward pass
        x = torch.randn(2, 17, 64)  # [batch, seq_len, dim]
        y = block(x)
        
        assert y.shape == (2, 17, 64), f"Expected shape (2, 17, 64), got {y.shape}"
        print("✓ kanBlock forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ kanBlock functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_create_kan_functionality():
    """Test create_kan functionality"""
    print("Testing create_kan functionality...")
    try:
        from vision_kan import create_kan, TIMM_AVAILABLE
        
        if not TIMM_AVAILABLE:
            print("✓ create_kan correctly reports timm dependency")
            return True
        
        # Create a Vision KAN model
        model = create_kan('deit_tiny_patch16_224_KAN', pretrained=False, num_classes=10)
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        
        assert y.shape == (1, 10), f"Expected shape (1, 10), got {y.shape}"
        print("✓ create_kan forward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ create_kan functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_gradient_flow():
    """Test gradient flow through KAN models"""
    print("Testing gradient flow...")
    try:
        from ekan import KAN
        from fasterkan import FasterKAN
        
        # Test KAN gradient flow
        model = KAN([2, 5, 1])
        x = torch.randn(10, 2, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Check if gradients exist
        assert x.grad is not None, "Input gradients should exist"
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                has_grads = True
                break
        assert has_grads, "At least some model parameter gradients should exist"
        
        print("✓ KAN gradient flow successful")
        
        # Test FasterKAN gradient flow
        model = FasterKAN([2, 5, 1])
        x = torch.randn(10, 2, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()
        
        # Check if gradients exist
        assert x.grad is not None, "Input gradients should exist"
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                has_grads = True
                break
        assert has_grads, "At least some model parameter gradients should exist"
        
        print("✓ FasterKAN gradient flow successful")
        
        return True
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("KAN Module Test Suite")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Vision Imports", test_vision_imports),
        ("KAN Functionality", test_kan_functionality),
        ("FasterKAN Functionality", test_faster_kan_functionality),
        ("MoE KAN Functionality", test_moe_kan_functionality),
        ("kanBlock Functionality", test_kan_block_functionality),
        ("create_kan Functionality", test_create_kan_functionality),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25} | {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! KAN module is working correctly.")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)