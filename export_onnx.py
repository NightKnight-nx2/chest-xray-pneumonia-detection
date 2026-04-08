import os
import torch
import torch.nn as nn
from torchvision import models

def export_to_onnx():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'best_pytorch_model.pth')
    web_dir = os.path.join(base_dir, "web", "public", "model")
    
    if not os.path.exists(model_path):
        print("Model file not found. Train the PyTorch model first.")
        return

    print("Loading PyTorch model...")
    # Same architecture defined in train_pytorch.py
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Create dummy input for ONNX tracing
    dummy_input = torch.randn(1, 3, 224, 224)

    # Output paths
    web_dir = os.path.join(base_dir, "web", "public", "model")
    os.makedirs(web_dir, exist_ok=True)
    onnx_path = os.path.join(web_dir, 'model.onnx')

    print("Exporting to ONNX format...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX export successful! Model saved to: {onnx_path}")

if __name__ == '__main__':
    export_to_onnx()
