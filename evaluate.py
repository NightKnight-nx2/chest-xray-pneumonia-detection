import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import get_data_generators

def evaluate():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "chest_xray")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(base_dir, 'best_model.h5')

    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        return

    # Load Model
    model = load_model(model_path)
    
    # Get test generator
    _, _, test_gen, _ = get_data_generators(data_dir, target_size=(224, 224), batch_size=32)
    
    # Predict
    print("Evaluating model on test set...")
    predictions = model.predict(test_gen, steps=len(test_gen))
    y_pred = (predictions > 0.5).astype(int).reshape(-1)
    y_true = test_gen.classes
    
    # Classification Report
    target_names = ['Normal', 'Pneumonia']
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 14})
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    cm_path = os.path.join(artifacts_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, bbox_inches='tight')
    print(f"Saved confusion matrix plot to {cm_path}")
    plt.close()

if __name__ == "__main__":
    evaluate()
