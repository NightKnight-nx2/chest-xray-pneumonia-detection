import os
import glob
import random
import matplotlib.pyplot as plt
import cv2

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "chest_xray", "train")
    normal_dir = os.path.join(data_dir, "NORMAL")
    pneumonia_dir = os.path.join(data_dir, "PNEUMONIA")
    
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    if not os.path.exists(normal_dir) or not os.path.exists(pneumonia_dir):
        print("Data directories not found. Make sure dataset is downloaded and extracted properly.")
        return

    normal_images = glob.glob(os.path.join(normal_dir, "*.*"))
    pneumonia_images = glob.glob(os.path.join(pneumonia_dir, "*.*"))

    num_normal = len(normal_images)
    num_pneumonia = len(pneumonia_images)

    print(f"Normal images: {num_normal}")
    print(f"Pneumonia images: {num_pneumonia}")

    # 1. Plot Class Distribution (Pie Chart)
    labels = ['Normal', 'Pneumonia']
    sizes = [num_normal, num_pneumonia]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.05, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90, textprops={'fontsize': 14})
    ax.axis('equal')
    plt.title('Training Data Class Distribution', fontsize=16)
    
    pie_path = os.path.join(artifacts_dir, 'class_distribution.png')
    plt.savefig(pie_path, bbox_inches='tight')
    print(f"Saved pie chart to {pie_path}")
    plt.close()

    # 2. Plot 4x5 Grid of Random Images (10 Normal, 10 Pneumonia)
    random.shuffle(normal_images)
    random.shuffle(pneumonia_images)

    selected_normal = normal_images[:10]
    selected_pneumonia = pneumonia_images[:10]
    
    # We will arrange them alternatively or 2 rows Normal, 2 rows Pneumonia
    # Let's do 2 rows Normal and 2 rows Pneumonia. 4 rows, 5 cols total.
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    import numpy as np
    
    for i in range(10):
        # Normal
        img = cv2.imdecode(np.fromfile(selected_normal[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title('Normal', color='blue', fontsize=12)
        axes[i].axis('off')
        
    for i in range(10):
        # Pneumonia
        img = cv2.imdecode(np.fromfile(selected_pneumonia[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        axes[10 + i].imshow(img, cmap='gray')
        axes[10 + i].set_title('Pneumonia', color='red', fontsize=12)
        axes[10 + i].axis('off')

    plt.tight_layout()
    grid_path = os.path.join(artifacts_dir, 'sample_grid.png')
    plt.savefig(grid_path, bbox_inches='tight')
    print(f"Saved sample grid to {grid_path}")
    plt.close()

if __name__ == "__main__":
    main()
