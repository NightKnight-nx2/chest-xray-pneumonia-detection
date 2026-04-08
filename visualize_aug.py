import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from data_loader import get_data_generators
import numpy as np

def visualize_augmentations():
    base_dir = r"c:\Users\ferat\Masaüstü\Kodlama\Göğüs Röntgeni ile Zatürre (Pneumonia) Teşhisi"
    data_dir = os.path.join(base_dir, "data", "chest_xray")
    train_dir = os.path.join(data_dir, "train")
    
    normal_dir = os.path.join(train_dir, "NORMAL")
    pneumonia_dir = os.path.join(train_dir, "PNEUMONIA")
    
    artifacts_dir = r"C:\Users\ferat\.gemini\antigravity\brain\47d1492d-955c-4ba1-920e-e357f93a30dc"

    if not os.path.exists(normal_dir):
        print("Data directories not found.")
        return

    # Select 1 random Normal and 1 random Pneumonia image
    normal_img_path = os.path.join(normal_dir, random.choice(os.listdir(normal_dir)))
    pneumonia_img_path = os.path.join(pneumonia_dir, random.choice(os.listdir(pneumonia_dir)))

    _, _, _, train_datagen = get_data_generators(data_dir)

    def process_and_plot(img_path, title_prefix, ax_row):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        # Plot original
        ax_row[0].imshow(img)
        ax_row[0].set_title(f"{title_prefix} - Original")
        ax_row[0].axis('off')

        # Generate 4 augmentations
        i = 1
        for batch in train_datagen.flow(img_array, batch_size=1):
            ax_row[i].imshow(batch[0])
            ax_row[i].set_title(f"Augmented {i}")
            ax_row[i].axis('off')
            i += 1
            if i > 4:
                break

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    process_and_plot(normal_img_path, "Normal", axes[0])
    process_and_plot(pneumonia_img_path, "Pneumonia", axes[1])

    plt.tight_layout()
    aug_path = os.path.join(artifacts_dir, 'augmentation_samples.png')
    plt.savefig(aug_path, bbox_inches='tight')
    print(f"Saved augmentation visualization to {aug_path}")
    plt.close()

if __name__ == "__main__":
    visualize_augmentations()
