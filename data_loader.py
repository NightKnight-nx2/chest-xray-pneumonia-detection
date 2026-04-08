import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

def get_data_generators(data_dir, target_size=(224, 224), batch_size=32):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Data Augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2]
    )
    
    # Only rescale for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator, train_datagen

def compute_weights(train_generator):
    classes = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(classes),
        y=classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Calculated Class Weights: {class_weight_dict}")
    return class_weight_dict

if __name__ == "__main__":
    base_dir = r"c:\Users\ferat\Masaüstü\Kodlama\Göğüs Röntgeni ile Zatürre (Pneumonia) Teşhisi"
    data_dir = os.path.join(base_dir, "data", "chest_xray")
    if os.path.exists(data_dir):
        train_gen, val_gen, test_gen, _ = get_data_generators(data_dir)
        compute_weights(train_gen)
    else:
        print("Data directory not found. Please wait for the download to finish.")
