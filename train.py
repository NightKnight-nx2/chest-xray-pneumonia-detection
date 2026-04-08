import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from data_loader import get_data_generators, compute_weights
from model import build_model

def train():
    base_dir = r"c:\Users\ferat\Masaüstü\Kodlama\Göğüs Röntgeni ile Zatürre (Pneumonia) Teşhisi"
    data_dir = os.path.join(base_dir, "data", "chest_xray")
    artifacts_dir = r"C:\Users\ferat\.gemini\antigravity\brain\47d1492d-955c-4ba1-920e-e357f93a30dc"

    if not os.path.exists(os.path.join(data_dir, "train")):
        print("Data directory not found.")
        return

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 3 # Adjusted for reasonable turnaround time
    TARGET_SIZE = (224, 224)
    LEARNING_RATE = 1e-4

    train_gen, val_gen, test_gen, _ = get_data_generators(
        data_dir, target_size=TARGET_SIZE, batch_size=BATCH_SIZE
    )

    class_weights = compute_weights(train_gen)

    model = build_model(input_shape=TARGET_SIZE + (3,))
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model_path = os.path.join(base_dir, 'best_model.h5')
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks
    )
    print("Training finished.")

    # Plot Accuracy and Loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Accuracy vs Epoch')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    # Loss Plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Loss vs Epoch')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(artifacts_dir, 'training_curves.png')
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved training curves to {plot_path}")
    plt.close()

if __name__ == "__main__":
    train()
