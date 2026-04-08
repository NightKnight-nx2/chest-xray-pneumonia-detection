import os
import cv2
import glob
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def compute_gradcam(model, img_array, last_conv_layer_name='conv5_block3_out'):
    # In some Keras versions loading .h5 flattens the nested model.
    try:
        last_conv_layer = model.get_layer('resnet50').get_layer(last_conv_layer_name)
    except Exception:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    
    # Create grad model
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
        
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def generate_visualization():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "chest_xray", "test")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(base_dir, 'best_model.h5')

    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        return

    model = load_model(model_path)
    
    pneumonia_dir = os.path.join(data_dir, "PNEUMONIA")
    img_path = random.choice(glob.glob(os.path.join(pneumonia_dir, "*.jpeg")))
    
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0) / 255.0

    heatmap = compute_gradcam(model, img_array_expanded, 'conv5_block3_out')

    # Resize and overlay
    img_cv = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_cv = cv2.resize(img_cv, (224, 224))
    
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

    overlayed_img = heatmap_color * 0.4 + img_cv * 0.6
    overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)

    # Plot Original, Heatmap, Overlayed
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlayed Heatmap')
    axes[2].axis('off')

    plt.tight_layout()
    grad_cam_path = os.path.join(artifacts_dir, 'gradcam_visualization.png')
    plt.savefig(grad_cam_path, bbox_inches='tight')
    print(f"Saved Grad-CAM visualization to {grad_cam_path}")
    plt.close()

if __name__ == "__main__":
    generate_visualization()
