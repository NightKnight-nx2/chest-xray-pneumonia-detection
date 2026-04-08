from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(224, 224, 3)):
    # Load ResNet50 pretrained on ImageNet without top layers
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base layers
    for layer in base_model.layers:
        layer.trainable = False
        
    # Add custom head for Binary Classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Final layer for binary classification (Pneumonia vs Normal)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Output model summary
    model.summary()
    
    return model

if __name__ == "__main__":
    model = build_model()
