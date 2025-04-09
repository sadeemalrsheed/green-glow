import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.layers import Dense, Input  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

base_dir = 'dataset'  # Folder with lily, sunflower, french_rose
model_path = 'C:/Users/sdeem/OneDrive/Documents/GitHub/green-glow/model.h5'
new_model_path = 'C:/Users/sdeem/OneDrive/Documents/GitHub/green-glow/retrained_model.h5'

image_size = (256, 256)  # Resize images to 256x256
batch_size = 16
epochs = 5

# Load the old model
old_model = load_model(model_path)

# Freeze all layers of the old model
for layer in old_model.layers:
    layer.trainable = False

# Define the input layer explicitly

input_layer = Input(shape=(256, 256, 3))  # Ensure input shape matches the image size
x = old_model(input_layer)

# Pass the input through the old model to reuse its features
x = old_model(input_layer)

# Add new layers on top
num_classes = len(os.listdir(base_dir))  # Count folders = number of classes
out = Dense(num_classes, activation='softmax')(x)
  # 31 original + 3 new classes

# Create the new model
new_model = Model(inputs=input_layer, outputs=out)

# Compile the new model
new_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Create the data generators for training and validation
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# Train the new model
history = new_model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

# Save the retrained model
new_model.save(new_model_path)
print("Model updated and saved to:", new_model_path)

# Print the class mapping for the new disease_map
print("\nAdd the following to your disease_map in data.py:")
for label, index in train_gen.class_indices.items():
    print(f"{index}: '{label.replace('_', ' ').title()}'")
