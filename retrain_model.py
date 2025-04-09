import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.models import load_model, Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Input  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.utils.class_weight import compute_class_weight  # type: ignore

# Paths
base_dir = 'dataset'  # Folder with lily, sunflower, french_rose, etc.
model_path = 'C:/Users/sdeem/OneDrive/Documents/GitHub/green-glow/model.h5'
new_model_path = 'C:/Users/sdeem/OneDrive/Documents/GitHub/green-glow/retrained_model.h5'

# Settings
image_size = (256, 256)
batch_size = 32
epochs = 25  # Increase from 5 to 25 for better training

# Load the existing model
old_model = load_model(model_path)

# Unfreeze last 10 layers for fine-tuning (optional)
for layer in old_model.layers[:-10]:
    layer.trainable = False
for layer in old_model.layers[-10:]:
    layer.trainable = True

# Build new model on top of old one
input_layer = Input(shape=(256, 256, 3))
x = old_model(input_layer)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(os.listdir(base_dir)), activation='softmax')(x)

new_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the new model
new_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Image augmentation and data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
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

# Compute class weights (optional but useful for imbalance)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))

# Train the model
history = new_model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen,
    class_weight=class_weights
)

# Save the retrained model
new_model.save(new_model_path)
print(" Model updated and saved to:", new_model_path)

# Print updated class mapping
print("\n Add the following to your disease_map in data.py:")
for label, index in train_gen.class_indices.items():
    print(f"{index}: '{label.replace('_', ' ').title()}'")

