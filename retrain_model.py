import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

base_dir = 'new_flowers'  # Folder with lily, sunflower, french_rose
model_path = 'Green_Glow/model.h5'
new_model_path = 'Green-Glow/model_updated.h5'

image_size = (224, 224)
batch_size = 16
epochs = 5

old_model = load_model(model_path)
for layer in old_model.layers:
    layer.trainable = False

x = old_model.layers[-2].output
x = Dense(256, activation='relu')(x)
out = Dense(34, activation='softmax')(x)  # 31 original + 3 new

new_model = Model(inputs=old_model.input, outputs=out)
new_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

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

history = new_model.fit(
    train_gen,
    epochs=epochs,
    validation_data=val_gen
)

new_model.save(new_model_path)
print("Model updated and saved to:", new_model_path)

print("\nAdd the following to your disease_map in data.py:")
for i, label in enumerate(train_gen.class_indices):
    print(f"{i + 31}: '{label.replace('_', ' ').title()}'")