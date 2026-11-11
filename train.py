
import os
import tensorflow as tf
from tensorflow.keras import layers, models

dataset_path = "dataset"

image_size = (128, 128)
batch_size = 4
epochs = 10

class_names = ["Clean", "Dirty"]

# Dataset mein images check karna
for category in class_names:
    folder = os.path.join(dataset_path, category)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(category, ":", len(images), "images")

    if len(images) < 2:
        raise ValueError("Har category mein images honi chahiye.")

# Dataset load karna
train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    class_names=class_names
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=image_size,
    batch_size=batch_size,
    class_names=class_names
)

# Simple CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Model training
model.fit(
    train_data,
    validation_data=validation_data,
    epochs=epochs
)

# Model save karna
os.makedirs("models", exist_ok=True)

model.save("models/cleanliness_model.keras")

print("Model training completed!")
