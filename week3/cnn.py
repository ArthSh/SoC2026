"""
Week 3 - Deep Learning & CNN

Simple CNN implementation using TensorFlow/Keras
for handwritten digit classification.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# ------------------------------
# Load Dataset
# ------------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize

x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

print("Training Images:", x_train.shape)
print("Testing Images :", x_test.shape)

# ------------------------------
# CNN Model
# ------------------------------

model = Sequential([

    layers.Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu",
        input_shape=(28,28,1)
    ),

    layers.MaxPooling2D((2,2)),

    layers.Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation="relu"
    ),

    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dense(10, activation="softmax")

])

# ------------------------------
# Compile Model
# ------------------------------

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

print(model.summary())

# ------------------------------
# Train Model
# ------------------------------

history = model.fit(

    x_train,

    y_train,

    epochs=5,

    validation_split=0.2

)

# ------------------------------
# Evaluate
# ------------------------------

loss, accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", round(accuracy*100,2),"%")

# ------------------------------
# Prediction Example
# ------------------------------

prediction = model.predict(x_test[:1])

print("\nPredicted Digit:", prediction.argmax())
print("Actual Digit   :", y_test[0])
