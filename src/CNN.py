
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model("official_cnn.keras")

print("Expected input shape:", model.input_shape)

# Load and preprocess the image
name = input("Which image would you like to test? (e.g., 'dog1'): ")
img = keras.utils.load_img(
    name+".jpg",
    target_size=(32, 32),
    color_mode="rgb"
)

img_array = keras.utils.img_to_array(img)

# Normalise pixel values to [0, 1]
img_array = img_array / 255.0

# Add batch dimension
img_batch = np.expand_dims(img_array, axis=0)

# Generate predictions (raw logits)
logits = model.predict(img_batch, verbose=0)

# Convert logits into probabilities
probabilities = tf.nn.softmax(logits[0]).numpy()

# CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Find the most likely class
predicted_index = np.argmax(probabilities)
confidence = probabilities[predicted_index]

print(f"Prediction: {class_names[predicted_index]}")
print(f"Confidence: {confidence * 100:.2f}%")
