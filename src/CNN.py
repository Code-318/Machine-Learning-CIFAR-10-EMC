import numpy as np
import tensorflow as tf
from tensorflow import keras

#Load the trained model
model = keras.models.load_model("my_cnn.keras")

# Output the expected input
print("Expected input shape:", model.input_shape)

# Load and preprocess the new image
img_height, img_width = 32, 32 #Model dimensions
img = keras.utils.load_img("test_image.jpg", target_size=(img_height, img_width))
img_array = keras.utils.img_to_array(img)

# Match model normalization
img_array = img_array / 255.0

# Add a batch dimension
img_batch = np.expand_dims(img_array, axis=0)

# Run prediction
predictions = model.predict(img_batch)

# Map the output to your class names (in the exact order used during training)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
predicted_index = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print(f"Prediction: {class_names[predicted_index]} ({confidence * 100:.2f}%)")
