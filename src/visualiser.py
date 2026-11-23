from datasets import load_dataset
import numpy as np

ds = load_dataset("uoft-cs/cifar10")

#Load the val and training data
x_full = np.stack([np.array(img) for img in ds["train"]["img"]]).astype("float32") / 255.0
y_full = np.array(ds["train"]["label"]).reshape(-1, 1)

#Load Test Data
x_test = np.stack([np.array(img) for img in ds["test"]["img"]]).astype("float32") / 255.0
y_test = np.array(ds["test"]["label"]).reshape(-1, 1)

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("GPUs:", tf.config.list_physical_devices('GPU'))

#X represents the actual image, Y represents the label of said image.
x_train, y_train = x_full[:45000], y_full[:45000]
x_val, y_val = x_full[45000:], y_full[45000:]

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[y_train[i][0]])
plt.show()
