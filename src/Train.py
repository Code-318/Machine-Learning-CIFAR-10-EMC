#Imports **Required to run**
from datasets import load_dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#Constants
EPOCHS = 60
BATCH = 64

#Load dataset
ds = load_dataset("uoft-cs/cifar10")

#Load full dataset
x_full = np.stack([np.array(img) for img in ds["train"]["img"]]).astype("float32") / 255.0
y_full = np.array(ds["train"]["label"]).reshape(-1, 1)

#Split data into training and validation (validate once per epoch)
x_train, y_train = x_full[:45000], y_full[:45000]
x_val, y_val = x_full[45000:], y_full[45000:]

#Fresh data for testing (Only test once at end)
x_test = np.stack([np.array(img) for img in ds["test"]["img"]]).astype("float32") / 255.0
y_test = np.array(ds["test"]["label"]).reshape(-1, 1)

#Residual Block Function - residual learning mechanism
def res_block(x, filters):
    shortcut = x #Saves the original input, to resolve vanishing gradient problem - and to alow deeper learning
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x) #Applies a 3x3 convolutional layer to identify features or patterns
    x = layers.BatchNormalization()(x) #Normalises the inputs so that each epoch has similar inputs each time, which prevents 'moving targets'.
    x = layers.Activation('relu')(x) #Activation layer to allow for some linearity
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, use_bias=False)(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.Activation('relu')(x)

inputs = layers.Input((32, 32, 3))
x = layers.RandomFlip('horizontal_and_vertical')(inputs)
x = layers.RandomTranslation(0.1, 0.1)(x)
x = layers.RandomRotation(0.2)(x)


x = res_block(x, 64)
x = res_block(x, 64)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.1)(x)

x = res_block(x, 128)
x = res_block(x, 128)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.2)(x)

x = res_block(x, 256)
x = res_block(x, 256)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.3)(x)

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10)(x)
model = models.Model(inputs, outputs)

steps_per_epoch = -(-len(x_train) // BATCH)   # 704
lr = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=3e-5,           # start really small
    decay_steps=(EPOCHS - 3) * steps_per_epoch,
    warmup_target=1e-3,                   # ramp it up to this
    warmup_steps=3 * steps_per_epoch)     # over 3 epochs it increases, then decreases till the end.

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=3e-4
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    x_train, y_train, epochs=EPOCHS, batch_size=BATCH,
    validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

model.save("my_cnn.keras")

h = history.history

plt.figure()
plt.plot(h["accuracy"], label="Training accuracy")
plt.plot(h["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.savefig("goat_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(h["loss"], label="Training loss")
plt.plot(h["val_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.savefig("goat_loss.png", dpi=300, bbox_inches="tight")
plt.show()

from google.colab import files
files.download("goat_accuracy.png")
files.download("goat_loss.png")
files.download("my_cnn.keras")
