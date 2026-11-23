#Imports **Required to run**
from datasets import load_dataset
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#Constants
EPOCHS = 60
BATCH = 64

#Check if GPU is connected (Reccommended to have GPU otherwise it will be very slow)
print("GPUs:", tf.config.list_physical_devices('GPU'))

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
def res_block(x, filters): #Takes 2 parameters, x - inputs and filters-number of filters
    shortcut = x #Saves the original input, to resolve vanishing gradient problem - and bypass the conv layers later (during backpropogation).
    #First Convolution
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x) #Applies a 3x3 convolutional layer (kernel) to identify features or patterns
    x = layers.BatchNormalization()(x) #Normalises the activations so that each batch has similar distributions. This helps stabilise training and makes optimisation easier.
    x = layers.Activation('relu')(x) #Activation layer to allow for non-linearity. (Relu makes negative values 0, and leaves positives unchanged)
    #Second convolution - almost identical to the prior
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters: #Checks whether the number of channels in input matches the number of filters in the conv layer.
        shortcut = layers.Conv2D(filters, 1, use_bias=False)(shortcut) #If not - fix the dimensions
    x = layers.Add()([x, shortcut]) #Adds the shortcut back onto the output.
    return layers.Activation('relu')(x) #Only normalises after adding the shortcut back onto the output. This reduces vanishing gradients by making backpropogation easier.

#Take the inputs
inputs = layers.Input((32, 32, 3))

#Apply augmentation every single epoch to avoid overfitting - where the model memorises the training datas pixels rather than patterns.
x = layers.RandomFlip('horizontal_and_vertical')(inputs)
x = layers.RandomTranslation(0.1, 0.1)(x)
x = layers.RandomRotation(0.2)(x)

#First residual block 
x = res_block(x, 64)
x = res_block(x, 64)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.1)(x) #Dropout disables some random neurons, which reduces overfitting and helps to make the model more well rounded.

#Second residual block
x = res_block(x, 128)
x = res_block(x, 128)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.2)(x) #Increase dropout for higher filter res block

#Third residual block
x = res_block(x, 256)
x = res_block(x, 256)
x = layers.MaxPooling2D()(x)
x = layers.Dropout(0.3)(x) #Increase dropout for higher filter res block

x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(10)(x)
model = models.Model(inputs, outputs) #Initialise the model with keras

steps_per_epoch = -(-len(x_train) // BATCH)   # 704 in this case
lr = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=3e-5,           # start really small
    decay_steps=(EPOCHS - 3) * steps_per_epoch,
    warmup_target=1e-3,                   # ramp it up to this during warmup
    warmup_steps=3 * steps_per_epoch)     # over 3 epochs it increases, then decreases using cosine-delay till the end.

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=lr,
        weight_decay=3e-4 #Decays the weight, preventing the model from fixating on local maxima, and reduces overfitting.
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(
    x_train, y_train, epochs=EPOCHS, batch_size=BATCH,
    validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) #Evaluate the model
print(f"Test accuracy: {test_acc:.4f}") #Output the test result

model.save("my_cnn.keras") #Save the model

h = history.history

#Plotting training figures
#Accuracy figs
plt.figure()
plt.plot(h["accuracy"], label="Training accuracy")
plt.plot(h["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.savefig("goat_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

#Loss figs
plt.figure()
plt.plot(h["loss"], label="Training loss")
plt.plot(h["val_loss"], label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.savefig("goat_loss.png", dpi=300, bbox_inches="tight")
plt.show()

#Download from google colab
from google.colab import files
files.download("goat_accuracy.png")
files.download("goat_loss.png")
files.download("my_cnn.keras")
