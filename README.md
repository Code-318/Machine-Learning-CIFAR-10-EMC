# CIFAR-10 Image Classification CNN EMC Machine Learning

## Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images from the CIFAR-10 dataset.

CIFAR-10 contains 60,000 colour images, each measuring 32 × 32 pixels, divided into 10 different classes.

The objective is to train a neural network that can identify the object represented in an image.

The project uses a custom residual CNN architecture, includes convolutional layers, batch normalisation, residual connections and dropout to improve learning and generalisation.

## Technologies and Dependencies

The project is written in Python and uses the following libraries:

- **TensorFlow / Keras** – Neural Network construction and training.
- **NumPy** – Numerical operations and data manipulation.
- **Matplotlib** – Visualisation of training results.

You can install the required dependencies using:

```bash
pip install tensorflow numpy matplotlib
```
(Note: The installation method may vary depending on your IDE or environment.)
## Dataset

The model uses the CIFAR-10 dataset, which contains 60,000 labelled images across 10 classes:

- Aeroplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is divided into three subsets:

| Subset | Number of images | Reason |
|---|:---:|:---:|
| Training | 45,000 | To train |
| Validation | 5,000 | Test each Epoch |
| Testing | 10,000 | Final test |

Pixel values are normalised to the range 0–1 by dividing by 255 to allow for smooth steady adjustments in weight.

Data augmentation is applied during training using random horizontal and vertical flips, translations and rotations. This introduces variation/augmentation into the training images and helps to reduce overfitting. (where the model memorises exact pixel values rather than patterns.)

## Model Architecture

The model uses a custom ResNet-style architecture consisting of six residual blocks arranged into three stages.

Each residual block contains:

1. A 3 × 3 convolutional layer.
2. Batch normalisation.
3. ReLU activation.
4. A second 3 × 3 convolutional layer.
5. Batch normalisation.
6. A residual connection that adds the original input to the processed output.
7. A final ReLU activation.

When the input and output have different channel dimensions, a 1 × 1 convolution is used to adjust the shortcut connection so that it can cleanly be added to the output. We add the shortcut to avoid 'vanishing gradients'.

### Network Structure

| Stage | Residual blocks | Filters | Output resolution |
|---|---:|---:|---:|
| Initial | 2 | 64 | 32 × 32 |
| Second | 2 | 128 | 16 × 16 |
| Third | 2 | 256 | 8 × 8 |

Max pooling is applied after each stage to reduce spatial dimensions.

Dropout is also applied after each stage to help prevent overfitting. The dropout rates increase from 0.1 to 0.3.

Finally, Global Average Pooling reduces the feature maps to a single vector, which is passed into a Dense layer containing 10 outputs, one for each CIFAR-10 class.

## Training Configuration

The model is trained using the following configuration:

| Parameter | Value |
|---|---:|
| Epochs | 60 |
| Batch size | 64 |
| Optimiser | Adam |
| Initial learning rate | 0.00003 |
| Maximum learning rate | 0.001 |
| Weight decay | 0.0003 |
| Learning rate schedule | Warmup followed by cosine decay |
| Loss function | Sparse Categorical Crossentropy |

A learning rate warmup is used during the first three epochs, gradually increasing the learning rate from 0.00001 to 0.001.

After warmup, cosine decay gradually reduces the learning rate over the remaining training epochs.

## Evaluation

The dataset's validation subset is used to monitor model performance during training.

After training, the model is evaluated against the separate test dataset.

The main evaluation metrics are:

- **Accuracy** – the proportion of images classified correctly.
- **Loss** – a measure of the difference between the predicted class probabilities and the correct labels.

Training and validation accuracy and loss are also plotted to help assess learning performance and identify potential overfitting.

### Results

| Metric | Result |
|---|---|
| Final training accuracy | [Add result] |
| Final validation accuracy | [Add result] |
| Test accuracy | [Add result] |
| Test loss | [Add result] |

## Saving the Model

After training, the model is saved in Keras format:

```python
model.save("my_cnn.keras")
```

This allows the trained model to be loaded later without having to repeat the training process.

## Running the Project

1. Open Google Collab or your chosen IDE
2. If using collab, use our .ipynb File, Otherwise use our .py file.
3. Run the code 
4. Wait for training and evaluation to complete.
5. View the generated accuracy and loss graphs and the final test results.

(We recommend running the code in Google Collab, that way you can use their GPU and re run the cell 2 without having to re import the Ciphar 10 database)
