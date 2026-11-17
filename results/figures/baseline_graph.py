import re
import matplotlib.pyplot as plt

text = """Epoch 1/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 8s 5ms/step - accuracy: 0.4578 - loss: 1.5149 - val_accuracy: 0.5744 - val_loss: 1.2287
Epoch 2/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.5908 - loss: 1.1750 - val_accuracy: 0.6316 - val_loss: 1.0688
Epoch 3/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.6418 - loss: 1.0315 - val_accuracy: 0.6458 - val_loss: 1.0400
Epoch 4/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.6693 - loss: 0.9486 - val_accuracy: 0.6630 - val_loss: 0.9674
Epoch 5/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.6914 - loss: 0.8888 - val_accuracy: 0.6798 - val_loss: 0.9277
Epoch 6/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.7087 - loss: 0.8378 - val_accuracy: 0.6966 - val_loss: 0.9009
Epoch 7/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.7236 - loss: 0.7906 - val_accuracy: 0.6964 - val_loss: 0.8908
Epoch 8/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.7402 - loss: 0.7508 - val_accuracy: 0.6874 - val_loss: 0.9222
Epoch 9/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.7498 - loss: 0.7092 - val_accuracy: 0.6978 - val_loss: 0.8980
Epoch 10/10
1407/1407 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.7623 - loss: 0.6756 - val_accuracy: 0.6980 - val_loss: 0.8863"""

val_accuracy = re.findall(r'val_accuracy: ([0-9.]+)', text)
accuracy = re.findall(r' accuracy: ([0-9.]+)', text)
print(val_accuracy)
print(accuracy)
Epochs = list(range(1, 11))


plt.plot(Epochs, [float(x) for x in val_accuracy], label='Validation Accuracy')
plt.plot(Epochs, [float(x) for x in accuracy], label='Training Accuracy')

plt.ylim(0,1)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.show()
