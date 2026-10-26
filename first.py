import tensorflow as ts
import numpy as np
import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist

with np.load("mnist.npz") as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
# (x_train,y_train), (x_test,y_test) = mnist.load_data()

print(f"Training set shape: {x_train.shape}, Labels: {y_train.shape}")
print(f"Test set shape: {x_test.shape}, Labels: {y_test.shape}")

# نرمال‌سازی داده‌ها
x_train = x_train / 255.0
x_test = x_test / 255.0

# بررسی شکل داده‌ها
print(x_train.shape)
print(y_train.shape)


fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(x_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")

plt.show()
