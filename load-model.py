import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# بارگذاری داده‌های MNIST از فایل 'mnist.npz'
with np.load('mnist.npz') as data:
    x_test, y_test = data['x_test'], data['y_test']

x_test = x_test / 255.0
# بارگذاری مدل ذخیره‌شده
model = load_model('mnist_model.h5')

# اگر داده‌های تست (x_test) را قبلاً بارگذاری کرده‌اید، نیازی به بارگذاری مجدد آن نیست.
# فقط پیش‌بینی روی داده‌های تست انجام می‌دهید:
predictions = model.predict(x_test)

# نمایش ۵ تصویر از داده‌های تست و پیش‌بینی مدل
fig, axes = plt.subplots(1, 10, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(x_test[i], cmap="gray")
    ax.set_title(f"Predicted: {np.argmax(predictions[i])}")
    ax.axis("off")

plt.tight_layout()
plt.show()
