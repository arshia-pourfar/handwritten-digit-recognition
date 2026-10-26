import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models # type: ignore
# from tensorflow.keras.models import load_model

with np.load("mnist.npz") as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

# نرمال‌سازی داده‌ها
x_train = x_train / 255.0
x_test = x_test / 255.0

# بررسی شکل داده‌ها
print(x_train.shape)
print(y_train.shape)

model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # تبدیل تصاویر به یک بردار
    layers.Dense(128, activation='relu'),  # لایه مخفی با 128 نود
    layers.Dense(128, activation='relu'),  # لایه مخفی با 128 نود
    layers.Dropout(0.2),  # Dropout برای جلوگیری از overfitting
    layers.Dense(10, activation='softmax')  # لایه خروجی با 10 نود (برای هر عدد از 0 تا 9)
])

# کامپایل کردن مدل
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# آموزش مدل
model.fit(x_train, y_train, epochs=5)

# ارزیابی مدل روی داده‌های تست
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')

# پیش‌بینی بر روی تصویر تست
predictions = model.predict(x_test)

# نمایش اولین پیش‌بینی
# print(f"Predicted class: {np.argmax(predictions[0])}")

# نمایش اولین تصویر تست و پیش‌بینی مدل
fig, axes = plt.subplots(1, 5, figsize=(10, 5))
for i, ax in enumerate(axes):
    ax.imshow(x_test[i], cmap='gray')
    ax.set_title(f"Predicted: {np.argmax(predictions[i])}")
    # نمایش اولین پیش‌بینی پنج مدل اول
    print(f"Predicted class: {np.argmax(predictions[i])}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# ذخیره مدل
model.save("mnist_model.h5")

# بارگذاری مدل ذخیره‌شده
# model = load_model("mnist_model.h5")


# ///////////////////// برای 1 بعدا ////////////////////////////
# ///////////////////// برای بعدا ////////////////////////////
# ///////////////////// برای بعدا ////////////////////////////
# import cv2

# # لود کردن تصویر
# img = cv2.imread("my_digit.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (28, 28))  # تغییر سایز
# img = img / 255.0  # نرمال‌سازی
# img = img.reshape(1, 28, 28)  # تغییر شکل برای ورود به مدل

# # پیش‌بینی مدل
# prediction = model.predict(img)
# print(f"Predicted class: {np.argmax(prediction)}") /\/\/\/
# نمایش عددی که مدل تشخیص داده
# predicted_digit = np.argmax(prediction)
# print(f"Predicted class: {predicted_digit}")

# //////////////////////////////////// 2 ////////////////////////////////////////////////////////////

# import numpy as np
# import matplotlib.pyplot as plt

# # پیش‌بینی روی داده‌های تست
# predictions = model.predict(x_test)  # خروجی مدل برای داده‌های تست

# # نمایش ۵ تصویر اول داده‌های تست و پیش‌بینی آن‌ها
# fig, axes = plt.subplots(1, 5, figsize=(10, 3))  # تنظیم تعداد ساب‌پلات‌ها
# for i in range(5):
#     axes[i].imshow(x_test[i], cmap='gray')  # نمایش تصویر
#     predicted_label = np.argmax(predictions[i])  # پیدا کردن بیشترین مقدار در خروجی مدل
#     axes[i].set_title(f"Predicted: {predicted_label}")  # نمایش مقدار پیش‌بینی شده
#     axes[i].axis('off')  # حذف محور‌ها برای زیبایی
# plt.show()

# ///////////////////// برای 2 بعدا ////////////////////////////
# ///////////////////// برای بعدا ////////////////////////////
# ///////////////////// برای بعدا ////////////////////////////