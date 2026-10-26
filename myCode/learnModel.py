import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# بارگذاری داده‌ها
with np.load("mnist.npz") as data:
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]
    
# نرمال‌سازی داده‌ها
x_train = x_train / 255.0
x_test = x_test / 255.0

# مدل با لایه‌های کانولوشن
model = models.Sequential([
    layers.Reshape(target_shape=(28, 28, 1), input_shape=(28, 28)),  # تبدیل ورودی به شکل مورد نیاز CNN
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),  # افزایش نرخ Dropout برای جلوگیری از Overfitting
    layers.Dense(10, activation="softmax")
])

# کامپایل مدل
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# استفاده از EarlyStopping برای جلوگیری از Overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# آموزش مدل
model.fit(x_train, y_train, epochs=20, validation_split=0.2, callbacks=[early_stopping])

# ارزیابی مدل
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

# پیش‌بینی
predictions = model.predict(x_test)

# ذخیره مدل
model.save("my_model_2.h5")
