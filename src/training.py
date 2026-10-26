"""آموزش مدل‌های تشخیص رقم دست‌نویس.

کد آموزش از اسکریپت‌های اصلی پروژه (second.py و learnModel.py) منتقل شده است؛
معماری مدل‌ها، ابرپارامترها و روش آموزش دقیقاً مثل قبل حفظ شده‌اند.
"""

from pathlib import Path

import numpy as np
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "mnist.npz"
MODELS_DIR = PROJECT_ROOT / "models"

# مدل‌های پشتیبانی‌شده برای آموزش
SUPPORTED_MODELS = {
    "1": {
        "name": "MLP (baseline)",
        "description": "Fully connected network: Flatten -> Dense(128) -> Dense(128) "
                       "-> Dropout(0.2) -> Dense(10)",
        "save_path": MODELS_DIR / "mnist_model.h5",
        "default_epochs": 5,
    },
    "2": {
        "name": "CNN (convolutional)",
        "description": "3 conv blocks (32/64/128 filters) with pooling "
                       "-> Dense(128) -> Dropout(0.3) -> Dense(10)",
        "save_path": MODELS_DIR / "my_model.h5",
        "default_epochs": 20,
    },
}


def load_mnist():
    """بارگذاری داده‌های MNIST از datasets/mnist.npz و نرمال‌سازی پیکسل‌ها"""

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "MNIST is required for training. Place a file named 'mnist.npz' inside the "
            "'datasets' folder. You can create it with:\n"
            "    from tensorflow.keras.datasets import mnist\n"
            "    import numpy as np\n"
            "    (x_train, y_train), (x_test, y_test) = mnist.load_data()\n"
            "    np.savez('mnist.npz', x_train=x_train, y_train=y_train, "
            "x_test=x_test, y_test=y_test)"
        )

    with np.load(DATASET_PATH) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]

    # نرمال‌سازی داده‌ها
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


def build_mlp():
    """مدل MLP پایه (مثل second.py)"""

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
    return model


def build_cnn():
    """مدل کانولوشنی (مثل learnModel.py)"""

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
    return model


def train_model(model_key, epochs):
    """آموزش مدل انتخاب‌شده و ذخیره‌ی آن در پوشه‌ی models

    خروجی: مسیر فایل مدل ذخیره‌شده
    """

    info = SUPPORTED_MODELS[model_key]
    x_train, y_train, x_test, y_test = load_mnist()

    print(f"Training set shape: {x_train.shape}, Labels: {y_train.shape}")
    print(f"Test set shape: {x_test.shape}, Labels: {y_test.shape}")

    if model_key == "1":
        model = build_mlp()
        # آموزش مدل
        model.fit(x_train, y_train, epochs=epochs)
    else:
        model = build_cnn()
        # استفاده از EarlyStopping برای جلوگیری از Overfitting
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        # آموزش مدل
        model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])

    # ارزیابی مدل روی داده‌های تست
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc}")

    # ذخیره مدل
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(info["save_path"])
    return info["save_path"]
