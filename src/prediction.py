"""پیش‌بینی رقم دست‌نویس از روی تصاویر.

پیش‌پردازش تصویر از اسکریپت اصلی checkHandriteNum.py منتقل شده است؛
مراحل پردازش و روش پیش‌بینی دقیقاً مثل قبل حفظ شده‌اند.
"""

from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "images"
MODELS_DIR = PROJECT_ROOT / "models"

# فرمت‌های تصویری پشتیبانی‌شده
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

# مدل‌های آموزش‌دیده‌ای که این اپلیکیشن می‌تواند برای پیش‌بینی استفاده کند
KNOWN_MODELS = [
    ("MLP (baseline)", MODELS_DIR / "mnist_model.h5"),
    ("CNN", MODELS_DIR / "my_model.h5"),
    ("CNN (variant 2)", MODELS_DIR / "my_model_2.h5"),
]


def available_models():
    """فهرست مدل‌های آموزش‌دیده‌ای که فایلشان وجود دارد"""

    return [(label, path) for label, path in KNOWN_MODELS if path.is_file()]


def list_images(images_dir=IMAGES_DIR):
    """همه‌ی تصاویر پشتیبانی‌شده‌ی داخل پوشه (به‌ترتیب الفبا، بدون زیرپوشه‌ها)"""

    if not images_dir.is_dir():
        return []
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def preprocess_digit_image(image_path):
    """پیش‌پردازش یک تصویر: حذف نویز، معکوس کردن در صورت نیاز، آستانه‌گذاری،
    برش با کانتور و وسط‌چین کردن عدد روی بوم ۲۸x۲۸"""

    # خواندن تصویر سیاه و سفید
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("could not read image")

    # اعمال فیلتر بلور برای کاهش نویز
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # معکوس کردن رنگ در صورت نیاز (پس‌زمینه‌ی روشن)
    if np.mean(image) > 127:
        image = cv2.bitwise_not(image)

    # آستانه‌گذاری برای جلوگیری از رنگ‌های طوسی
    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

    # پیدا کردن کانتورها
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("no digit contour found in the image")

    # از بزرگترین کانتور استفاده می‌کنیم و آن را برش می‌دهیم
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    image_cropped = image[y:y + h, x:x + w]

    # تغییر اندازه به ۲۸x۲۸ با حفظ نسبت ابعاد
    aspect_ratio = w / h
    if aspect_ratio > 1:  # اگر تصویر عریض‌تر است
        new_w = 20
        new_h = int(20 / aspect_ratio)
    else:  # اگر تصویر بلندتر است
        new_h = 20
        new_w = int(20 * aspect_ratio)

    image_resized = cv2.resize(image_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ایجاد تصویر ۲۸x۲۸ و وسط‌چین کردن عدد
    image_padded = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    image_padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = image_resized

    # نرمال‌سازی تصویر
    image_final = image_padded / 255.0
    image_final = image_final.reshape(1, 28, 28)

    return image_final


def predict_digit(model, image_path):
    """پیش‌بینی رقم یک تصویر با مدل داده‌شده

    خروجی: (predicted_digit, probability)
    """

    image_final = preprocess_digit_image(image_path)

    # پیش‌بینی کلاس
    predictions = model.predict(image_final, verbose=0)
    predicted_class = int(np.argmax(predictions))

    return predicted_class, float(predictions[0][predicted_class])


def predict_images(model, image_files):
    """پیش‌بینی همه‌ی تصاویر داده‌شده و چاپ خلاصه‌ی نتایج

    خروجی: (تعداد موفق، فهرست فایل‌های ناموفق)
    """

    print()
    print("Predictions")
    print("-------------------------")

    succeeded = 0
    failed = []
    for image_path in image_files:
        try:
            predicted_digit, probability = predict_digit(model, image_path)
        except Exception as error:
            print(f"{image_path.name} -> skipped ({error})")
            failed.append(image_path.name)
            continue
        print(f"{image_path.name} -> {predicted_digit} (confidence {probability:.4f})")
        succeeded += 1

    print()
    print(f"Successfully processed: {succeeded}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed files: {', '.join(failed)}")

    return succeeded, failed
