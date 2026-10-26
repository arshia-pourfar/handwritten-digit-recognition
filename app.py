"""Handwritten Digit Recognition — main entry point.

Run:  python app.py
"""

import sys
from pathlib import Path

# اضافه کردن ریشه‌ی پروژه به sys.path تا ایمپورت‌ها از هر پوشه‌ای کار کنند
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# جلوگیری از خطای کدنوگذاری کاراکترهای خاص (نوار پیشرفت TensorFlow) در کنسول ویندوز
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def print_main_menu():
    print()
    print("=================================")
    print("      Computer Vision Project")
    print("=================================")
    print()
    print("1. Learn / Train a Model")
    print("2. Predict Digits from Images")
    print("3. Exit")


def learn_menu():
    """گزینه ۱: آموزش یکی از مدل‌های پشتیبانی‌شده"""

    # ایمپورت سنگین فقط وقتی لازم است، تا منو سریع نمایش داده شود
    from src import training

    print()
    print("Learn / Train a Model")
    print("-------------------------")
    for key, info in training.SUPPORTED_MODELS.items():
        print(f"{key}. {info['name']}")
        print(f"   {info['description']}")
        print(f"   Default epochs: {info['default_epochs']}, "
              f"saves to: models/{info['save_path'].name}")
    print("B. Back to main menu")

    choice = input("\nSelect a model: ").strip()
    if choice.lower() == "b":
        return

    info = training.SUPPORTED_MODELS.get(choice)
    if info is None:
        print("Invalid choice.")
        return

    # تعداد اپوک‌ها با مقدار پیش‌فرض
    raw = input(f"Number of epochs [Enter = {info['default_epochs']}]: ").strip()
    if raw == "":
        epochs = info["default_epochs"]
    else:
        try:
            epochs = int(raw)
            if epochs < 1:
                raise ValueError
        except ValueError:
            print("Invalid number of epochs. Please enter a positive whole number.")
            return

    print()
    print(f"Starting training: {info['name']} for {epochs} epoch(s)...")
    print("(training progress is printed below)")

    try:
        save_path = training.train_model(choice, epochs)
    except FileNotFoundError as error:
        print()
        print(f"Cannot train: {error}")
        return
    except Exception as error:
        print()
        print(f"Training failed: {error}")
        return

    print()
    print(f"Done. Model saved to: {save_path.relative_to(PROJECT_ROOT)}")
    print("You can now use option 2 (Predict) from the main menu.")


def predict_menu():
    """گزینه ۲: پیش‌بینی تصاویر داخل پوشه‌ی images"""

    from src import prediction

    print()
    print("Predict Digits from Images")
    print("-------------------------")

    # بررسی پوشه‌ی تصاویر
    if not prediction.IMAGES_DIR.is_dir():
        print(f"Images folder not found: {prediction.IMAGES_DIR}")
        print("Create a folder named 'images' in the project root and put your "
              "digit images (.png, .jpg, .jpeg, .bmp) inside it.")
        return

    image_files = prediction.list_images()
    if not image_files:
        print(f"No supported images found in: {prediction.IMAGES_DIR}")
        print("Add one or more digit images (.png, .jpg, .jpeg, .bmp) "
              "and run this option again.")
        return

    # انتخاب مدل
    available = prediction.available_models()
    if not available:
        print("No trained model found.")
        print("Train a model first with option 1 (Learn / Train), or copy a compatible "
              "model file (.h5) into the 'models' folder:")
        for label, path in prediction.KNOWN_MODELS:
            print(f"    models/{path.name}  ({label})")
        return

    if len(available) == 1:
        label, model_path = available[0]
        print(f"Using model: {label} (models/{model_path.name})")
    else:
        print("Available models:")
        for index, (label, path) in enumerate(available, start=1):
            print(f"{index}. {label} (models/{path.name})")
        raw = input("Select a model: ").strip()
        try:
            index = int(raw)
            if index < 1 or index > len(available):
                raise ValueError
            label, model_path = available[index - 1]
        except (ValueError, IndexError):
            print("Invalid choice.")
            return
        print(f"Using model: {label} (models/{model_path.name})")

    # بارگذاری مدل و پیش‌بینی
    try:
        from tensorflow.keras.models import load_model  # type: ignore
        model = load_model(model_path)
    except Exception as error:
        print(f"Could not load model {model_path.name}: {error}")
        return

    print(f"Found {len(image_files)} image(s) in images/")
    prediction.predict_images(model, image_files)


def main():
    while True:
        print_main_menu()
        try:
            choice = input("\nChoose an option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Goodbye.")
            return

        if choice == "1":
            try:
                learn_menu()
            except (EOFError, KeyboardInterrupt):
                print()
                print("Returning to the main menu.")
        elif choice == "2":
            try:
                predict_menu()
            except (EOFError, KeyboardInterrupt):
                print()
                print("Returning to the main menu.")
        elif choice == "3":
            print("Goodbye.")
            return
        else:
            print("Invalid option. Please enter 1, 2 or 3.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print()
        print("Goodbye.")
