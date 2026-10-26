import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model  # type: ignore

# بارگذاری مدل
model = load_model("my_model.h5")

# خواندن تصویر سیاه و سفید
image_path = "my_digit_3_marjan.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# اعمال فیلتر بلور برای کاهش نویز
image = cv2.GaussianBlur(image, (5, 5), 0)

# معکوس کردن رنگ در صورت نیاز (پس‌زمینه‌ی روشن)
if np.mean(image) > 127:
    image = cv2.bitwise_not(image)

# آستانه‌گذاری برای جلوگیری از رنگ‌های طوسی
_, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

# پیدا کردن کانتورها
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# اگر کانتوری پیدا شد، آن را برش می‌دهیم
if contours:
    # از بزرگترین کانتور استفاده می‌کنیم
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    image_cropped = image[y:y+h, x:x+w]
else:
    image_cropped = image  # اگر کانتور پیدا نشد، همان تصویر اصلی را استفاده کن

# نمایش تصویر کانتور و برش خورده
image_with_contours = cv2.drawContours(image.copy(), [contour], -1, (0, 255, 0), 2)

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
image_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_resized

# نمایش تصویر پردازش‌شده
plt.imshow(image_padded, cmap="gray")
plt.axis("off")
plt.show()

# نرمال‌سازی تصویر
image_final = image_padded / 255.0
image_final = image_final.reshape(1, 28, 28)

# پیش‌بینی کلاس
predictions = model.predict(image_final)
predicted_class = np.argmax(predictions)

print(f"Predicted class: {predicted_class}, with probability: {predictions[0][predicted_class]:.4f}")















# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model  # type: ignore

# # بارگذاری مدل
# model = load_model("my_model.h5")

# # خواندن تصویر
# image_path = "my_digit_6.jpg"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # **1️⃣ حذف نویز با GaussianBlur**
# image_blurred = cv2.GaussianBlur(image, (5,5), 0)

# # **2️⃣ آستانه‌گذاری دستی برای جدا کردن پس‌زمینه**
# _, thresh = cv2.threshold(image_blurred, 127, 255, cv2.THRESH_BINARY_INV)

# kernel = np.ones((2,2), np.uint8)
# thresh = cv2.dilate(thresh, kernel, iterations=1)


# # **3️⃣ پیدا کردن کانتورها**
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # **4️⃣ پیدا کردن مستطیل حاوی عدد**
# x, y, w, h = cv2.boundingRect(contours[0])

# # **5️⃣ برش تصویر برای حذف فضای اضافی**
# image_cropped = thresh[y:y+h, x:x+w]

# # **6️⃣ تغییر اندازه به ۲۸x۲۸ با حفظ نسبت ابعاد**
# aspect_ratio = w / h
# if aspect_ratio > 1:  
#     new_w = 20
#     new_h = int(20 / aspect_ratio)
# else:  
#     new_h = 20
#     new_w = int(20 * aspect_ratio)

# image_resized = cv2.resize(image_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

# # **7️⃣ ایجاد تصویر ۲۸x۲۸ با پس‌زمینه مشکی**
# image_padded = np.zeros((28, 28), dtype=np.uint8)

# # **8️⃣ چسباندن عدد وسط تصویر**
# x_offset = (28 - new_w) // 2
# y_offset = (28 - new_h) // 2
# image_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_resized

# # **نمایش تصویر پردازش‌شده**
# plt.imshow(image_padded, cmap="gray")
# plt.title("Processed Image")
# plt.axis("off")
# plt.show()

# # **9️⃣ نرمال‌سازی تصویر**
# image_final = image_padded / 255.0
# image_final = image_final.reshape(1, 28, 28)

# # **🔟 پیش‌بینی عدد**
# predictions = model.predict(image_final)
# predicted_class = np.argmax(predictions)

# print(f"Predicted class: {predicted_class}, with probability: {predictions[0][predicted_class]}")







# //////////////////////////////////////////////////////////////////////////////////////////////
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model  # type: ignore

# # بارگذاری مدل
# model = load_model("my_model.h5")

# # خواندن تصویر
# image_path = "my_digit.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # اگر پس‌زمینه روشن است، آن را معکوس می‌کنیم (تصویر روشن و عدد تیره است)
# if np.mean(image) > 127:  # اگر تصویر روشن باشد
#     image = cv2.bitwise_not(image)

# # آستانه‌گذاری برای جداسازی تصویر از پس‌زمینه
# _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# # پیدا کردن کانتورها
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # پیدا کردن مستطیل حاوی عدد
# x, y, w, h = cv2.boundingRect(contours[0])

# # برش تصویر برای حذف فضای اضافی
# image_cropped = image[y:y+h, x:x+w]

# # تغییر اندازه به ۲۸x۲۸ با حفظ نسبت ابعاد
# aspect_ratio = w / h
# if aspect_ratio > 1:  # اگر تصویر عریض‌تر است
#     new_w = 20
#     new_h = int(20 / aspect_ratio)
# else:  # اگر تصویر بلندتر است
#     new_h = 20
#     new_w = int(20 * aspect_ratio)

# image_resized = cv2.resize(image_cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

# # ایجاد تصویر ۲۸x۲۸ با پس‌زمینه سیاه
# image_padded = np.zeros((28, 28), dtype=np.uint8)

# # چسباندن تصویر وسط پس‌زمینه
# x_offset = (28 - new_w) // 2
# y_offset = (28 - new_h) // 2
# image_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = image_resized

# # نمایش تصویر نهایی
# plt.imshow(image_padded, cmap="gray")
# plt.title("Processed Image")
# plt.axis("off")
# plt.show()

# # نرمال‌سازی تصویر (مقادیر به [0, 1] تبدیل می‌شوند)
# image_final = image_padded / 255.0
# image_final = image_final.reshape(1, 28, 28)

# # پیش‌بینی کلاس
# predictions = model.predict(image_final)
# predicted_class = np.argmax(predictions)

# # نمایش پیش‌بینی
# print(f"Predicted class: {predicted_class}, with probability: {predictions[0][predicted_class]}")
# //////////////////////////////////////////////////////////////////////////////////////////////









# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model  # type: ignore

# # بارگذاری مدل
# model = load_model("my_model.h5")

# # خواندن تصویر
# image_path = "my_digit-1.png"
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# # اگر پس‌زمینه روشن است، آن را معکوس می‌کنیم (تصویر روشن و عدد تیره است)
# if np.mean(image) > 127:  # اگر تصویر روشن باشد
#     image = cv2.bitwise_not(image)

# # افزایش کنتراست و روشنایی به‌طور همزمان
# alpha = 1.8  # افزایش کنتراست
# beta = 0    # افزایش روشنایی
# image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# # اعمال فیلتر sharpen برای افزایش وضوح تصویر
# kernel = np.array([[0, -1, 0],
#                    [-1, 5,-1],
#                    [0, -1, 0]])
# image = cv2.filter2D(image, -1, kernel)

# # آستانه‌گذاری برای جداسازی تصویر از پس‌زمینه
# _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# # پیدا کردن کانتورها
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # پیدا کردن مستطیل حاوی عدد
# x, y, w, h = cv2.boundingRect(contours[0])

# # برش تصویر برای حذف فضای اضافی
# image_cropped = image[y:y+h, x:x+w]

# # تغییر اندازه به ۲۰x۲۰ با حفظ نسبت ابعاد
# image_resized = cv2.resize(image_cropped, (20, 20), interpolation=cv2.INTER_AREA)

# # ایجاد تصویر ۲۸x۲۸ با پس‌زمینه سیاه
# image_padded = np.zeros((28, 28), dtype=np.uint8)

# # چسباندن تصویر وسط پس‌زمینه
# x_offset = (28 - 20) // 2
# y_offset = (28 - 20) // 2
# image_padded[y_offset:y_offset+20, x_offset:x_offset+20] = image_resized

# # نمایش تصویر نهایی
# plt.imshow(image_padded, cmap="gray")
# plt.title("Processed Image")
# plt.axis("off")
# plt.show()

# # نرمال‌سازی تصویر (مقادیر به [0, 1] تبدیل می‌شوند)
# image_final = image_padded / 255.0
# image_final = image_final.reshape(1, 28, 28)

# # پیش‌بینی کلاس
# predictions = model.predict(image_final)
# predicted_class = np.argmax(predictions)

# # نمایش پیش‌بینی
# print(f"Predicted class: {predicted_class}, with probability: {predictions[0][predicted_class]}")








# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model # type: ignore

# model = load_model("my_model.h5")

# # image = cv2.imread("handwritten_digit.png" , cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("my_digit.png" , cv2.IMREAD_GRAYSCALE)

# image = cv2.resize(image, (28, 28))

# image = image / 255.0

# image = cv2.bitwise_not(image)
# image = image.reshape(1, 28, 28)

# predictions = model.predict(image)
# predicted_class = np.argmax(predictions)
# print(f"Predicted class: {np.argmax(predictions)}")

