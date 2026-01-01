import numpy as np
import time
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_data
from cnn_model import SignLanguageCNN

# --- 1. CẤU HÌNH ---
DATA_DIRS = ["dataset", "dataset2"]
IMG_SIZE = 96
LR = 0.005
EPOCHS = 10
LIMIT_DATA = None


# --- HÀM TĂNG CƯỜNG DỮ LIỆU (Giữ nguyên) ---
def simple_augmentation(image):
    img_uint8 = (image * 255).astype(np.uint8)
    rows, cols, _ = img_uint8.shape

    if random.random() > 0.5:
        img_uint8 = cv2.flip(img_uint8, 1)

    if random.random() > 0.5:
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_uint8 = cv2.warpAffine(img_uint8, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    if random.random() > 0.5:
        value = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] = hsv[:, :, 2] * value
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
        hsv = np.array(hsv, dtype=np.uint8)
        img_uint8 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.1)
        new_w, new_h = int(cols * scale), int(rows * scale)
        resized = cv2.resize(img_uint8, (new_w, new_h))
        if scale > 1:
            start_x = (new_w - cols) // 2
            start_y = (new_h - rows) // 2
            img_uint8 = resized[start_y:start_y + rows, start_x:start_x + cols]
        else:
            top = (rows - new_h) // 2
            bottom = rows - new_h - top
            left = (cols - new_w) // 2
            right = cols - new_w - left
            img_uint8 = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img_uint8.astype('float32') / 255.0


# --- 2. LOAD DATA ---
print("🚀 Đang khởi động quá trình tải dữ liệu đa nguồn...")
(x_train, y_train), (x_test, y_test), classes = load_data(DATA_DIRS, img_size=IMG_SIZE)

if LIMIT_DATA:
    x_train = x_train[:LIMIT_DATA]
    y_train = y_train[:LIMIT_DATA]
    print(f"⚠️ CHẾ ĐỘ TEST: Giới hạn {LIMIT_DATA} mẫu.")

# --- 3. KHỞI TẠO MODEL ---
num_classes = len(classes)
print(f"🛠️ Khởi tạo CNN với {num_classes} lớp đầu ra...")
model = SignLanguageCNN(num_classes=num_classes, img_size=IMG_SIZE)

# --- 4. VÒNG LẶP TRAINING ---
print(f"\n💪 BẮT ĐẦU HUẤN LUYỆN (Input: {IMG_SIZE}x{IMG_SIZE})...")
start_time = time.time()

history = {'loss': [], 'accuracy': []}

for epoch in range(EPOCHS):
    print(f'--- Epoch {epoch + 1}/{EPOCHS} ---')

    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    epoch_loss = 0
    epoch_acc = 0
    steps = len(x_train)

    for i, (img, label) in enumerate(zip(x_train, y_train)):
        # aug_img = simple_augmentation(img)
        loss, acc = model.train_step(img, label, lr=LR)

        epoch_loss += loss
        epoch_acc += acc

        # --- PHẦN CHỈNH SỬA: IN CẢ ACCURACY ---
        if (i + 1) % 100 == 0:
            # Tính trung bình tạm thời đến bước hiện tại
            current_mean_loss = epoch_loss / (i + 1)
            current_mean_acc = (epoch_acc / (i + 1)) * 100

            # In ra Loss và Acc cùng lúc
            print(f"\r   Step {i + 1}/{steps} | Loss: {current_mean_loss:.3f} | Acc: {current_mean_acc:.1f}%")

        # Tính trung bình toàn Epoch (để lưu vào history)
    avg_loss = epoch_loss / steps
    avg_acc = (epoch_acc / steps) * 100

    history['loss'].append(avg_loss)
    history['accuracy'].append(avg_acc)

    print(f"\n   ✅ Epoch Result: Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2f}%")
    LR *= 0.9

total_time = time.time() - start_time
print(f"\n✅ Huấn luyện xong trong {total_time:.1f} giây.")

# --- 5. LƯU MODEL ---
model.save_model("sign_language_model.pkl")

# ==========================================
# 6. ĐÁNH GIÁ & VẼ BIỂU ĐỒ
# ==========================================
print("\n📊 Đang tạo báo cáo đánh giá...")

# --- A. Vẽ biểu đồ Loss & Accuracy ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), history['loss'], marker='o', color='red', label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), history['accuracy'], marker='o', color='blue', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# --- B. Dự đoán trên tập Test ---
print("🧪 Đang chạy dự đoán trên tập Test...")
y_pred = []
test_limit = 1000
x_test_eval = x_test[:test_limit]
y_test_eval = y_test[:test_limit]

for img in x_test_eval:
    out = model.forward(img)
    y_pred.append(np.argmax(out))

y_pred = np.array(y_pred)

# --- C. Báo cáo chi tiết ---
print("\n📝 Báo cáo hiệu suất chi tiết:")
print(classification_report(y_test_eval, y_pred, target_names=classes))

# --- D. Confusion Matrix (Dùng Scikit-learn Display) ---
print("heatmap Đang vẽ Confusion Matrix...")
cm = confusion_matrix(y_test_eval, y_pred)

# Thay thế Seaborn bằng ConfusionMatrixDisplay của sklearn
plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Blues', xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.show()

# --- E. Hiển thị dự đoán ngẫu nhiên ---
print("\n🎲 Kết quả dự đoán ngẫu nhiên (15 ảnh):")
plt.figure(figsize=(15, 8))
indices = np.random.choice(len(x_test), 15, replace=False)

for i, idx in enumerate(indices):
    img = x_test[idx]
    true_label = classes[y_test[idx]]

    out = model.forward(img)
    pred_idx = np.argmax(out)
    pred_label = classes[pred_idx]
    prob = np.max(out)

    color = 'green' if pred_label == true_label else 'red'

    plt.subplot(3, 5, i + 1)
    plt.imshow(img)
    plt.title(f"True: {true_label}\nPred: {pred_label}\n({prob:.2f})", color=color, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

print("\n🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")