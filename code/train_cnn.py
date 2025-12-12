import numpy as np
import time
from data_loader import load_data
from cnn_model import SignLanguageCNN

# 1. CẤU HÌNH
DATA_DIR = "dataset_augmented"
IMG_SIZE = 64
LR = 0.005
EPOCHS = 5
LIMIT_DATA = None  # Chỉ dùng 1000 ảnh để test code trước (bỏ dòng này khi train thật)

# 2. LOAD DATA
print("🚀 Đang khởi động...")
(x_train, y_train), (x_test, y_test), classes = load_data(DATA_DIR, img_size=IMG_SIZE)

# Cắt bớt dữ liệu nếu cần test nhanh
if LIMIT_DATA:
    x_train = x_train[:LIMIT_DATA]
    y_train = y_train[:LIMIT_DATA]
    print(f"⚠️ Chế độ Test: Chỉ dùng {len(x_train)} ảnh để train.")

# 3. KHỞI TẠO MODEL
num_classes = len(classes)
print(f"🛠️ Khởi tạo CNN với {num_classes} lớp đầu ra...")
model = SignLanguageCNN(num_classes=num_classes)

# 4. VÒNG LẶP TRAINING
print("\n💪 BẮT ĐẦU HUẤN LUYỆN (64x64 RGB)...")
start_time = time.time()

for epoch in range(EPOCHS):
    print(f'--- Epoch {epoch + 1}/{EPOCHS} ---')

    # Shuffle
    perm = np.random.permutation(len(x_train))
    x_train = x_train[perm]
    y_train = y_train[perm]

    loss_sum = 0
    acc_sum = 0

    for i, (img, label) in enumerate(zip(x_train, y_train)):
        loss, acc = model.train_step(img, label, lr=LR)
        loss_sum += loss
        acc_sum += acc

        # In tiến độ mỗi 100 ảnh
        if (i + 1) % 100 == 0:
            avg_loss = loss_sum / 100
            avg_acc = (acc_sum / 100) * 100
            print(f"   [Step {i + 1}] Loss: {avg_loss:.3f} | Acc: {avg_acc:.1f}%")
            loss_sum = 0
            acc_sum = 0

    # Decay LR
    LR *= 0.8

total_time = time.time() - start_time
print(f"\n✅ Huấn luyện xong trong {total_time:.1f} giây.")

# 5. LƯU MODEL
model.save_model("sign_language_model.pkl")

# 6. TEST
print("\n🧪 Đang kiểm tra trên tập Test...")
correct = 0
# Test trên 100 ảnh thôi cho nhanh
for img, label in zip(x_test[:100], y_test[:100]):
    out = model.forward(img)
    if np.argmax(out) == label:
        correct += 1

print(f"🎯 Test Accuracy (trên 100 ảnh): {correct}%")