import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_data
import os

# --- 1. CẤU HÌNH ---
DATA_DIRS = ["dataset", "dataset2"]
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001


def main():
    # --- 2. LOAD DỮ LIỆU ---
    print("🚀 Đang tải dữ liệu...")
    (x_train, y_train), (x_test, y_test), classes = load_data(DATA_DIRS, img_size=IMG_SIZE)

    num_classes = len(classes)
    print(f"✅ Đã tải: {len(x_train)} train, {len(x_test)} test. Số lớp: {num_classes}")

    # --- 3. XÂY DỰNG MODEL (TRANSFER LEARNING) ---
    print("🛠️ Đang tải MobileNetV2 (ImageNet weights)...")

    # Base model (bỏ phần đầu)
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                             include_top=False,
                             weights='imagenet')
    base_model.trainable = False  # Đóng băng

    # Custom Head
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(optimizer=Adam(learning_rate=LR),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # --- 4. HUẤN LUYỆN ---
    print("\n💪 Bắt đầu train MobileNetV2...")
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(x_test, y_test))

    # Lưu model
    model.save("mobilenet_sign_language.h5")
    print("💾 Đã lưu model vào 'mobilenet_sign_language.h5'")

    # ==========================================
    # 5. VẼ BIỂU ĐỒ & ĐÁNH GIÁ (FULL)
    # ==========================================

    # --- A. Biểu đồ Loss & Accuracy ---
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("mobilenet_history.png")  # Lưu ảnh lại
    plt.show()

    # Chuẩn bị dữ liệu dự đoán
    print("🧪 Đang đánh giá trên tập Test...")
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # --- B. Báo cáo chi tiết (Classification Report) ---
    print("\n📝 Báo cáo hiệu suất chi tiết (MobileNetV2):")
    report = classification_report(y_test, y_pred, target_names=classes)
    print(report)

    # Lưu ra file text
    with open("mobilenet_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # --- C. Confusion Matrix ---
    print("📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', xticks_rotation='vertical', values_format='d')
    plt.title('Confusion Matrix - MobileNetV2')
    plt.savefig("mobilenet_confusion_matrix.png")
    plt.show()

    # --- D. Dự đoán 15 ảnh ngẫu nhiên (Visualize Predictions) ---
    print("🎲 Kết quả dự đoán ngẫu nhiên (15 ảnh):")
    plt.figure(figsize=(15, 8))

    # Chọn ngẫu nhiên 15 chỉ số
    indices = np.random.choice(len(x_test), 15, replace=False)

    for i, idx in enumerate(indices):
        img = x_test[idx]
        true_label = classes[y_test[idx]]

        # Lấy kết quả dự đoán (đã tính ở trên)
        pred_label = classes[y_pred[idx]]
        prob = np.max(y_pred_probs[idx])

        # Màu tiêu đề: Xanh nếu đúng, Đỏ nếu sai
        color = 'green' if pred_label == true_label else 'red'

        plt.subplot(3, 5, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}\n({prob:.2f})", color=color, fontsize=10)
        plt.axis('off')

    plt.suptitle("MobileNetV2 Random Predictions", fontsize=16)
    plt.tight_layout()
    plt.savefig("mobilenet_random_predictions.png")
    plt.show()

    print("\n🎉 ĐÃ HOÀN TẤT TOÀN BỘ QUÁ TRÌNH CHO MOBILENET!")


if __name__ == "__main__":
    main()