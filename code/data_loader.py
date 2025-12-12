import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_dir, img_size=64, test_ratio=0.2):
    """
    Hàm đọc dữ liệu từ thư mục dataset của bạn.
    - Tự động bỏ qua thư mục 'temp'.
    - Resize ảnh về 32x32 (để NumPy chạy nổi).
    - Chia tập Train/Test thủ công.
    """
    images = []
    labels = []

    # Lấy danh sách các lớp, loại bỏ 'temp' và các file ẩn
    classes = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d)) and d != 'temp']
    classes.sort()  # Sắp xếp A, B, C... để nhãn thống nhất (A=0, B=1...)

    print(f"📂 Tìm thấy {len(classes)} lớp cần học: {classes}")

    total_count = 0

    for label_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)

        # Đếm số ảnh trong thư mục để in log
        files = os.listdir(class_path)
        print(f"   --> Đang đọc lớp '{class_name}': {len(files)} ảnh...")

        for file_name in files:
            try:
                img_path = os.path.join(class_path, file_name)

                # 1. Đọc ảnh
                img = cv2.imread(img_path)
                if img is None: continue

                # 2. Chuyển Grayscale (Ảnh xám)
                # Lý do: Ảnh màu (3 kênh) sẽ làm tăng gấp 3 khối lượng tính toán.
                # Với shape tay, ảnh xám là đủ để nhận diện.
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 3. Resize (Quan trọng nhất)
                # 32x32 là kích thước vàng cho bài toán "NumPy from Scratch"
                img = cv2.resize(img, (img_size, img_size))

                images.append(img)
                labels.append(label_idx)
                total_count += 1

            except Exception as e:
                print(f"Lỗi ảnh {file_name}: {e}")

    print(f"✅ Đã tải tổng cộng {total_count} ảnh.")

    # 4. Chuẩn hóa dữ liệu & Shuffle
    X = np.array(images).astype('float32') / 255.0  # Về khoảng [0, 1]
    y = np.array(labels)

    # Xáo trộn ngẫu nhiên (Rất quan trọng để model không học vẹt theo thứ tự)
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]

    # 5. Chia Train/Test (Thủ công)
    split_index = int(len(X) * (1 - test_ratio))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"📊 Chia dữ liệu: Train ({len(X_train)}), Test ({len(X_test)})")

    return (X_train, y_train), (X_test, y_test), classes


# --- HÀM KIỂM TRA DỮ LIỆU ---
def visualize_sample(X, y, classes):
    plt.figure(figsize=(10, 5))
    for i in range(10):  # Vẽ 10 ảnh ngẫu nhiên
        idx = np.random.randint(0, len(X))
        plt.subplot(2, 5, i + 1)
        plt.imshow(X[idx], cmap='gray')
        plt.title(f"Label: {classes[y[idx]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Đổi đường dẫn này trỏ tới thư mục dataset của bạn
    # Ví dụ: "C:/Users/tlmqu/PycharmProjects/data_collector/dataset"
    DATA_DIR = "dataset"

    if os.path.exists(DATA_DIR):
        (x_train, y_train), (x_test, y_test), class_names = load_data(DATA_DIR, img_size=64)

        # Hiển thị thử để chắc chắn data đọc đúng
        visualize_sample(x_train, y_train, class_names)

        # In shape để kiểm tra kích thước ma trận
        # Kỳ vọng: (Số lượng ảnh, 32, 32)
        print("Shape X_train:", x_train.shape)
    else:
        print("❌ Không tìm thấy thư mục dataset!")