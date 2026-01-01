import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_dirs, img_size=64, test_ratio=0.2):
    """
    Hàm đọc dữ liệu từ NHIỀU thư mục.
    Tham số:
        data_dirs: List các đường dẫn (VD: ["dataset", "dataset2"])
        img_size: Kích thước resize
        test_ratio: Tỉ lệ chia tập test
    """

    # Nếu người dùng truyền vào 1 string đơn lẻ, tự chuyển thành list
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    images = []
    labels = []

    # BƯỚC 1: QUÉT TOÀN BỘ CÁC LỚP (CLASSES) TỪ TẤT CẢ THƯ MỤC
    # Để đảm bảo nếu dataset2 thiếu chữ cái nào đó thì index vẫn đúng
    all_classes = set()
    for d_dir in data_dirs:
        if not os.path.exists(d_dir):
            print(f"⚠️ Cảnh báo: Không tìm thấy thư mục '{d_dir}'")
            continue

        classes_in_dir = [d for d in os.listdir(d_dir)
                          if os.path.isdir(os.path.join(d_dir, d)) and d != 'temp']
        all_classes.update(classes_in_dir)

    # Sắp xếp để đảm bảo thứ tự nhất quán (A=0, B=1, ...)
    sorted_classes = sorted(list(all_classes))

    if not sorted_classes:
        print("❌ Lỗi: Không tìm thấy lớp dữ liệu nào!")
        return None, None, []

    print(f"📂 Tìm thấy {len(sorted_classes)} lớp dữ liệu: {sorted_classes}")
    print(f"🔄 Đang tổng hợp dữ liệu từ: {data_dirs}...")

    total_count = 0

    # BƯỚC 2: DUYỆT QUA TỪNG THƯ MỤC NGUỒN
    for d_dir in data_dirs:
        if not os.path.exists(d_dir): continue

        print(f"   ↳ Đang đọc thư mục: '{d_dir}'")

        for label_idx, class_name in enumerate(sorted_classes):
            class_path = os.path.join(d_dir, class_name)

            # Nếu thư mục này không chứa lớp đó (ví dụ dataset2 thiếu chữ Z) thì bỏ qua
            if not os.path.exists(class_path):
                continue

            files = os.listdir(class_path)
            # print(f"      - Lớp '{class_name}': {len(files)} ảnh") # Bỏ comment nếu muốn xem chi tiết

            for file_name in files:
                try:
                    img_path = os.path.join(class_path, file_name)

                    # Đọc ảnh
                    img = cv2.imread(img_path)
                    if img is None: continue

                    # Chuyển RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Resize
                    img = cv2.resize(img, (img_size, img_size))

                    images.append(img)
                    labels.append(label_idx)
                    total_count += 1

                except Exception as e:
                    print(f"Lỗi ảnh {file_name}: {e}")

    print(f"✅ Đã tải TỔNG CỘNG {total_count} ảnh từ tất cả nguồn.")

    # BƯỚC 3: XỬ LÝ MẢNG NUMPY & SHUFFLE
    X = np.array(images).astype('float32') / 255.0
    y = np.array(labels)

    # Trộn ngẫu nhiên dữ liệu (Trộn lẫn dataset1 và dataset2 với nhau)
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]

    # BƯỚC 4: CHIA TRAIN/TEST
    split_index = int(len(X) * (1 - test_ratio))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"📊 Dữ liệu Train: {len(X_train)} ảnh | Dữ liệu Test: {len(X_test)} ảnh")

    return (X_train, y_train), (X_test, y_test), sorted_classes


if __name__ == "__main__":
    # Test thử
    dirs = ["dataset", "dataset2"]  # Thử nghiệm đọc 2 thư mục
    (x_train, y_train), _, classes = load_data(dirs, img_size=64)
    if x_train is not None:
        print("Shape X_train:", x_train.shape)