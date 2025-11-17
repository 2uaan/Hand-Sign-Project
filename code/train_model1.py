import cv2
import os
import numpy as np
import pickle
from collections import Counter
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


# --- 1. LỚP KNN TỰ CODE (KHÔNG DÙNG THƯ VIỆN) ---

class SimpleKNN:

    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        "Huấn luyện" mô hình đơn giản là ghi nhớ dữ liệu.
        X: Dữ liệu (các vector đặc trưng)
        y: Nhãn (labels) tương ứng
        """
        self.X_train = X
        self.y_train = y
        print("Đã 'fit' mô hình: Ghi nhớ dữ liệu huấn luyện.")

    def predict(self, X_test):
        """
        Dự đoán nhãn cho một bộ dữ liệu test.
        """
        print(f"Bắt đầu dự đoán cho {len(X_test)} mẫu thử...")
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x_test):
        """
        Dự đoán nhãn cho MỘT mẫu thử (x_test).
        """
        # 1. Tính khoảng cách Euclidean từ x_test đến TẤT CẢ các điểm trong X_train
        distances = [np.linalg.norm(x_test - x_train) for x_train in self.X_train]

        # 2. Sắp xếp và lấy chỉ số (index) của k hàng xóm gần nhất
        k_nearest_indices = np.argsort(distances)[:self.k]

        # 3. Lấy nhãn (labels) của k hàng xóm đó
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        # 4. Bỏ phiếu: Lấy nhãn xuất hiện nhiều nhất
        #    Ví dụ: [1, 0, 1] -> Counter({1: 2, 0: 1}) -> most_common(1) -> [(1, 2)] -> [0][0] -> 1
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# --- 2. CÀI ĐẶT & HÀM XỬ LÝ DỮ LIỆU ---

DATA_DIR = 'dataset'
# Kích thước ảnh HOG (phải nhất quán)
# Ảnh sẽ được resize về kích thước này TRƯỚC KHI trích xuất HOG
HOG_IMG_SIZE = (128, 128)


def load_and_process_data(data_dir):
    """
    Tải tất cả ảnh từ thư mục, tiền xử lý, trích xuất HOG và lật ảnh (augmentation).
    """
    print("Bắt đầu tải và xử lý dữ liệu...")
    data = []
    labels = []

    # Lấy danh sách các thư mục con (A, B, C...)
    try:
        labels_list = os.listdir(data_dir)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy thư mục '{data_dir}'.")
        print("Bạn đã tạo thư mục 'dataset' và thu thập ảnh chưa?")
        return None, None, None

    if not labels_list:
        print(f"LỖI: Thư mục 'dataset' bị rỗng.")
        return None, None, None

    for label in labels_list:
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue

        print(f"  Đang xử lý thư mục: {label}")
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)

            try:
                # 1. Đọc ảnh và chuyển sang ảnh xám (grayscale)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"    Cảnh báo: Không thể đọc {img_path}")
                    continue

                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 2. Resize ảnh về kích thước chuẩn HOG_IMG_SIZE
                img_resized = cv2.resize(img_gray, HOG_IMG_SIZE, interpolation=cv2.INTER_AREA)

                # --- 3. TRÍCH XUẤT HOG & TĂNG CƯỜNG DỮ LIỆU ---

                # 3a. Ảnh gốc (Tay phải)
                hog_features_original = hog(img_resized,
                                            orientations=9,
                                            pixels_per_cell=(8, 8),
                                            cells_per_block=(2, 2),
                                            visualize=False)
                data.append(hog_features_original)
                labels.append(label)

                # 3b. (QUAN TRỌNG) Ảnh lật (Tay trái) - Data Augmentation
                img_flipped = cv2.flip(img_resized, 1)  # 1 = lật ngang

                hog_features_flipped = hog(img_flipped,
                                           orientations=9,
                                           pixels_per_cell=(8, 8),
                                           cells_per_block=(2, 2),
                                           visualize=False)
                data.append(hog_features_flipped)
                labels.append(label)  # Vẫn là nhãn đó

            except Exception as e:
                print(f"    Lỗi khi xử lý {img_path}: {e}")

    print(f"Đã xử lý xong. Tổng cộng có {len(data)} mẫu (đã bao gồm lật ảnh).")

    # Chuyển đổi sang NumPy array để xử lý
    data = np.array(data)
    labels = np.array(labels)

    return data, labels


# --- 3. KHỐI CHẠY CHÍNH ---

if __name__ == "__main__":
    # Bước 1: Tải và xử lý dữ liệu
    data, labels = load_and_process_data(DATA_DIR)

    if data is not None and len(data) > 0:

        # Bước 2: Mã hóa Nhãn (Label Encoding)
        # Chuyển nhãn chữ ("A", "B", "C") thành số (0, 1, 2)
        # vì mô hình KNN chỉ hiểu số
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        print("\n--- Bắt đầu Huấn luyện ---")

        # Bước 3: Tách dữ liệu Train / Test
        # Tách 80% để huấn luyện, 20% để kiểm tra
        # stratify=labels_encoded đảm bảo cả train và test đều có đủ 10 ký tự
        X_train, X_test, y_train, y_test = train_test_split(
            data,
            labels_encoded,
            test_size=0.2,
            random_state=42,  # Để đảm bảo kết quả có thể tái lặp
            stratify=labels_encoded
        )

        print(f"Tổng số mẫu: {len(data)}")
        print(f"Số mẫu Train: {len(X_train)}")
        print(f"Số mẫu Test: {len(X_test)}")

        # Bước 4: Khởi tạo và Huấn luyện mô hình KNN "tự code"
        # Chúng ta thử với k=5 (5 hàng xóm gần nhất)
        k_value = 5
        knn_model = SimpleKNN(k=k_value)
        knn_model.fit(X_train, y_train)  # "Huấn luyện" (ghi nhớ)

        # Bước 5: Dự đoán trên tập Test
        y_pred = knn_model.predict(X_test)

        print("\n--- Đánh giá Kết quả ---")

        # Bước 6: Tính toán và In kết quả
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Độ chính xác (Accuracy) với k={k_value}: {accuracy * 100:.2f}%")

        # In báo cáo chi tiết (Precision, Recall, F1-Score)
        # Đây là phần "so sánh" mà giảng viên của bạn có thể muốn thấy
        print("\nBáo cáo Phân loại (Classification Report):")
        print(classification_report(y_test,
                                    y_pred,
                                    target_names=le.classes_))

        print("\n--- Đang lưu mô hình và LabelEncoder ---")

        # 1. Lưu mô hình KNN (chứa toàn bộ dữ liệu X_train và y_train)
        with open('knn_model.pkl', 'wb') as f:
            pickle.dump(knn_model, f)
        print("Đã lưu mô hình vào 'knn_model.pkl'")

        # 2. Lưu LabelEncoder (để chuyển 0, 1, 2... về A, B, C...)
        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(le, f)
        print("Đã lưu LabelEncoder vào 'label_encoder.pkl'")
    else:
        print("Không có dữ liệu để huấn luyện. Vui lòng kiểm tra lại thư mục 'dataset'.")