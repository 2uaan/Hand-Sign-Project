import cv2
import pickle
import numpy as np
from collections import Counter
from skimage.feature import hog


# --- 1. ĐỊNH NGHĨA LẠI LỚP SimpleKNN ---
# (CẦN THIẾT để pickle có thể 'hiểu' file đã lưu)
class SimpleKNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x_test):
        distances = [np.linalg.norm(x_test - x_train) for x_train in self.X_train]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# --- 2. CÀI ĐẶT CÁC THAM SỐ (PHẢI GIỐNG HỆT KHI TRAIN) ---

# Tọa độ ROI (y hệt data_collector.py)
x_roi, y_roi = 100, 100
w_roi, h_roi = 300, 300
IMG_PADDING = 20

# Cài đặt HOG (y hệt train_and_evaluate.py)
HOG_IMG_SIZE = (128, 128)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# --- 3. TẢI MÔ HÌNH VÀ LABEL ENCODER ---

print("Đang tải mô hình KNN...")
try:
    with open('knn_model.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    print("Tải 'knn_model.pkl' thành công.")
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'knn_model.pkl'.")
    print("Bạn đã chạy lại file 'train_and_evaluate.py' để lưu mô hình chưa?")
    exit()

print("Đang tải LabelEncoder...")
try:
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    print("Tải 'label_encoder.pkl' thành công.")

    print("CÁC KÝ TỰ MÔ HÌNH ĐÃ HỌC:", le.classes_)
    print("TỔNG SỐ KÝ TỰ:", len(le.classes_))
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'label_encoder.pkl'.")
    print("Bạn đã chạy lại file 'train_and_evaluate.py' để lưu mô hình chưa?")
    exit()

print("\n--- Bắt đầu Webcam ---")
print("Nhấn 'q' để thoát.")

# --- 4. KHỞI ĐỘNG WEBCAM ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở webcam!")
    exit()

# Biến lưu trữ dự đoán cuối cùng để hiển thị (cho mượt)
current_prediction = "?"

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # Tính toán tọa độ hộp ROI (x1, y1, x2, y2)
    x1 = x_roi - IMG_PADDING
    y1 = y_roi - IMG_PADDING
    x2 = x_roi + w_roi + IMG_PADDING
    y2 = y_roi + h_roi + IMG_PADDING

    try:
        # --- 5. XỬ LÝ ẢNH TRONG ROI ---

        # Cắt ảnh từ frame gốc
        img_crop = frame[y1:y2, x1:x2]

        # Chuyển sang ảnh xám và resize
        img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, HOG_IMG_SIZE, interpolation=cv2.INTER_AREA)

        # --- 6. TRÍCH XUẤT HOG VÀ DỰ ĐOÁN ---

        # Trích xuất HOG
        hog_features = hog(img_resized,
                           orientations=HOG_ORIENTATIONS,
                           pixels_per_cell=HOG_PIXELS_PER_CELL,
                           cells_per_block=HOG_CELLS_PER_BLOCK,
                           visualize=False)

        # Reshape lại để đưa vào mô hình (1 mẫu)
        hog_features = hog_features.reshape(1, -1)

        # Dự đoán bằng KNN (trả về số, ví dụ: [0])
        prediction_encoded = knn_model.predict(hog_features)

        # Chuyển số về chữ (ví dụ: [0] -> ['A'])
        prediction_label = le.inverse_transform(prediction_encoded)

        # Cập nhật dự đoán
        current_prediction = prediction_label[0]

    except Exception as e:
        # Nếu có lỗi (ví dụ: tay ra ngoài khung), hiện "?"
        current_prediction = "?"
        # print(f"Lỗi xử lý frame: {e}") # Bật dòng này nếu muốn debug

    # --- 7. HIỂN THỊ KẾT QUẢ ---

    # Vẽ hộp ROI màu xanh
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Hiển thị dự đoán
    cv2.putText(display_frame, f"KY HIEU: {current_prediction}",
                (x_roi - IMG_PADDING, y_roi - IMG_PADDING - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 2)

    cv2.imshow("Live Sign Language Recognition", display_frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()