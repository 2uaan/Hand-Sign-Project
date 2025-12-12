import cv2
import numpy as np
import time
from cnn_model import SignLanguageCNN

# 1. CẤU HÌNH
MODEL_PATH = 'sign_language_model.pkl'
IMG_SIZE = 64

# Danh sách nhãn (Phải khớp đúng thứ tự lúc train)
# Bạn kiểm tra lại log lúc train xem thứ tự classes là gì nhé
# Ví dụ: Dựa trên thư mục dataset của bạn
CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L']


def extract_skin(image):
    """
    Hàm lọc lấy vùng da người, biến nền thành màu đen.
    Input: Ảnh RGB (hoặc BGR từ OpenCV)
    Output: Ảnh chỉ còn tay trên nền đen
    """
    # 1. Chuyển sang hệ màu HSV (Hue, Saturation, Value)
    # HSV tách biệt màu sắc (Hue) khỏi độ sáng (Value), giúp lọc màu da tốt hơn RGB
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. Định nghĩa khoảng màu da (Cần tinh chỉnh tùy ánh sáng)
    # Đây là khoảng màu da phổ biến
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # 3. Tạo mặt nạ (Mask): Chỗ nào là da thì = 1 (Trắng), nền = 0 (Đen)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 4. Lọc nhiễu (Morphological Operations)
    # Dùng thuật toán "Mở" (Open) để xóa các đốm trắng nhỏ li ti (nhiễu)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Dùng thuật toán "Giãn" (Dilate) để làm liền các vết nứt trên tay
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 5. Ghép mặt nạ vào ảnh gốc
    # Chỗ nào mask đen thì ảnh gốc thành đen, mask trắng giữ nguyên màu
    skin_only = cv2.bitwise_and(image, image, mask=mask)

    return skin_only

def main():
    # 2. LOAD MODEL
    print("⏳ Đang tải mô hình...")
    # Khởi tạo lại kiến trúc y hệt lúc train
    model = SignLanguageCNN(num_classes=len(CLASSES))
    try:
        model.load_model(MODEL_PATH)
        print("✅ Mô hình đã sẵn sàng!")
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file '{MODEL_PATH}'. Hãy chạy train.py trước!")
        return

    # 3. MỞ CAMERA
    cap = cv2.VideoCapture(0)  # 0 là camera mặc định

    # Cài đặt khung chữ nhật (ROI - Region of Interest) để đặt tay vào
    # Tọa độ góc trên bên phải (để thuận tay phải)
    x1, y1 = 300, 50
    x2, y2 = 600, 350

    print("🎥 Nhấn 'q' để thoát chương trình.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Lật ngược ảnh cho giống gương (Mirror)
        frame = cv2.flip(frame, 1)

        # Copy frame để vẽ vời mà không ảnh hưởng ảnh gốc
        display_frame = frame.copy()

        # 4. TRÍCH XUẤT VÙNG ẢNH TAY (ROI)
        # Cắt vùng trong khung xanh
        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            # --- TIỀN XỬ LÝ (GIỐNG HỆT LÚC TRAIN) ---
            # 1. Resize về 64x64
            roi_skin = extract_skin(roi)

            # Hiển thị thử cái roi_skin này xem lọc sạch không
            #cv2.imshow("Skin Detection", roi_skin)

            # Sau đó mới resize và đưa vào model
            roi_resized = cv2.resize(roi_skin, (IMG_SIZE, IMG_SIZE))

            # 2. Chuyển màu BGR -> RGB (Cực kỳ quan trọng vì OpenCV đọc BGR)
            # Lúc train bạn dùng data_loader đã convert RGB chưa?
            # Dựa vào code trước của bạn là cv2.imread -> resize -> X,
            # OpenCV mặc định là BGR. Nếu lúc train bạn để nguyên BGR thì ở đây cũng để nguyên.
            # Tuy nhiên, để chuẩn, ta nên giả định model học đặc trưng hình khối là chính.
            # Hãy thử để nguyên (mặc định OpenCV) trước.

            # 3. Chuẩn hóa [0, 1]
            roi_normalized = roi_resized.astype('float32') / 255.0

            # 4. Dự đoán
            # Model nhận input 3D (64, 64, 3) -> Forward -> Output
            start_time = time.time()
            out = model.forward(roi_normalized)
            infer_time = (time.time() - start_time) * 1000  # ms

            # Lấy kết quả
            pred_idx = np.argmax(out)
            prob = np.max(out)  # Độ tin cậy

            pred_label = CLASSES[pred_idx]

            # 5. HIỂN THỊ KẾT QUẢ
            # Vẽ khung xanh
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Viết chữ kết quả
            text = f"Ky tu: {pred_label} ({prob * 100:.1f}%)"
            color = (0, 255, 0) if prob > 0.5 else (0, 0, 255)  # Xanh nếu tự tin, đỏ nếu ko chắc

            cv2.putText(display_frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(display_frame, "Q [exit]", (540, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Hiển thị tốc độ xử lý
            cv2.putText(display_frame, f"Speed: {infer_time:.1f}ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Hiển thị ảnh nhỏ góc màn hình để xem model "nhìn" thấy gì
            #roi_display = cv2.resize(roi_resized, (150, 150))
            display_frame[100:250, 10:160] = cv2.resize(roi_resized, (150, 150))

        # Hiển thị ra màn hình
        cv2.imshow("Sign Language Translator (From Scratch)", display_frame)

        # Thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()