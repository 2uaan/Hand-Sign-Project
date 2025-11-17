import cv2
import os
import time

# --- CÀI ĐẶT CƠ BẢN ---
DATA_DIR = 'dataset'  # Tên thư mục gốc chứa dataset
IMG_PADDING = 20  # Vùng đệm nhỏ xung quanh ROI

# Tọa độ của Vùng Quan Tâm (ROI) - Bạn có thể tùy chỉnh
# Đây là tọa độ (x, y) của góc trên bên trái
x_roi, y_roi = 100, 100
# Đây là chiều rộng và chiều cao của hộp ROI
w_roi, h_roi = 300, 300
# --- KẾT THÚC CÀI ĐẶT ---


# Hỏi người dùng tên label
label = input("Nhập tên ký hiệu (label) bạn muốn thu thập (ví dụ: A): ")

# Tạo đường dẫn thư mục cho label
label_dir = os.path.join(DATA_DIR, label)

# Kiểm tra và tạo thư mục nếu chưa tồn tại
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"Đã tạo thư mục: {DATA_DIR}")

if not os.path.exists(label_dir):
    os.makedirs(label_dir)
    print(f"Đã tạo thư mục: {label_dir}")
else:
    print(f"Thư mục '{label_dir}' đã tồn tại. Sẽ thêm ảnh vào đây.")

# Đếm số lượng ảnh đã có để bắt đầu từ số tiếp theo
counter = len(os.listdir(label_dir))
print(f"Bắt đầu đếm từ: {counter}")

# Khởi động webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở webcam!")
    exit()

print("\n--- HƯỚNG DẪN ---")
print("Đặt tay vào khung MÀU XANH.")
print("Nhấn 's' để LƯU ảnh.")
print("Nhấn 'q' để THOÁT.")
print("------------------")

while True:
    # Đọc từng frame
    success, frame = cap.read()
    if not success:
        print("Lỗi: Không thể đọc frame.")
        break

    # Lật ảnh (giống như soi gương)
    frame = cv2.flip(frame, 1)

    # Tạo một bản sao của frame để vẽ lên,
    # chúng ta sẽ lưu frame gốc (chưa vẽ)
    display_frame = frame.copy()

    # Tính toán tọa độ hộp ROI (x1, y1, x2, y2)
    x1 = x_roi - IMG_PADDING
    y1 = y_roi - IMG_PADDING
    x2 = x_roi + w_roi + IMG_PADDING
    y2 = y_roi + h_roi + IMG_PADDING

    # Vẽ hộp ROI lên màn hình (màu xanh lá)
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(display_frame, f"Collecting: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Count: {counter}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_frame, "('s': save)", (250, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, "('q': quit)", (400, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Hiển thị cửa sổ webcam
    cv2.imshow("Data Collector", display_frame)

    # Chờ phím nhấn
    key = cv2.waitKey(1) & 0xFF

    # Nhấn 'q' để thoát
    if key == ord('q'):
        break

    # Nhấn 's' để lưu ảnh
    if key == ord('s'):
        try:
            # Cắt ảnh từ frame gốc (không có chữ hay hình vẽ)
            # Lưu ý: thứ tự là [y1:y2, x1:x2]
            img_crop = frame[y1:y2, x1:x2]

            # Kiểm tra xem có cắt được không (phòng trường hợp ROI ra ngoài màn hình)
            if img_crop.size == 0:
                print("Lỗi: Vùng ROI nằm ngoài khung hình!")
                continue

            # Tạo tên file duy nhất (dùng timestamp cho đơn giản và tuyệt đối)
            timestamp = int(time.time() * 1000)
            img_path = os.path.join(label_dir, f"{label}_{timestamp}.jpg")

            # Lưu ảnh
            cv2.imwrite(img_path, img_crop)

            print(f"Đã lưu: {img_path}")
            counter += 1

        except Exception as e:
            print(f"Lỗi khi lưu ảnh: {e}")

# Giải phóng webcam và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()