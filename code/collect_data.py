import cv2
import os
import time

# --- CÀI ĐẶT CƠ BẢN ---
DATA_DIR = 'dataset2'
IMG_PADDING = 20
x_roi, y_roi = 100, 100
w_roi, h_roi = 300, 300
# --- KẾT THÚC CÀI ĐẶT ---

# --- CẤU HÌNH AUTO SAVE ---
auto_flash = False
auto_interval = 2.0  # Thời gian giãn cách (giây)
last_save_time = 0  # Thời điểm lưu lần cuối

# Nhập label
label = input("Nhập tên ký hiệu (label) (ví dụ: A): ")
label_dir = os.path.join(DATA_DIR, label)

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(label_dir): os.makedirs(label_dir)

counter = len(os.listdir(label_dir))
print(f"Label: {label} - Bắt đầu đếm từ: {counter}")

cap = cv2.VideoCapture(0)

print("\n--- HƯỚNG DẪN ---")
print("['a']: BẬT/TẮT chế độ tự động (2 giây/ảnh)")
print("['s']: Lưu thủ công")
print("['q']: Thoát")

while True:
    success, frame = cap.read()
    if not success: break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    # Tọa độ ROI
    x1 = x_roi - IMG_PADDING
    y1 = y_roi - IMG_PADDING
    x2 = x_roi + w_roi + IMG_PADDING
    y2 = y_roi + h_roi + IMG_PADDING

    # Vẽ khung ROI
    color_roi = (0, 255, 0) if not auto_flash else (0, 255, 255)  # Xanh lá thường, Vàng khi Auto
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_roi, 2)

    # Hiển thị thông tin
    cv2.putText(display_frame, f"Label: {label} ({counter})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Trạng thái Auto
    status_text = "AUTO: ON" if auto_flash else "AUTO: OFF"
    cv2.putText(display_frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_roi, 2)

    # --- XỬ LÝ LOGIC AUTO FLASH (NON-BLOCKING) ---
    if auto_flash:
        # Tính thời gian trôi qua kể từ lần lưu cuối
        current_time = time.time()
        elapsed = current_time - last_save_time

        # Tính thời gian còn lại để hiển thị đếm ngược
        remaining = auto_interval - elapsed

        if remaining > 0:
            # Hiển thị đếm ngược ngay giữa ROI
            countdown_text = f"{remaining:.1f}"
            cv2.putText(display_frame, countdown_text, (x_roi + 100, y_roi + 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
        else:
            # Đã ĐỦ 2 giây -> Thực hiện LƯU
            try:
                img_crop = frame[y1:y2, x1:x2]
                if img_crop.size != 0:
                    timestamp = int(time.time() * 1000)
                    img_path = os.path.join(label_dir, f"{label}_{timestamp}.jpg")
                    cv2.imwrite(img_path, img_crop)
                    print(f"Auto Saved: {img_path}")
                    counter += 1

                    # Reset lại đồng hồ
                    last_save_time = time.time()

                    # Hiệu ứng nháy màn hình khi lưu thành công
                    cv2.rectangle(display_frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 10)
            except Exception as e:
                print(f"Lỗi auto save: {e}")

    # --- XỬ LÝ PHÍM BẤM ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    # Bật tắt chế độ Auto
    if key == ord('a'):
        auto_flash = not auto_flash
        last_save_time = time.time()

    # if key == ord('w'):
    #     auto_flash = False

    # Lưu thủ công (vẫn giữ tính năng này)
    if key == ord('s'):
        img_crop = frame[y1:y2, x1:x2]
        if img_crop.size != 0:
            timestamp = int(time.time() * 1000)
            img_path = os.path.join(label_dir, f"{label}_{timestamp}.jpg")
            cv2.imwrite(img_path, img_crop)
            counter += 1
            print(f"Manual Saved: {img_path}")

    cv2.imshow("Data Collector Pro", display_frame)

cap.release()
cv2.destroyAllWindows()