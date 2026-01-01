import cv2
import numpy as np
import socket
import struct
import time
import os
from cnn_model import SignLanguageCNN

# =============================================================
# 1. CẤU HÌNH SERVER
# =============================================================
HOST = '0.0.0.0'
PORT = 8888
MODEL_PATH = 'sign_language_model.pkl'
IMG_SIZE = 96
CONFIDENCE_THRESHOLD = 0.6

CLASSES = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y']

# =============================================================
# 2. LOAD MODEL
# =============================================================
print(f"⏳ Đang tải mô hình từ {MODEL_PATH}...")
model = SignLanguageCNN(num_classes=len(CLASSES), img_size=IMG_SIZE)

if not os.path.exists(MODEL_PATH):
    print(f"❌ LỖI: Không tìm thấy file '{MODEL_PATH}'")
    exit()

try:
    model.load_model(MODEL_PATH)
    print("✅ Mô hình đã sẵn sàng!")
except Exception as e:
    print(f"❌ Lỗi load model: {e}")
    exit()


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print(f"🚀 SERVER AI ĐANG CHẠY! (Port: {PORT})")
    print("📡 Đang đợi ESP32 kết nối...")

    conn, addr = server_socket.accept()
    print(f"✅ Đã kết nối: {addr}")

    buffer = b''
    prev_time = 0

    while True:
        try:
            # --- A. NHẬN DỮ LIỆU ---
            while b'\n' not in buffer:
                data = conn.recv(1024)
                if not data: break
                buffer += data
            if not buffer: break

            header, buffer = buffer.split(b'\n', 1)
            header_str = header.decode('utf-8').strip()

            if header_str.startswith("SIZE:"):
                img_size = int(header_str.split(":")[1])

                while len(buffer) < img_size:
                    data = conn.recv(4096)
                    if not data: break
                    buffer += data

                img_data = buffer[:img_size]
                buffer = buffer[img_size:]

                # --- B. XỬ LÝ ẢNH ---
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    # Tính FPS
                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
                    prev_time = curr_time

                    # 1. Cắt ảnh vuông (Center Crop)
                    h, w = frame.shape[:2]
                    side = min(h, w)
                    x1 = (w - side) // 2
                    y1 = (h - side) // 2
                    x2, y2 = x1 + side, y1 + side

                    roi = frame[y1:y2, x1:x2]

                    # 2. Tiền xử lý (Resize -> RGB -> Normalize)
                    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
                    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)  # Đảo màu cho AI
                    roi_norm = roi_rgb.astype('float32') / 255.0

                    # 3. Dự đoán
                    out = model.forward(roi_norm)
                    pred_idx = np.argmax(out)
                    prob = np.max(out)

                    # --- C. GỬI KẾT QUẢ VỀ ESP32 ---
                    message = "?"
                    display_color = (0, 0, 255)  # Đỏ

                    if prob > CONFIDENCE_THRESHOLD:
                        pred_label = CLASSES[pred_idx]
                        display_color = (0, 255, 0)  # Xanh
                        message = pred_label

                    # [QUAN TRỌNG] Thêm \n để ESP32 nhận diện kết thúc dòng
                    try:
                        final_msg = message + "\n"
                        conn.send(final_msg.encode('utf-8'))
                        print(final_msg)
                    except:
                        pass

                    # --- D. HIỂN THỊ (VISUALIZATION) ---
                    display_frame = cv2.resize(frame, (640, 480))

                    scale_x = 640 / w
                    scale_y = 480 / h
                    cv2.rectangle(display_frame,
                                  (int(x1 * scale_x), int(y1 * scale_y)),
                                  (int(x2 * scale_x), int(y2 * scale_y)),
                                  display_color, 3)

                    text = f"Pred: {message} ({prob:.0%})"
                    cv2.putText(display_frame, text, (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, display_color, 3)
                    cv2.putText(display_frame, f"FPS: {int(fps)}", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    # [SỬA LỖI MÀU] Dùng roi_resized (BGR) để hiển thị, KHÔNG dùng roi_rgb
                    debug_roi = cv2.resize(roi_resized, (100, 100))
                    display_frame[380:480, 540:640] = debug_roi
                    cv2.rectangle(display_frame, (540, 380), (640, 480), (255, 255, 255), 1)

                    cv2.imshow("Sign Language AI System", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"⚠️ Mất kết nối: {e}")
            conn.close()
            print("⏳ Đợi kết nối lại...")
            conn, addr = server_socket.accept()
            print(f"✅ Đã kết nối lại: {addr}")
            buffer = b''

    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()