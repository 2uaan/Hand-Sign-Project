#include <WiFi.h>
#include "esp_camera.h"

// --- 1. CẤU HÌNH WIFI & SERVER ---
const char* ssid = "12B05";        // <--- SỬA LẠI
const char* password = "11111112";       // <--- SỬA LẠI
const char* serverIP = "192.168.1.2";         // <--- SỬA LẠI (IP của Laptop)
const int serverPort = 8888;                  // Port kết nối

WiFiClient client;

// --- 2. CẤU HÌNH CAMERA (AI THINKER) ---
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM       5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void setup() {
  Serial.begin(115200);
  
  // Cấu hình Camera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG; // Gửi ảnh JPEG cho nhẹ
  
  if(psramFound()){
    config.frame_size = FRAMESIZE_QVGA; // 320x240 (Đủ dùng, gửi nhanh)
    config.jpeg_quality = 12;           // Chất lượng tốt
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  
  // Khởi động Camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Lỗi Camera: 0x%x", err);
    return;
  }

  // Kết nối WiFi
  WiFi.begin(ssid, password);
  Serial.print("Dang ket noi WiFi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Nếu chưa kết nối tới Server thì thử kết nối
  if (!client.connected()) {
    Serial.println("Dang ket noi toi Laptop Server...");
    if (client.connect(serverIP, serverPort)) {
      Serial.println("Da ket noi voi Laptop!");
    } else {
      delay(500); // Thử lại sau 0.5s
      return;
    }
  }

  // Chụp ảnh
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Loi chup anh");
    return;
  }

  // Gửi kích thước ảnh trước (Để Laptop biết ảnh nặng bao nhiêu bytes)
  // Gửi header dạng chuỗi text + xuống dòng, ví dụ: "SIZE:1024\n"
  client.print("SIZE:");
  client.println(fb->len); // fb->len là độ dài byte của ảnh JPEG
  
  // Gửi nội dung ảnh (Raw bytes)
  client.write(fb->buf, fb->len);

  // Giải phóng bộ nhớ
  esp_camera_fb_return(fb);
  
  // Delay nhẹ để không bị quá tải
  // delay(50); 
}