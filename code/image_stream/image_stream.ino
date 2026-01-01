#include <WiFi.h>
#include "esp_camera.h"

// =============================================================
// 1. CẤU HÌNH WIFI & SERVER
// =============================================================
const char* ssid = "2uaan";      // <--- Thay tên WiFi
const char* password = "22222888";     // <--- Thay Pass WiFi
const char* serverIP = "10.149.209.3";       // <--- Thay IP Laptop (xem ipconfig)
const int serverPort = 8888;

// [THAY ĐỔI QUAN TRỌNG] Đổi sang LED đỏ mặt sau
#define LED_PIN 33 

// Pin Map AI-Thinker
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

WiFiClient client;

void setup() {
  Serial.begin(115200);
  
  // Cấu hình LED đỏ
  pinMode(LED_PIN, OUTPUT);
  // [QUAN TRỌNG] GPIO 33 hoạt động ngược: HIGH là TẮT
  digitalWrite(LED_PIN, HIGH); 

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
  config.pixel_format = PIXFORMAT_JPEG; 
  
  if(psramFound()){
    config.frame_size = FRAMESIZE_QVGA; 
    config.jpeg_quality = 12;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }
  
  esp_camera_init(&config);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(200);
    // Nháy đèn báo đang kết nối
    // Logic ngược: LOW là sáng, HIGH là tắt
    digitalWrite(LED_PIN, !digitalRead(LED_PIN)); 
  }
  // Kết nối xong thì TẮT đèn (HIGH)
  digitalWrite(LED_PIN, HIGH); 
}

void loop() {
  // Tự động kết nối lại
  if (!client.connected()) {
    if (!client.connect(serverIP, serverPort)) {
      delay(500); return;
    }
  }

  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) return;

  client.print("SIZE:");
  client.println(fb->len);
  client.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);

  // --- XỬ LÝ TÍN HIỆU TỪ SERVER ---
  if (client.available()) {
    String msg = client.readStringUntil('\n');
    msg.trim();
    
    // Nếu nhận được nhãn (khác dấu ?)
    if (msg.length() > 0 && msg != "?") {
       // Nháy đèn LED đỏ (Logic ngược)
       digitalWrite(LED_PIN, LOW);  // Bật (Sáng)
       delay(50);                   // Giữ 50ms
       digitalWrite(LED_PIN, HIGH); // Tắt
    }
    
    // Xóa bộ đệm
    while(client.available()) client.read();
  }
}