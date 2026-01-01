import numpy as np
import pickle  # Để lưu model sau khi train


# ==========================================
# 1. CÁC LỚP CƠ BẢN (BUILDING BLOCKS)
# ==========================================

class Conv3x3_RGB:
    def __init__(self, num_filters, input_depth=3):
        self.num_filters = num_filters
        self.input_depth = input_depth
        # He Init: Tốt cho ReLU
        self.filters = np.random.randn(num_filters, 3, 3, input_depth) * 0.1

    def iterate_regions(self, image):
        h, w, _ = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3), :]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, _ = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2, 3))
        return output

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        self.filters -= learn_rate * d_L_d_filters
        return None


class MaxPool2:
    def iterate_regions(self, image):
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backward(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
        return d_L_d_input


class ReLU:
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)

    def backward(self, d_L_d_out):
        d_L_d_input = d_L_d_out.copy()
        d_L_d_input[self.last_input <= 0] = 0
        return d_L_d_input


class FullyConnected:
    def __init__(self, input_len, nodes):
        # He Initialization
        self.weights = np.random.randn(input_len, nodes) * np.sqrt(2.0 / input_len)
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, d_L_d_out, learn_rate):
        d_L_d_w = self.last_input[np.newaxis].T @ d_L_d_out[np.newaxis]
        d_L_d_b = d_L_d_out
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)

        self.weights -= learn_rate * d_L_d_w
        self.biases -= learn_rate * d_L_d_b

        return d_L_d_input.reshape(self.last_input_shape)


class Softmax:
    def forward(self, input):
        exp_vals = np.exp(input - np.max(input))
        return exp_vals / np.sum(exp_vals, axis=0)

    def backward(self, d_L_d_out):
        return d_L_d_out




class SignLanguageCNN:
    def __init__(self, num_classes=4, img_size=96):
        """
        img_size: Kích thước ảnh đầu vào (Ví dụ: 96 hoặc 64)
        """
        self.num_classes = num_classes
        self.img_size = img_size

        # 1. Khởi tạo các lớp
        self.conv = Conv3x3_RGB(num_filters=8, input_depth=3)
        self.relu = ReLU()
        self.pool = MaxPool2()
        self.softmax = Softmax()

        conv_output_size = self.img_size - 2

        pool_output_size = conv_output_size // 2

        self.fc_input_len = pool_output_size * pool_output_size * 8

        print(
            f"ℹ️ Cấu hình mạng: Input({img_size}x{img_size}) -> Conv -> Pool({pool_output_size}x{pool_output_size}x8) -> Flatten({self.fc_input_len})")

        self.fc = FullyConnected(input_len=self.fc_input_len, nodes=num_classes)

    def forward(self, image):

        if len(image.shape) == 4:
            image = image[0]

        out = self.conv.forward(image)
        out = self.relu.forward(out)
        out = self.pool.forward(out)
        out = self.fc.forward(out)
        out = self.softmax.forward(out)
        return out

    def train_step(self, image, label, lr=0.005):
        # 1. Forward
        out = self.forward(image)

        # 2. Tính Loss (Cross-Entropy)
        loss = -np.log(out[label] + 1e-9)
        acc = 1 if np.argmax(out) == label else 0

        # 3. Tính Gradient
        gradient = out.copy()
        gradient[label] -= 1

        # 4. Backward
        gradient = self.fc.backward(gradient, lr)
        gradient = self.pool.backward(gradient)
        gradient = self.relu.backward(gradient)
        gradient = self.conv.backward(gradient, lr)

        return loss, acc

    def save_model(self, filename='sign_language_model.pkl'):
        model_data = {
            'img_size': self.img_size,  # Lưu luôn kích thước ảnh để lúc load biết đường resize
            'num_classes': self.num_classes,
            'conv_filters': self.conv.filters,
            'fc_weights': self.fc.weights,
            'fc_biases': self.fc.biases
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Đã lưu model vào '{filename}'")

    def load_model(self, filename='sign_language_model.pkl'):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            # Kiểm tra xem file model có khớp cấu hình không
            if data.get('img_size') and data['img_size'] != self.img_size:
                print(f"⚠️ Cảnh báo: Model được train với size {data['img_size']}, nhưng bạn đang set {self.img_size}.")

            self.conv.filters = data['conv_filters']
            self.fc.weights = data['fc_weights']
            self.fc.biases = data['fc_biases']
            print(f"📂 Đã load trọng số từ '{filename}'")
            return True
        except Exception as e:
            print(f"❌ Lỗi load model: {e}")
            return False

    def load_model(self, filename='model_weights.pkl'):
        # Đọc trọng số vào
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.conv.filters = data['conv_filters']
        self.fc.weights = data['fc_weights']
        self.fc.biases = data['fc_biases']
        print(f"📂 Đã load model từ '{filename}'")