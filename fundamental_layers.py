import torch

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化卷积核权重和偏置
        self.weights = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1
        self.bias = torch.randn(out_channels) * 0.1

    def forward(self, x):
        # 添加 padding
        if self.padding > 0:
            x = torch.nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        batch_size, in_channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        # 初始化输出
        output = torch.zeros(batch_size, self.out_channels, out_height, out_width)

        # 滑动窗口计算卷积
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        # 提取局部区域
                        local_region = x[b, :, h_start:h_end, w_start:w_end]
                        # 计算点积并加上偏置
                        output[b, oc, oh, ow] = torch.sum(local_region * self.weights[oc]) + self.bias[oc]

        return output
    
    
class MaxPool2d:
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        batch_size, channels, in_height, in_width = x.shape
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        # 初始化输出
        output = torch.zeros(batch_size, channels, out_height, out_width)

        # 滑动窗口计算最大池化
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        # 提取局部区域
                        local_region = x[b, c, h_start:h_end, w_start:w_end]
                        # 取最大值
                        output[b, c, oh, ow] = torch.max(local_region)

        return output
    
    
class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # 初始化权重和偏置
        self.weights = torch.randn(out_features, in_features) * 0.1
        self.bias = torch.randn(out_features) * 0.1

    def forward(self, x):
        # 矩阵乘法并加上偏置
        return torch.matmul(x, self.weights.t()) + self.bias
    
class ReLU:
    def forward(self, x):
        return torch.maximum(torch.tensor(0), x)

    
class LeNet5:
    def __init__(self):
        self.conv1 = Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = Linear(in_features=120, out_features=84)
        self.fc3 = Linear(in_features=84, out_features=10)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu.forward(self.conv1.forward(x))
        x = self.pool1.forward(x)
        x = self.relu.forward(self.conv2.forward(x))
        x = self.pool2.forward(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.relu.forward(self.fc1.forward(x))
        x = self.relu.forward(self.fc2.forward(x))
        x = self.fc3.forward(x)
        return x
    
if __name__ == "__main__":
    # 创建手动实现的 LeNet-5 模型
    model = LeNet5()

    # 生成随机输入
    x = torch.randn(1, 1, 32, 32)  # 输入大小为 (batch_size, channels, height, width)

    # 前向传播
    output = model.forward(x)
    print("Output :", output)
    print("Output shape:", output.shape)