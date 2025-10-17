# import paddle
# import paddle.nn as nn
# from paddle.vision import datasets, transforms
# from paddle.io import DataLoader

# # 1. 数据预处理：MNIST是28x28灰度图，适配模型输入
# transform = transforms.Compose([
#     transforms.Resize((28, 28)),  # MNIST默认28x28，确保尺寸一致
#     transforms.ToTensor(),        # 转为Tensor（像素值0-1）
#     transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图仅1个通道，标准化
# ])

# # 2. 自动下载MNIST数据集（关键修改：root→image_path）
# train_dataset = datasets.MNIST(
#     image_path='./data',        # 改为image_path，数据自动保存在此路径
#     mode='train',               # 旧版本用mode指定“训练集”，而非train=True
#     transform=transform,        # 应用预处理
#     download=True               # 自动下载（仅首次需要）
# )
# test_dataset = datasets.MNIST(
#     image_path='./data',        # 同样改为image_path
#     mode='test',                # mode指定“测试集”，而非train=False
#     transform=transform,
#     download=True
# )

# # 3. 创建数据加载器（批量读取数据，加速训练）
# train_loader = DataLoader(
#     dataset=train_dataset,
#     batch_size=32,    # 每次读32张图
#     shuffle=True      # 训练集打乱，提升泛化能力
# )
# test_loader = DataLoader(
#     dataset=test_dataset,
#     batch_size=32,
#     shuffle=False     # 测试集无需打乱
# )

# # 4. 定义简单CNN模型（适配手写数字识别，结构轻量化）
# class MNIST_CNN(nn.Layer):
#     def __init__(self, num_classes=10):  # 10个类别（0-9）
#         super(MNIST_CNN, self).__init__()
#         # 卷积层：提取数字的边缘、纹理特征（灰度图输入通道=1）
#         self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         # 池化层：缩小特征图，减少计算量
#         self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
#         # 全连接层：将特征映射到10个类别
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 28→池化2次→7，64个通道
#         self.fc2 = nn.Linear(128, num_classes)
#         self.relu = nn.ReLU()  # 激活函数，增加非线性

#     def forward(self, x):
#         # 前向传播：卷积→激活→池化（2次）→展平→全连接
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = paddle.flatten(x, start_axis=1)  # 展平为一维向量（batch_size, 64*7*7）
#         x = self.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # 5. 初始化模型、损失函数、优化器
# model = MNIST_CNN(num_classes=10)
# criterion = nn.CrossEntropyLoss()  # 交叉熵损失（适合多分类）
# optimizer = paddle.optimizer.Adam(
#     learning_rate=0.001,
#     parameters=model.parameters()  # 待优化的模型参数
# )

# # 6. 模型训练（仅训练3轮，快速验证，不用等太久）
# print("开始训练MNIST数字识别模型...")
# model.train()  # 切换到训练模式
# for epoch in range(3):
#     total_loss = 0.0
#     for batch_id, (images, labels) in enumerate(train_loader):
#         # 1. 前向传播：预测数字类别
#         outputs = model(images)
#         # 2. 计算损失（预测值与真实标签的差距）
#         loss = criterion(outputs, labels)
#         # 3. 反向传播：计算梯度
#         loss.backward()
#         # 4. 优化器更新模型参数
#         optimizer.step()
#         # 5. 清空梯度（避免累积）
#         optimizer.clear_grad()

#         # 打印每100个batch的损失（方便观察训练进度）
#         total_loss += loss.numpy()[0]
#         if (batch_id + 1) % 100 == 0:
#             print(f"Epoch {epoch+1}/3, Batch {batch_id+1}, 平均损失: {total_loss/100:.4f}")
#             total_loss = 0.0

# # 7. 模型验证（测试识别准确率）
# print("\n开始验证模型准确率...")
# model.eval()  # 切换到评估模式（关闭 dropout 等）
# correct = 0  # 正确识别的数量
# total = 0    # 总测试数量
# with paddle.no_grad():  # 关闭梯度计算，节省内存和时间
#     for images, labels in test_loader:
#         outputs = model(images)
#         # 取概率最大的类别作为预测结果
#         _, predicted = paddle.max(outputs, 1)
#         total += labels.shape[0]
#         # 统计正确数量（预测值==真实标签）
#         correct += (predicted == labels).sum().numpy()[0]

# # 打印最终准确率
# accuracy = 100 * correct / total
# print(f"MNIST测试集准确率: {accuracy:.2f}%")
# print("="*50)
# print("若准确率>95%，说明PaddlePaddle框架完全正常！")