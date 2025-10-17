UAV-Based Crop Health Monitoring with CNN
基于无人机 RGB 图像与 CNN 的作物健康监测系统
🔍 项目概述 (Project Overview)
目标：通过无人机搭载的 RGB 摄像头采集作物图像，结合卷积神经网络（CNN）实现作物健康状态的自动分类（健康、轻度胁迫、重度胁迫），为农田管理提供可视化决策支持。
Goal: Automatically classify crop health status (healthy, slight stress, severe stress) using RGB images captured by UAV-mounted cameras and Convolutional Neural Networks (CNN), providing visual decision support for farm management.
🛠️ 技术栈 (Tech Stack)
类型 (Type)	具体工具 (Tools)
硬件 (Hardware)	消费级无人机（如大疆 Mini 系列，RGB 摄像头）、普通电脑（支持 Python 运行）
Consumer-grade UAV (e.g., DJI Mini series with RGB camera), general-purpose computer (Python-compatible)
软件 (Software)	Python 3.9、PaddlePaddle 2.5.2（深度学习框架）、OpenCV（图像处理）、Matplotlib（结果可视化）
Python 3.9, PaddlePaddle 2.5.2 (deep learning framework), OpenCV (image processing), Matplotlib (visualization)
⚙️ 环境配置 (Environment Setup)
1. 依赖安装 (Dependency Installation)
先安装 Anaconda，再创建虚拟环境并安装依赖：
First install Anaconda, then create a virtual environment and install dependencies:
bash
# 1. 创建并激活虚拟环境 (Create and activate virtual environment)
conda create -n paddle_env python=3.9 -y
conda activate paddle_env

# 2. 安装核心依赖（国内镜像加速）
# Install core dependencies (via Tsinghua mirror for speed)
pip install paddlepaddle==2.5.2 opencv-python numpy matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
2. 环境验证 (Environment Verification)
在项目根目录运行测试脚本，验证 PaddlePaddle 是否正常工作：
Run the test script in the project root to verify PaddlePaddle:
bash
python test_paddle.py
若输出 PaddlePaddle is installed successfully!，则环境配置完成。If PaddlePaddle is installed successfully! is displayed, the environment is ready.
📂 项目文件结构 (Project File Structure)
plaintext
uav-based-crop-health-cnn/
├── data/                 # 作物图像数据集 (Crop image dataset)
│   ├── train/            # 训练集（70%）(Training set, 70% of data)
│   │   ├── healthy/      # 健康作物图像 (Healthy crop images)
│   │   ├── slight_stress/ # 轻度胁迫图像 (Slight stress images)
│   │   └── severe_stress/ # 重度胁迫图像 (Severe stress images)
│   └── test/             # 测试集（30%）(Test set, 30% of data)
├── code/                 # 核心代码 (Core code)
│   ├── data_preprocess.py # 图像预处理 (Image preprocessing)
│   ├── model.py          # CNN模型定义 (CNN model definition)
│   ├── train.py          # 模型训练与验证 (Training & validation)
│   └── predict.py        # 无人机图像推理 (UAV image inference)
├── models/               # 训练好的模型权重 (Trained model weights)
├── results/              # 推理结果（健康地图等）(Inference results)
├── test_paddle.py        # 环境验证脚本 (Environment test script)
└── README.md             # 项目说明文档 (Project documentation)