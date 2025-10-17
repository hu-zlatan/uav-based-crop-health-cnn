# UAV-Based Crop Health Monitoring with CNN
基于无人机 RGB 图像与 CNN 的作物健康监测系统
## 🔍 项目概述 (Project Overview)
- 目标：通过无人机搭载的 RGB 摄像头采集作物图像，结合卷积神经网络（CNN）实现作物健康状态的自动分类（健康、轻度胁迫、重度胁迫），为农田管理提供可视化决策支持。
- Goal: Automatically classify crop health status (healthy, slight stress, severe stress) using RGB images captured by UAV-mounted cameras and Convolutional Neural Networks (CNN), providing visual decision support for farm management.
### 技术栈 (Tech Stack)  
- **硬件 (Hardware)**  
  消费级无人机（如大疆Mini系列，配备RGB摄像头）、电脑（支持Python运行）  
  Consumer-grade UAV (e.g., DJI Mini series with RGB camera), PC (compatible with Python)  

- **软件 (Software)**  
  Python 3.9、PaddlePaddle 3.2.0（深度学习框架）、OpenCV（图像处理）、Matplotlib（结果可视化）  
  Python 3.9, PaddlePaddle 3.2.0 (deep learning framework), OpenCV (image processing), Matplotlib (result visualization)  
## ⚙️ 环境配置 (Environment Setup)
### 1. 依赖安装 (Dependency Installation)
先安装 Anaconda，再创建虚拟环境并安装依赖：
First install Anaconda, then create a virtual environment and install dependencies:
bash
 1. 创建并激活虚拟环境 (Create and activate virtual environment)

    conda create -n paddle_env python=3.9 -y
    
    conda activate paddle_env

 3. 安装核心依赖 (Install core dependencies)

https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html
### 2. 环境验证 (Environment Verification)
在项目根目录运行测试脚本，验证 PaddlePaddle 是否正常工作：

Run the test script in the project root to verify PaddlePaddle:

bash

python test_paddle.py
- 若输出 PaddlePaddle is installed successfully!，则环境配置完成。
- If PaddlePaddle is installed successfully! is displayed, the environment is ready.
## 📂 项目文件结构 (Project File Structure)
