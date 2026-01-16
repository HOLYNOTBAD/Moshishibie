# OpenCV实验系列 - 代码实现

本文件夹包含了《Lab07 OpenCV图像处理与特征匹配》实验的完整代码实现。

## 文件结构

```
experiments/
├── README.md                           # 本说明文件
├── run_all_experiments.py             # 批量运行所有实验
├── create_sample_images.py            # 创建示例图像资源
├── exp1_setup_verification.py         # 实验一：环境验证
├── exp2_image_operations.py           # 实验二：图像文件操作
├── exp3_image_processing.py           # 实验三：图像处理
└── exp4_feature_matching.py           # 实验四：特征检测与匹配

../resources/                          # 示例图像资源文件夹
../results/                            # 实验结果输出文件夹
```

## 实验内容概述

### 实验一：环境验证
- OpenCV版本检查
- 基本图像操作测试
- 摄像头捕获测试
- contrib模块可用性检查

### 实验二：图像文件操作
- 图像读取写入（不同格式和模式）
- 图像数据结构分析
- 像素访问和ROI操作
- 视频捕获和处理

### 实验三：图像处理
- 自定义卷积核与滤波
- Canny边缘检测
- 轮廓检测与形状分析
- 霍夫变换（直线和圆检测）

### 实验四：特征检测与匹配
- Harris角点检测
- SIFT/SURF特征检测
- ORB特征检测
- 特征匹配（BFMatcher和FLANN）
- 单应性变换与物体定位

## 使用方法

### 1. 准备环境
确保已安装必要的Python包：
```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

### 2. 运行单个实验
```bash
# 运行实验一
python exp1_setup_verification.py

# 运行实验二
python exp2_image_operations.py

# 运行实验三
python exp3_image_processing.py

# 运行实验四
python exp4_feature_matching.py
```

### 3. 批量运行所有实验
```bash
python run_all_experiments.py
```

### 4. 创建示例图像
```bash
python create_sample_images.py
```

## 输出结果

所有实验结果将保存在 `../results/` 文件夹中，包括：
- 处理后的图像文件（.jpg, .png）
- 可视化图表（.png）
- 中间结果文件

## 注意事项

1. **摄像头测试**：实验一和二包含摄像头测试，需要摄像头硬件支持
2. **contrib模块**：实验四中的SIFT和SURF需要opencv-contrib-python包
3. **执行时间**：某些实验（如特征检测）可能需要较长时间
4. **文件路径**：脚本会自动创建必要的文件夹和示例图像

## 依赖包

- opencv-python
- opencv-contrib-python (推荐，用于完整功能)
- numpy
- matplotlib

## 故障排除

1. **ImportError**: 确保已安装所有必需的包
2. **摄像头不可用**: 跳过摄像头相关的测试部分
3. **contrib功能不可用**: SIFT/SURF可能需要许可证，ORB是免费替代方案
4. **内存不足**: 对于大图像，可以调整脚本中的参数

## 扩展

这些脚本可以作为OpenCV学习的起点，您可以：
- 修改参数来观察不同效果
- 添加新的图像处理算法
- 集成到更大的计算机视觉项目中
- 作为教学演示工具使用

---

如有问题，请参考原始的Lab07.md文档或OpenCV官方文档。
