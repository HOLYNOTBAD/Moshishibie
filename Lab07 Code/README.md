# OpenCV图像处理与特征匹配实验

基于《Lab07 OpenCV图像处理与特征匹配》的完整代码实现，包含四个实验的Python脚本和可视化演示。

## 项目结构

```
├── Lab07.md                    # 提炼出的原始实验文档
├── README.md                   # 项目总说明
├── experiments/                # 实验代码文件夹
│   ├── README.md              # 实验使用说明
│   ├── run_all_experiments.py # 批量运行脚本
│   ├── create_sample_images.py # 示例图像生成
│   ├── exp1_setup_verification.py    # 实验一：环境验证
│   ├── exp2_image_operations.py      # 实验二：图像文件操作
│   ├── exp3_image_processing.py      # 实验三：图像处理
│   └── exp4_feature_matching.py      # 实验四：特征检测与匹配
├── resources/                 # 示例图像资源
│   ├── basic_shapes.jpg      # 基本形状图像
│   ├── chessboard.jpg        # 棋盘格（角点检测）
│   ├── gradient.jpg          # 渐变图案
│   ├── texture.jpg           # 纹理图像
│   ├── line_patterns.jpg     # 线条图案（霍夫变换）
│   ├── object.jpg            # 测试物体
│   └── scene.jpg             # 场景图像
└── results/                   # 实验结果输出
```

## 快速开始

### 1. 环境要求
- Python 3.x
- OpenCV (opencv-python + opencv-contrib-python)
- NumPy, Matplotlib

### 2. 安装依赖
```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

### 3. 运行所有实验
```bash
cd experiments
python run_all_experiments.py
```

### 4. 查看结果
所有输出结果保存在 `results/` 文件夹中。

## 实验内容

### 实验一：环境验证
- ✅ OpenCV版本和环境检查
- ✅ 基本图像操作测试
- ✅ 摄像头功能验证
- ✅ contrib模块可用性检查

### 实验二：图像文件操作
- ✅ 多格式图像读取写入
- ✅ NumPy数组数据结构分析
- ✅ 像素级操作和ROI处理
- ✅ 视频流捕获和处理

### 实验三：图像处理
- ✅ 自定义卷积核滤波（锐化、模糊、边缘检测、浮雕）
- ✅ Canny多级边缘检测
- ✅ 轮廓检测与形状描述（边界框、凸包、多边形拟合）
- ✅ 霍夫变换（直线检测HoughLinesP、圆检测HoughCircles）

### 实验四：特征检测与匹配
- ✅ Harris角点检测
- ✅ SIFT/SURF关键点检测（需要contrib模块）
- ✅ ORB特征检测与描述
- ✅ 特征匹配（BFMatcher、FLANN + KNN + 比率检验）
- ✅ 单应性变换与物体定位

## 主要特性

- **完整实现**：涵盖Lab07文档中的所有实验内容
- **可视化丰富**：每个实验都包含matplotlib图表和图像输出
- **模块化设计**：每个实验独立运行，功能分离清晰
- **自动资源生成**：脚本自动创建所需的测试图像
- **中英文注释**：代码包含详细的中英文注释
- **错误处理**：包含完善的异常处理和用户提示

## 使用说明

### 单个实验运行
```bash
# 实验一：环境验证
python experiments/exp1_setup_verification.py

# 实验二：图像操作
python experiments/exp2_image_operations.py

# 实验三：图像处理
python experiments/exp3_image_processing.py

# 实验四：特征匹配
python experiments/exp4_feature_matching.py
```

### 生成示例图像
```bash
python experiments/create_sample_images.py
```

## 输出文件说明

运行实验后，`results/` 文件夹将包含：

- **exp1_*.png/jpg** - 环境验证结果
- **exp2_*.png/jpg** - 图像操作演示
- **exp3_*.png/jpg** - 图像处理结果
- **exp4_*.png/jpg** - 特征检测匹配结果
- ***_comparison.png** - 算法对比图
- ***_demo.png** - 综合演示图

## 注意事项

1. **硬件要求**：摄像头测试需要摄像头硬件
2. **依赖包**：SIFT/SURF需要opencv-contrib-python
3. **执行时间**：特征检测实验可能需要较长时间
4. **内存使用**：大图像处理需要足够内存

## 学习建议

1. **循序渐进**：建议按实验顺序学习每个概念
2. **参数调整**：尝试修改脚本中的参数观察效果变化
3. **算法对比**：比较不同算法的优缺点和适用场景
4. **扩展应用**：尝试将这些技术应用到实际项目中

## 参考资料

- [OpenCV官方文档](https://docs.opencv.org/)
- [OpenCV Python教程](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- 原始实验文档：Lab07.md

---


