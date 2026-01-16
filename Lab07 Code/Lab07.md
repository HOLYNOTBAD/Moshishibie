## Lab07 OpenCV图像处理与特征匹配

### 一、 实验一：OpenCV安装与环境搭建

#### 1. 实验目的
* 在 Windows 环境下搭建 Python 机器视觉开发环境。
* 掌握 pip 安装与源码编译（CMake + VS）两种 OpenCV 安装方法。
* 学会运行官方示例代码以验证环境。

#### 2. 实验环境
* **操作系统**：Windows (x64)
* **语言**：Python 3.x
* **工具**：Visual Studio 2019, CMake (源码编译需用)

#### 3. 实验步骤

**步骤 1：基础环境配置**
* 安装 Python 3.x（勾选 Add Python to PATH）。
* 安装基础科学计算库：
  `pip install numpy scipy`

**步骤 2：安装 OpenCV**
*(根据需求选择一种方式)*

* **方式 A：使用 pip 安装（常规推荐）**
    直接安装包含扩展模块的预编译包：
    `pip install opencv-contrib-python`

* **方式 B：源码编译安装（支持深度相机/定制）**
    1.  下载 OpenCV 及 opencv_contrib 源码。
    2.  安装 Visual Studio 2019 (C++) 及 CMake。
    3.  **CMake 配置**：指定 Python 库路径、Include 目录，开启 `WITH_OPENNI2` (如需)，指定 `OPENCV_EXTRA_MODULES_PATH`。
    4.  **编译**：VS 打开解决方案 -> 选择 **Release** -> 构建 **INSTALL** 项目。
    5.  **配置变量**：将构建生成的 `bin` 目录添加到系统 Path 环境变量，注销并重登。

**步骤 3：验证安装**
从 OpenCV 源码包中提取 `samples/python` 目录，运行以下脚本：
* **直方图测试**：`python hist.py` (显示图像及RGB/灰度直方图)。
* **光流法测试**：`python opt_flow.py` (需摄像头，显示运动检测可视化)。
* *注：若无报错且窗口正常弹出，即表示安装成功。*

#### 4. 文档查阅
* **API 文档**：http://docs.opencv.org/
* **模块导入**：在 Python 4.x/3.x 中，OpenCV 模块名统一为 `cv2`。





## 实验二：图像文件操作

### 一、 实验目的
1.  掌握 OpenCV 中图像读写的基本函数（`imread`, `imwrite`）及支持的文件格式。
2.  理解图像在内存中的数据结构（NumPy 数组）及 BGR 通道顺序。
3.  掌握像素级访问（`item`/`itemset`）与感兴趣区域（ROI）的切片操作。
4.  学会处理视频流：读取视频文件、捕获摄像头数据、保存视频及窗口显示交互。

### 二、 实验内容与步骤

#### 1. 图像文件的读取与写入 (2.2.1)
OpenCV 处理图像的核心在于将图像文件加载为多维 NumPy 数组。

* **读取图像**：使用 `cv2.imread()` 函数。
    * OpenCV 默认使用 **BGR**（蓝-绿-红）颜色空间，而非标准的 RGB。
    * **常用参数**：
        * `cv2.IMREAD_COLOR` (默认)：加载彩色图像。
        * `cv2.IMREAD_GRAYSCALE`：加载灰度图像。
        * `cv2.IMREAD_UNCHANGED`：加载包含 Alpha 通道的图像。
* **写入图像**：使用 `cv2.imwrite()` 将 NumPy 数组保存为图像文件（如 .jpg, .png）。支持 BMP, PNG, JPEG, TIFF 等格式。

**代码示例：格式转换**
```python
import cv2
# 以灰度模式读取 PNG 图像
gray_image = cv2.imread('MyPic.png', cv2.IMREAD_GRAYSCALE)
# 保存为新的文件
cv2.imwrite('MyPicGray.png', gray_image)
```

#### 2. 图像表示与字节转换 (2.2.2)
图像在 OpenCV 中本质上是一个多维 NumPy 数组。

* **结构特征**：
    * **灰度图**：2维数组 `(行/高, 列/宽)`。
    * **彩色图**：3维数组 `(行/高, 列/宽, 通道数)`。
* **坐标系**：`image[y, x]`，其中 `y` 代表行（高度，从顶部开始），`x` 代表列（宽度，从左侧开始）。
* **字节操作**：可以将图像数组转换为 Python 的 `bytearray`，也可以将随机字节流重塑（reshape）为图像。

**代码示例：生成随机噪声图像**
```python
import cv2
import numpy
import os

# 生成 120,000 个随机字节
randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = numpy.array(randomByteArray)

# 重塑为 400x300 的灰度图 (2维)
grayImage = flatNumpyArray.reshape(300, 400)
cv2.imwrite('RandomGray.png', grayImage)

# 重塑为 400x100 的彩色图 (3维 BGR)
bgrImage = flatNumpyArray.reshape(100, 400, 3)
cv2.imwrite('RandomColor.png', bgrImage)
```

#### 3. 像素访问与 ROI 操作 (2.2.3)
利用 NumPy 的特性进行高效的像素操作。

* **属性查看**：
    * `img.shape`：返回 `(高, 宽, 通道数)`。
    * `img.size`：像素总数（彩色图为 像素数×3）。
    * `img.dtype`：数据类型（通常为 `uint8`，即 0-255）。
* **像素读写**：
    * **直接索引**：`img[0, 0] = [255, 255, 255]` (语法简单但效率较低)。
    * **优化方法**：使用 `img.item(y, x, channel)` 读取，`img.itemset((y, x, channel), value)` 写入 (效率更高)。
* **感兴趣区域 (ROI)**：使用数组切片操作图像块。
    * 可以将图像的一部分（ROI）赋值给变量，或将其复制/覆盖到图像的另一位置。

**代码示例：ROI 操作**
```python
import cv2
img = cv2.imread('MyPic.png')
# 定义 ROI (0到100行, 0到100列)
my_roi = img[0:100, 0:100]
# 将 ROI 复制到图像的另一区域
img[300:400, 300:400] = my_roi
```

#### 4. 视频文件读写与摄像头捕获 (2.2.4 - 2.2.7)
视频处理依赖 `VideoCapture` 和 `VideoWriter` 类。

* **读取/捕获**：使用 `cv2.VideoCapture()`。
    * 参数为**文件名**（如 `'video.avi'`）时读取文件。
    * 参数为**设备索引**（如 `0`）时调用摄像头。
* **写入视频**：使用 `cv2.VideoWriter()`。
    * 需指定文件名、编码器（FourCC）、帧率和帧大小。
    * 常用编码：`cv2.VideoWriter_fourcc('I','4','2','0')` (未压缩YUV) 或 `'X','2','6','4'` (MPEG-4)。
* **窗口交互**：
    * `cv2.imshow()`：显示当前帧。
    * `cv2.waitKey(delay)`：等待键盘输入。
        * 参数为 `0`：无限等待。
        * 参数 `>0`：等待指定毫秒数（处理视频流时常用，如 `cv2.waitKey(1)`）。
    * `cv2.setMouseCallback()`：绑定鼠标事件回调函数。

**代码示例：摄像头捕获与窗口交互**
```python
import cv2

clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

# 打开摄像头 (索引0)
cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow', onMouse)

print('显示摄像头画面。点击窗口或按键停止。')
success, frame = cameraCapture.read()

# 循环读取帧，直到读取失败、按键或鼠标点击
while success and cv2.waitKey(1) == -1 and not clicked:
    cv2.imshow('MyWindow', frame)
    success, frame = cameraCapture.read()

cv2.destroyWindow('MyWindow')
cameraCapture.release()
```

### 三、 实验总结
通过本次实验，理解了 OpenCV 中图像数据的本质是 NumPy 数组，这使得我们可以利用 Python 强大的科学计算库对图像像素进行高效操作（如切片处理 ROI）。同时，掌握了 `VideoCapture` 和 `VideoWriter` 的使用，实现了从静态图像处理到动态视频流处理的跨越，并了解了基于 `waitKey` 和回调函数的简单 GUI 交互机制。



## 实验三：图像处理

### 一、 实验目的
1.  **掌握卷积与自定义核**：理解卷积矩阵（核）的概念，学会使用 `cv2.filter2D` 实现锐化、边缘检测、模糊及浮雕等效果。
2.  **掌握边缘检测**：熟悉 Canny 边缘检测算法的原理及 OpenCV 实现。
3.  **掌握轮廓检测与形状分析**：学会查找和绘制轮廓，计算轮廓的边界框、最小外接矩形、最小外接圆、凸包及多边形拟合。
4.  **掌握霍夫变换**：学会使用霍夫变换检测图像中的直线（`HoughLinesP`）和圆（`HoughCircles`）。

### 二、 实验内容与步骤

#### 1. 自定义核与卷积 (3.6)
卷积滤波器通过一个“核”（Kernel，也称卷积矩阵）对图像像素及其邻域进行加权求和。OpenCV 提供了通用的 `cv2.filter2D()` 函数来应用任意核。

* **常用核示例**：
    * **锐化 (Sharpen)**：增强中心像素与其邻域的差异。
    * **边缘检测 (Find Edges)**：使边缘变白，非边缘变黑（权值和通常为0）。
    * **模糊 (Blur)**：邻域像素求平均（权值和通常为1）。
    * **浮雕 (Emboss)**：一边模糊（正权值），一边锐化（负权值），产生立体感。

**代码示例：应用锐化滤波器**
```python
import cv2
import numpy as np

img = cv2.imread("input.jpg")
# 定义锐化核
kernel_sharpen = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
# 应用滤波器 (深度为-1表示与原图一致)
dst = cv2.filter2D(img, -1, kernel_sharpen)
cv2.imshow("Sharpen", dst)
cv2.waitKey()
cv2.destroyAllWindows()
```

#### 2. Canny 边缘检测 (3.8)
Canny 算法是一种流行的多级边缘检测算法，包含高斯去噪、梯度计算、非极大值抑制、双阈值筛选和边缘连接五个步骤。在 OpenCV 中仅需一行代码即可实现。

* **函数**：`cv2.Canny(image, threshold1, threshold2)`
* **参数**：`threshold1` 和 `threshold2` 分别为最小和最大阈值，用于滞后阈值处理。

**代码示例**
```python
import cv2
img = cv2.imread("statue_small.jpg", 0) # 读取为灰度图
# 运行 Canny 算法
edges = cv2.Canny(img, 200, 300)
cv2.imshow("Canny Edges", edges)
```

#### 3. 轮廓检测与形状描述 (3.9)
轮廓检测用于提取物体边界，常在二值化图像（如阈值处理或 Canny 检测后的图像）上进行。

* **查找与绘制**：
    * `cv2.findContours(image, mode, method)`：返回轮廓列表和层次结构。常用 `cv2.RETR_EXTERNAL`（只检索外轮廓）和 `cv2.CHAIN_APPROX_SIMPLE`（压缩水平/垂直/对角线段）。
    * `cv2.drawContours(image, contours, index, color, thickness)`：在图像上绘制轮廓。
* **形状描述子 (3.9.1)**：
    * **边界框 (Bounding Box)**：`x, y, w, h = cv2.boundingRect(c)`。
    * **最小外接矩形 (Min Area Rect)**：`rect = cv2.minAreaRect(c)`，可处理旋转矩形。
    * **最小外接圆 (Min Enclosing Circle)**：`(x, y), radius = cv2.minEnclosingCircle(c)`。
* **多边形拟合与凸包 (3.9.2)**：
    * **多边形拟合**：`cv2.approxPolyDP(cnt, epsilon, True)`。`epsilon` 指定近似精度（通常取周长的百分比，如 `0.01 * cv2.arcLength`）。
    * **凸包 (Convex Hull)**：`cv2.convexHull(cnt)`，找出包围轮廓的最小凸多边形。

**代码示例：轮廓与外接形状**
```python
import cv2
import numpy as np

img = cv2.imread("hammer.jpg", cv2.IMREAD_UNCHANGED)
# 预处理：灰度 -> 阈值化
ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# 查找轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # 1. 绘制边界框 (绿色)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 2. 绘制最小外接圆 (红色)
    (cx, cy), radius = cv2.minEnclosingCircle(c)
    cv2.circle(img, (int(cx), int(cy)), int(radius), (0, 0, 255), 2)
    
    # 3. 计算凸包
    hull = cv2.convexHull(c)
```

#### 4. 霍夫变换检测线条与圆 (3.10)
霍夫变换（Hough Transform）用于检测可以用数学公式表达的几何形状。

* **检测直线 (HoughLinesP)**：
    使用概率霍夫变换检测线段。
    * `cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)`
    * 输入通常是 Canny 边缘检测后的二值图。
* **检测圆 (HoughCircles)**：
    * `cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)`
    * `minDist`：检测到的圆心之间的最小距离。
    * `param1`：Canny 边缘检测的高阈值。
    * `param2`：圆心检测的累加器阈值（越小检测到的圆越多，误检也越多）。

**代码示例：霍夫直线检测**
```python
import cv2
import numpy as np

img = cv2.imread('lines.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 120)

# 概率霍夫变换
lines = cv2.HoughLinesP(edges, 1, np.pi/180.0, 20, minLineLength=20, maxLineGap=5)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

### 三、 实验总结
通过本次实验，我们深入了解了图像处理中的核心操作。利用卷积核，我们能够实现图像增强和特效；Canny 算法提供了稳健的边缘提取能力；轮廓检测和形状拟合技术（如外接矩形、凸包）是进行物体识别和测量的基础；而霍夫变换则为从杂乱背景中提取几何图元（线、圆）提供了有效手段。这些技术共同构成了计算机视觉应用的重要基石。





## 实验四：特征检测与匹配

### 一、 实验目的
1.  **理解特征概念**：掌握角点、边缘、斑点等图像特征的定义。
2.  **掌握特征检测算法**：学会使用 Harris 角点检测，以及 SIFT、SURF、ORB 等关键点检测与描述符提取算法。
3.  **掌握特征匹配技术**：能够使用蛮力匹配器（BFMatcher）和 FLANN 匹配器进行特征匹配。
4.  **优化匹配结果**：学会使用 KNN（K近邻）匹配与劳氏比率检验（Lowe's Ratio Test）过滤误匹配。
5.  **应用单应性（Homography）**：利用匹配点计算单应性矩阵，实现物体定位与透视变换。

### 二、 实验内容与步骤

#### 1. Harris 角点检测 (6.3)
Harris 算法通过检测窗口在各个方向移动时的像素强度变化来寻找角点。它对旋转具有不变性，但对尺度变化敏感。

* **核心函数**：`cv2.cornerHarris(gray, blockSize, ksize, k)`
    * `blockSize`：邻域大小。
    * `ksize`：Sobel 算子的孔径参数。
    * `k`：Harris 检测器的自由参数。

**代码示例**
```python
import cv2
import numpy as np

img = cv2.imread('chess_board.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行 Harris 角点检测
# block_size=2, ksize=23 (Sobel孔径), k=0.04
dst = cv2.cornerHarris(gray, 2, 23, 0.04)

# 阈值筛选并绘制角点（红色）
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```





## 实验四：特征检测与匹配

### 一、 实验目的
1.  **理解特征概念**：掌握角点、边缘、斑点等图像特征的定义。
2.  **掌握特征检测算法**：学会使用 Harris 角点检测，以及 SIFT、SURF、ORB 等关键点检测与描述符提取算法。
3.  **掌握特征匹配技术**：能够使用蛮力匹配器（BFMatcher）和 FLANN 匹配器进行特征匹配。
4.  **优化匹配结果**：学会使用 KNN（K近邻）匹配与劳氏比率检验（Lowe's Ratio Test）过滤误匹配。
5.  **应用单应性（Homography）**：利用匹配点计算单应性矩阵，实现物体定位与透视变换。

### 二、 实验内容与步骤

#### 1. Harris 角点检测 (6.3)
Harris 算法通过检测窗口在各个方向移动时的像素强度变化来寻找角点。它对旋转具有不变性，但对尺度变化敏感。

* **核心函数**：`cv2.cornerHarris(gray, blockSize, ksize, k)`
    * `blockSize`：邻域大小。
    * `ksize`：Sobel 算子的孔径参数。
    * `k`：Harris 检测器的自由参数。

**代码示例**
```python
import cv2
import numpy as np

img = cv2.imread('chess_board.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行 Harris 角点检测
# block_size=2, ksize=23 (Sobel孔径), k=0.04
dst = cv2.cornerHarris(gray, 2, 23, 0.04)

# 阈值筛选并绘制角点（红色）
# 选取响应值大于最大响应值 1% 的点
img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 2. SIFT/SURF 特征检测 (6.4 - 6.5)
为了解决 Harris 算法不具备尺度不变性的问题，引入了 SIFT (尺度不变特征变换) 和 SURF (加速稳健特征)。它们能检测关键点并计算描述符。
*注意：SIFT 和 SURF 是专利算法，在 OpenCV 中需要安装 `opencv-contrib-python` 并在 CMake 中开启 `OPENCV_ENABLE_NONFREE` 标志才可使用。*

* **流程**：创建检测器对象 -> `detectAndCompute` -> `drawKeypoints`。
* **关键点属性**：坐标 (`pt`)、直径 (`size`)、方向 (`angle`)、响应强度 (`response`) 等。

**代码示例 (SIFT)**
```python
import cv2

img = cv2.imread('varese.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建 SIFT 检测器
sift = cv2.xfeatures2d.SIFT_create()
# 检测关键点并计算描述符
keypoints, descriptors = sift.detectAndCompute(gray, None)

# 绘制关键点 (使用 RICH_KEYPOINTS 标志绘制大小和方向)
cv2.drawKeypoints(img, keypoints, img, (51, 163, 236),
                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('SIFT Keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 3. ORB 特征检测 (6.6)
ORB (Oriented FAST and Rotated BRIEF) 是 SIFT 和 SURF 的高效免费替代方案。它结合了 FAST 关键点检测器和 BRIEF 描述符，并具备旋转不变性。

* **特点**：速度快，适合实时应用，不受专利限制。
* **核心函数**：`cv2.ORB_create()`。

#### 4. 特征匹配 (6.6.3 - 6.8)
计算出两幅图像的描述符后，需要进行匹配。
* **蛮力匹配 (Brute-Force Matcher)**：`cv2.BFMatcher`。对查询集中的每个描述符，在训练集中穷举搜索距离最近的描述符。
* **FLANN 匹配**：`cv2.FlannBasedMatcher`。快速近似最近邻搜索，适合大数据集。

**代码示例：ORB + 蛮力匹配**
```python
import cv2
from matplotlib import pyplot as plt

img0 = cv2.imread('nasa_logo.png', 0) # 查询图
img1 = cv2.imread('kennedy_space_center.jpg', 0) # 场景图

# ORB 检测
orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

# 蛮力匹配 (Hamming 距离适合二进制描述符如 ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des0, des1)

# 按距离排序并绘制前 25 个匹配
matches = sorted(matches, key=lambda x: x.distance)
img_matches = cv2.drawMatches(img0, kp0, img1, kp1, matches[:25], None,
                              flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img_matches)
plt.show()
```

#### 5. 匹配过滤与单应性查找 (6.7 - 6.9)
直接匹配通常包含大量误匹配。使用 KNN (k=2) 结合**劳氏比率检验 (Lowe's Ratio Test)** 可以有效过滤。随后利用 RANSAC 算法计算**单应性矩阵 (Homography)**，定位物体。

* **比率检验**：若 `m.distance < 0.7 * n.distance` (最佳匹配距离明显优于次优匹配)，则认为是好匹配。
* **单应性**：`cv2.findHomography`，需要至少 4 个点（通常用 10 个以上好匹配点）。

**代码示例：KNN 匹配 + 比率检验 + 单应性**
```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ... (假设已加载图像并计算了 SIFT/SURF 描述符 des0, des1) ...

# FLANN 匹配参数
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# KNN 匹配 (k=2)
matches = flann.knnMatch(des0, des1, k=2)

# 1. 应用比率检验保存好匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 2. 计算单应性并绘制边界框 (若好匹配数 > 10)
if len(good_matches) > 10:
    # 获取关键点坐标
    src_pts = np.float32([kp0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵 M
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 映射边界框
    h, w = img0.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    # 在场景图中绘制边框
    img1_polylines = cv2.polylines(img1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # 绘制匹配连线
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask.ravel().tolist(), flags=2)
    img_homography = cv2.drawMatches(img0, kp0, img1_polylines, kp1, good_matches, None, **draw_params)
    plt.imshow(img_homography)
    plt.show()
```

### 三、 实验总结
本实验深入探讨了局部特征的提取与匹配。我们发现 Harris 算法虽简单但无法应对尺度变化；SIFT 和 SURF 鲁棒性强但计算复杂且受专利限制；ORB 则提供了良好的速度与性能平衡。在匹配阶段，直接的蛮力匹配往往包含噪声，而结合 KNN 的比率检验能显著提升准确率。最终，通过计算单应性矩阵，我们成功实现了在复杂场景中对目标物体的精确定位和透视变换，这是图像拼接、物体识别等高级视觉任务的基础。