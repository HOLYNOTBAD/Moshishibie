#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验二：图像文件操作
Lab 2: Image File Operations

包含图像读取、写入、像素操作、ROI处理、视频处理等功能
Includes image reading, writing, pixel operations, ROI processing, video processing, etc.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def create_sample_images():
    """创建示例图像用于测试"""
    print("创建示例图像...")

    # 创建一个彩色测试图像
    img = np.zeros((200, 300, 3), dtype=np.uint8)

    # 绘制一些几何形状
    cv2.rectangle(img, (50, 50), (250, 150), (0, 255, 0), -1)  # 绿色矩形
    cv2.circle(img, (150, 100), 30, (255, 0, 0), -1)  # 蓝色圆形
    cv2.line(img, (100, 75), (200, 125), (0, 0, 255), 3)  # 红色线条

    # 添加文字
    cv2.putText(img, 'OpenCV Test', (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imwrite('resources/sample_image.jpg', img)
    return img

def demonstrate_image_io():
    """演示图像读取和写入操作"""
    print("\n=== Image File Reading and Writing ===")

    # 创建示例图像
    img = create_sample_images()

    # 1. 以不同模式读取图像
    print("1. Different reading modes:")
    color_img = cv2.imread('resources/sample_image.jpg', cv2.IMREAD_COLOR)
    gray_img = cv2.imread('resources/sample_image.jpg', cv2.IMREAD_GRAYSCALE)
    unchanged_img = cv2.imread('resources/sample_image.jpg', cv2.IMREAD_UNCHANGED)

    print(f"Color image shape: {color_img.shape}")
    print(f"Grayscale image shape: {gray_img.shape}")
    print(f"Unchanged image shape: {unchanged_img.shape}")

    # 2. 保存图像
    print("\n2. Saving images:")
    cv2.imwrite('results/sample_gray.jpg', gray_img)
    cv2.imwrite('results/sample_png.png', color_img)
    print("Images saved successfully")

    # 3. 格式转换示例
    print("\n3. Format conversion example:")
    # 将彩色图像转换为灰度并保存
    gray_converted = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('results/converted_gray.jpg', gray_converted)
    print("Format conversion completed")

    return color_img, gray_img

def demonstrate_image_structure():
    """演示图像数据结构"""
    print("\n=== 2.2.2 图像表示与字节转换 / Image Representation and Byte Conversion ===")

    img = cv2.imread('resources/sample_image.jpg')

    # 1. 图像属性
    print("1. 图像属性 / Image properties:")
    print(f"形状 (高, 宽, 通道): {img.shape}")
    print(f"数据类型: {img.dtype}")
    print(f"像素总数: {img.size}")
    print(f"图像维度: {img.ndim}")

    # 2. 坐标系说明
    print("\n2. 坐标系 / Coordinate system:")
    print("img[y, x] - y是行(高度)，x是列(宽度)")
    print(f"左上角像素值: {img[0, 0]} (BGR格式)")
    print(f"中心像素值: {img[100, 150]} (BGR格式)")

    # 3. 字节操作
    print("\n3. 字节操作 / Byte operations:")
    # 转换为字节数组
    byte_array = bytearray(img)
    print(f"字节数组长度: {len(byte_array)}")

    # 从随机字节创建图像
    random_bytes = bytearray(os.urandom(180000))  # 300x200x3 = 180000
    flat_array = np.array(random_bytes)
    random_img = flat_array.reshape(200, 300, 3)
    cv2.imwrite('results/random_noise.jpg', random_img)
    print("随机噪声图像已创建 / Random noise image created")

    return img

def demonstrate_pixel_operations():
    """演示像素访问和ROI操作"""
    print("\n=== 2.2.3 像素访问与ROI操作 / Pixel Access and ROI Operations ===")

    img = cv2.imread('resources/sample_image.jpg')

    # 1. 像素访问方法
    print("1. 像素访问方法 / Pixel access methods:")

    # 方法1: 直接索引 (较慢)
    pixel_direct = img[100, 150]
    print(f"直接索引 - 像素值: {pixel_direct}")

    # 方法2: 使用item() (较快)
    b = img.item(100, 150, 0)  # 蓝色通道
    g = img.item(100, 150, 1)  # 绿色通道
    r = img.item(100, 150, 2)  # 红色通道
    print(f"item()方法 - BGR: ({b}, {g}, {r})")

    # 2. 修改像素值
    print("\n2. 修改像素值 / Modifying pixel values:")
    img_copy = img.copy()
    img_copy[50:100, 50:100] = [255, 0, 0]  # 将区域设置为蓝色
    cv2.imwrite('results/modified_pixels.jpg', img_copy)
    print("像素修改完成 / Pixel modification completed")

    # 3. ROI操作
    print("\n3. ROI操作 / ROI operations:")
    roi = img[50:150, 100:200]  # 提取感兴趣区域
    print(f"ROI形状: {roi.shape}")

    # 将ROI复制到另一个位置
    img_with_roi = img.copy()
    img_with_roi[0:100, 0:100] = roi
    cv2.imwrite('results/roi_copy.jpg', img_with_roi)
    print("ROI复制完成 / ROI copy completed")

    return img

def demonstrate_video_operations():
    """演示视频操作"""
    print("\n=== 2.2.4-2.2.7 视频文件读写与摄像头捕获 / Video File Operations and Camera Capture ===")

    # 注意：这个函数包含交互式内容，在实际运行时需要用户参与
    print("注意：视频操作包含交互式内容，请根据需要运行相应部分")
    print("Note: Video operations include interactive content, run relevant parts as needed")

    # 1. 摄像头捕获示例代码
    print("\n1. 摄像头捕获示例代码 / Camera capture example code:")
    print("""
# 摄像头捕获代码 (请单独运行)
import cv2

def camera_capture_demo():
    clicked = False

    def on_mouse(event, x, y, flags, param):
        global clicked
        if event == cv2.EVENT_LBUTTONUP:
            clicked = True

    camera = cv2.VideoCapture(0)
    cv2.namedWindow('Camera')
    cv2.setMouseCallback('Camera', on_mouse)

    print('点击窗口或按键停止 / Click window or press key to stop')
    success, frame = camera.read()

    while success and cv2.waitKey(1) == -1 and not clicked:
        cv2.imshow('Camera', frame)
        success, frame = camera.read()

    cv2.destroyWindow('Camera')
    camera.release()

# camera_capture_demo()
    """)

    # 2. 视频写入示例代码
    print("\n2. 视频写入示例代码 / Video writing example code:")
    print("""
# 视频写入代码 (请单独运行)
import cv2

def video_write_demo():
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('results/output.avi', fourcc, 20.0, (640, 480))

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 翻转帧 (可选)
            frame = cv2.flip(frame, 1)

            # 写入帧
            out.write(frame)

            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# video_write_demo()
    """)

def create_comprehensive_demo():
    """创建综合演示"""
    print("\n=== 综合演示 / Comprehensive Demo ===")

    # 创建测试图像
    img = create_sample_images()

    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Lab 2: Image File Operations Demo', fontsize=16)

    # 原始图像
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # 灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale')
    axes[0, 1].axis('off')

    # ROI演示
    roi = img[50:150, 100:200].copy()
    axes[0, 2].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('ROI Region')
    axes[0, 2].axis('off')

    # 像素修改演示
    modified = img.copy()
    modified[75:125, 125:175] = [255, 0, 0]  # 蓝色方块
    axes[1, 0].imshow(cv2.cvtColor(modified, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Pixel Modification')
    axes[1, 0].axis('off')

    # 颜色通道分离
    b, g, r = cv2.split(img)
    axes[1, 1].imshow(b, cmap='Blues')
    axes[1, 1].set_title('Blue Channel')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(g, cmap='Greens')
    axes[1, 2].set_title('Green Channel')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('results/exp2_comprehensive_demo.png', dpi=150, bbox_inches='tight')
    plt.close()  # close figure to avoid GUI/font issues

    print("综合演示图像已保存 / Comprehensive demo image saved")

def main():
    """主函数"""
    print("Lab 2: Image File Operations")
    print("=" * 60)

    # 确保资源目录存在
    os.makedirs('resources', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 演示各个功能
    color_img, gray_img = demonstrate_image_io()
    demonstrate_image_structure()
    demonstrate_pixel_operations()
    demonstrate_video_operations()
    create_comprehensive_demo()

    print("\n" + "=" * 60)
    print("实验二完成！请查看results文件夹中的结果。")
    print("Lab 2 completed! Check the results folder for outputs.")

if __name__ == "__main__":
    main()
