#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一：OpenCV安装与环境搭建 - 环境验证脚本
Lab 1: OpenCV Installation and Environment Setup - Verification Script

此脚本用于验证OpenCV环境是否正确安装和配置
This script verifies if OpenCV environment is correctly installed and configured
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_opencv_version():
    """检查OpenCV版本"""
    print("=== OpenCV Version Check ===")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Python version: {sys.version}")
    print()

def create_test_image():
    """创建测试图像"""
    print("=== Creating Test Image ===")
    # 创建一个彩色渐变图像
    rows, cols = 300, 400
    img = np.zeros((rows, cols, 3), dtype=np.uint8)

    # 创建彩色渐变
    for i in range(rows):
        for j in range(cols):
            img[i, j] = [i*255//rows, j*255//cols, 128]  # BGR格式

    return img

def test_basic_operations(img):
    """测试基本图像操作"""
    print("=== Basic Image Operations Test ===")

    # 显示图像信息
    print(f"Image shape: {img.shape}")
    print(f"Image data type: {img.dtype}")
    print(f"Image size: {img.size} bytes")

    # 测试颜色空间转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print(f"Grayscale shape: {gray.shape}")
    print(f"HSV shape: {hsv.shape}")

    # 测试图像保存
    cv2.imwrite('results/test_image.jpg', img)
    cv2.imwrite('results/test_gray.jpg', gray)
    print("Image save test completed")

    return gray, hsv

def test_video_capture():
    """测试视频捕获功能"""
    print("\n=== Video Capture Test ===")

    try:
        # 尝试打开摄像头
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            print("Camera opened successfully")

            # 读取一帧
            ret, frame = cap.read()
            if ret:
                print(f"Successfully read frame, shape: {frame.shape}")
                cv2.imwrite('results/camera_test.jpg', frame)
                print("Camera test image saved")
            else:
                print("Cannot read frame")

            cap.release()
        else:
            print("Cannot open camera")

    except Exception as e:
        print(f"Camera test failed: {e}")

def test_opencv_contrib():
    """测试OpenCV contrib模块"""
    print("\n=== OpenCV Contrib Module Test ===")

    try:
        # 测试SIFT（需要opencv-contrib）
        sift = cv2.SIFT_create()
        print("SIFT is available")

        # 测试ORB
        orb = cv2.ORB_create()
        print("ORB is available")

        # 测试SURF (可能会因专利限制不可用)
        try:
            surf = cv2.xfeatures2d.SURF_create()
            print("SURF is available")
        except cv2.error as e:
            print(f"SURF is not available (patented algorithm): {str(e)[:100]}...")

    except AttributeError as e:
        print(f"Some contrib features not available: {e}")
        print("Make sure opencv-contrib-python is installed")

def create_visualization():
    """创建可视化效果"""
    print("\n=== Visualization Test ===")

    # 创建测试图像
    img = create_test_image()

    # 显示图像
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale')
    plt.axis('off')

    plt.subplot(133)
    # 应用简单的滤波器
    blurred = cv2.GaussianBlur(img, (15, 15), 0)
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('results/visualization_test.png', dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形避免显示问题
    print("Visualization results saved to results/visualization_test.png")

def main():
    """主函数"""
    print("OpenCV Environment Verification Script")
    print("=" * 60)

    # 确保输出目录存在
    os.makedirs('results', exist_ok=True)

    # 检查OpenCV版本
    check_opencv_version()

    # 创建测试图像
    img = create_test_image()

    # 测试基本操作
    gray, hsv = test_basic_operations(img)

    # 测试视频捕获
    test_video_capture()

    # 测试contrib模块
    test_opencv_contrib()

    # 创建可视化
    create_visualization()

    print("\n" + "=" * 60)
    print("Environment verification completed! Check the output files in the results folder.")

if __name__ == "__main__":
    main()
