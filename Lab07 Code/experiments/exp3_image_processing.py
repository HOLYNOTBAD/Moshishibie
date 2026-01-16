#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验三：图像处理
Lab 3: Image Processing

包含卷积滤波、边缘检测、轮廓检测、霍夫变换等功能
Includes convolution filtering, edge detection, contour detection, Hough transform, etc.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_test_images():
    """创建测试图像"""
    print("创建测试图像...")

    # 1. 创建一个简单的几何形状图像用于轮廓检测
    img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 绘制各种形状
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)  # 白色矩形
    cv2.circle(img, (250, 100), 40, (255, 255, 255), -1)  # 白色圆形
    cv2.ellipse(img, (100, 220), (60, 30), 0, 0, 360, (255, 255, 255), -1)  # 白色椭圆

    # 添加一些噪声
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    cv2.imwrite('resources/shapes.jpg', img)

    # 2. 创建一个线条图像用于霍夫变换测试
    lines_img = np.zeros((300, 400), dtype=np.uint8)

    # 绘制直线
    cv2.line(lines_img, (50, 50), (350, 50), 255, 2)  # 水平线
    cv2.line(lines_img, (50, 50), (50, 250), 255, 2)  # 垂直线
    cv2.line(lines_img, (50, 250), (350, 50), 255, 2)  # 斜线

    # 添加一些随机线条
    for _ in range(5):
        pt1 = (np.random.randint(0, 400), np.random.randint(0, 300))
        pt2 = (np.random.randint(0, 400), np.random.randint(0, 300))
        cv2.line(lines_img, pt1, pt2, 255, 1)

    cv2.imwrite('resources/lines.jpg', lines_img)

    return img, lines_img

def demonstrate_convolution_filters():
    """演示自定义核与卷积操作"""
    print("\n=== 3.6 自定义核与卷积 / Custom Kernels and Convolution ===")

    # 读取或创建测试图像
    if os.path.exists('resources/shapes.jpg'):
        img = cv2.imread('resources/shapes.jpg')
    else:
        img, _ = create_test_images()

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 定义各种滤波器核
    kernels = {
        'identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # 恒等核
        'sharpen': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),  # 锐化核
        'edge_detect': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),  # 边缘检测核
        'blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,  # 模糊核
        'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])  # 浮雕核
    }

    results = {}
    for name, kernel in kernels.items():
        result = cv2.filter2D(gray, -1, kernel)
        results[name] = result
        cv2.imwrite(f'results/{name}_filter.jpg', result)
        print(f"{name}滤波器应用完成 / {name} filter applied")

    # 可视化比较
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Convolution Filter Comparison', fontsize=16)

    # 显示原始图像
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    # 显示各滤波结果
    filter_names = list(results.keys())[1:]  # 跳过identity
    for i, name in enumerate(filter_names):
        row, col = divmod(i + 1, 3)
        axes[row, col].imshow(results[name], cmap='gray')
        axes[row, col].set_title(name.capitalize())
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('results/convolution_filters_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results

def demonstrate_canny_edge_detection():
    """演示Canny边缘检测"""
    print("\n=== 3.8 Canny边缘检测 / Canny Edge Detection ===")

    # 读取测试图像
    if os.path.exists('resources/shapes.jpg'):
        img = cv2.imread('resources/shapes.jpg')
    else:
        img, _ = create_test_images()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 应用不同的Canny参数
    edges1 = cv2.Canny(gray, 100, 200)  # 较低阈值
    edges2 = cv2.Canny(gray, 200, 300)  # 较高阈值
    edges3 = cv2.Canny(gray, 50, 150)   # 更低阈值

    # 保存结果
    cv2.imwrite('results/canny_edges_100_200.jpg', edges1)
    cv2.imwrite('results/canny_edges_200_300.jpg', edges2)
    cv2.imwrite('results/canny_edges_50_150.jpg', edges3)

    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Canny Edge Detection Parameter Comparison', fontsize=14)

    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Original Grayscale')
    axes[0].axis('off')

    axes[1].imshow(edges1, cmap='gray')
    axes[1].set_title('Threshold: 100-200')
    axes[1].axis('off')

    axes[2].imshow(edges2, cmap='gray')
    axes[2].set_title('Threshold: 200-300')
    axes[2].axis('off')

    axes[3].imshow(edges3, cmap='gray')
    axes[3].set_title('Threshold: 50-150')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('results/canny_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Canny边缘检测完成 / Canny edge detection completed")
    return edges1

def demonstrate_contour_detection():
    """演示轮廓检测与形状分析"""
    print("\n=== 3.9 轮廓检测与形状描述 / Contour Detection and Shape Analysis ===")

    # 读取测试图像
    if os.path.exists('resources/shapes.jpg'):
        img = cv2.imread('resources/shapes.jpg')
    else:
        img, _ = create_test_images()

    # 转换为灰度并二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"找到 {len(contours)} 个轮廓 / Found {len(contours)} contours")

    # 在原始图像上绘制轮廓和形状描述
    contour_img = img.copy()

    for i, cnt in enumerate(contours):
        # 1. 边界框
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 2. 最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(contour_img, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)

        # 3. 凸包
        hull = cv2.convexHull(cnt)
        cv2.drawContours(contour_img, [hull], 0, (0, 0, 255), 2)

        # 4. 多边形拟合
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(contour_img, [approx], 0, (255, 255, 0), 2)

        # 显示轮廓信息
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        print(".1f"
              f"凸包顶点数: {len(hull)} / Convex hull vertices: {len(hull)}")

    # 保存结果
    cv2.imwrite('results/contours_analysis.jpg', contour_img)
    cv2.imwrite('results/threshold.jpg', thresh)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Contour Detection and Shape Analysis', fontsize=14)

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(thresh, cmap='gray')
    axes[1].set_title('Threshold')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Contours & Shape Analysis')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('results/contours_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("轮廓检测完成 / Contour detection completed")
    return contours, contour_img

def demonstrate_hough_transform():
    """演示霍夫变换"""
    print("\n=== 3.10 霍夫变换检测线条与圆 / Hough Transform for Lines and Circles ===")

    # 1. 直线检测
    print("1. 霍夫直线检测 / Hough Line Detection")

    # 读取线条图像
    if os.path.exists('resources/lines.jpg'):
        lines_img = cv2.imread('resources/lines.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        _, lines_img = create_test_images()

    # Canny边缘检测
    edges = cv2.Canny(lines_img, 50, 150, apertureSize=3)

    # 霍夫直线变换
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

    # 在原始图像上绘制检测到的直线
    lines_result = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(lines_result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print(f"检测到 {len(lines)} 条直线 / Detected {len(lines)} lines")

    # 2. 概率霍夫变换 (更常用)
    print("\n2. 概率霍夫变换 / Probabilistic Hough Transform")

    lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)

    lines_p_result = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2BGR)
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_p_result, (x1, y1), (x2, y2), (255, 0, 0), 2)

    print(f"概率霍夫变换检测到 {len(lines_p)} 条线段 / Probabilistic Hough detected {len(lines_p)} line segments")

    # 3. 圆检测
    print("\n3. 霍夫圆检测 / Hough Circle Detection")

    # 创建一个包含圆的图像用于测试
    circles_img = np.zeros((300, 400), dtype=np.uint8)
    cv2.circle(circles_img, (100, 100), 50, 255, 3)
    cv2.circle(circles_img, (250, 150), 30, 255, 3)
    cv2.circle(circles_img, (150, 250), 40, 255, 3)

    # 添加噪声
    noise = np.random.normal(0, 10, circles_img.shape).astype(np.uint8)
    circles_img = cv2.add(circles_img, noise)

    # 霍夫圆变换
    circles = cv2.HoughCircles(circles_img, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=50, param2=30, minRadius=20, maxRadius=60)

    circles_result = cv2.cvtColor(circles_img, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # 绘制外圆
            cv2.circle(circles_result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 绘制圆心
            cv2.circle(circles_result, (i[0], i[1]), 2, (0, 0, 255), 3)

    print(f"检测到 {len(circles[0]) if circles is not None else 0} 个圆 / Detected {len(circles[0]) if circles is not None else 0} circles")

    # 保存结果
    cv2.imwrite('results/hough_lines.jpg', lines_result)
    cv2.imwrite('results/hough_lines_p.jpg', lines_p_result)
    cv2.imwrite('results/hough_circles.jpg', circles_result)
    cv2.imwrite('resources/circles.jpg', circles_img)

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hough Transform Demonstration', fontsize=16)

    axes[0, 0].imshow(lines_img, cmap='gray')
    axes[0, 0].set_title('Lines Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(lines_result, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Standard Hough Lines')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(cv2.cvtColor(lines_p_result, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Probabilistic Hough Lines')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(circles_img, cmap='gray')
    axes[1, 0].set_title('Circles Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(circles_result, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Hough Circles')
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')  # 空占位符

    plt.tight_layout()
    plt.savefig('results/hough_transform_demo.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("霍夫变换演示完成 / Hough transform demonstration completed")

def create_comprehensive_demo():
    """创建综合演示"""
    print("\n=== 实验三综合演示 / Lab 3 Comprehensive Demo ===")

    # 确保有测试图像
    create_test_images()

    # 运行所有演示
    filters = demonstrate_convolution_filters()
    edges = demonstrate_canny_edge_detection()
    contours, contour_img = demonstrate_contour_detection()
    demonstrate_hough_transform()

    print("\n实验三所有功能演示完成！")
    print("All Lab 3 functions demonstrated!")

if __name__ == "__main__":
    print("实验三：图像处理 / Lab 3: Image Processing")
    print("=" * 60)

    # 确保目录存在
    os.makedirs('resources', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 运行综合演示
    create_comprehensive_demo()

    print("\n" + "=" * 60)
    print("实验三完成！请查看results文件夹中的结果。")
    print("Lab 3 completed! Check the results folder for outputs.")
