#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建实验用的示例图像
Create sample images for experiments
"""

import cv2
import numpy as np
import os

def create_basic_shapes():
    """创建基本形状图像"""
    print("创建基本形状图像...")

    # 创建一个400x400的彩色图像
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # 绘制各种形状
    # 1. 矩形
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)  # 蓝色填充矩形
    cv2.rectangle(img, (200, 50), (350, 150), (0, 255, 0), 3)   # 绿色边框矩形

    # 2. 圆形
    cv2.circle(img, (100, 250), 50, (0, 0, 255), -1)   # 红色填充圆
    cv2.circle(img, (250, 250), 30, (255, 255, 0), 3)  # 青色边框圆

    # 3. 椭圆
    cv2.ellipse(img, (100, 350), (60, 30), 0, 0, 360, (255, 0, 255), -1)  # 品红填充椭圆
    cv2.ellipse(img, (250, 350), (40, 60), 45, 0, 270, (0, 255, 255), 3)  # 黄色边框椭圆

    # 4. 线条和多边形
    # 三角形
    pts = np.array([[50, 200], [100, 180], [80, 220]], np.int32)
    cv2.fillPoly(img, [pts], (128, 128, 128))

    # 五角星
    star_pts = np.array([[320, 200], [340, 230], [370, 230], [350, 250],
                        [360, 280], [320, 270], [280, 280], [290, 250],
                        [270, 230], [300, 230]], np.int32)
    cv2.fillPoly(img, [star_pts], (255, 255, 255))

    # 5. 添加文字
    cv2.putText(img, 'OpenCV Shapes', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    cv2.imwrite('resources/basic_shapes.jpg', img)
    print("基本形状图像已创建")
    return img

def create_chessboard():
    """创建棋盘格图像用于角点检测"""
    print("创建棋盘格图像...")

    # 创建400x400的灰度图像
    img = np.zeros((400, 400), dtype=np.uint8)

    # 绘制棋盘格
    square_size = 50
    for i in range(0, 400, square_size):
        for j in range(0, 400, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                img[i:i+square_size, j:j+square_size] = 255

    cv2.imwrite('resources/chessboard.jpg', img)
    print("棋盘格图像已创建")
    return img

def create_gradient_pattern():
    """创建渐变图案图像"""
    print("创建渐变图案图像...")

    # 创建400x400的彩色图像
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # 水平渐变
    for i in range(400):
        for j in range(400):
            img[i, j] = [i*255//400, j*255//400, 128]

    cv2.imwrite('resources/gradient.jpg', img)
    print("渐变图案图像已创建")
    return img

def create_texture_image():
    """创建纹理图像"""
    print("创建纹理图像...")

    # 创建400x400的灰度图像
    img = np.zeros((400, 400), dtype=np.uint8)

    # 添加随机噪声纹理
    np.random.seed(42)  # 固定种子保证结果一致
    noise = np.random.normal(128, 50, (400, 400)).astype(np.uint8)
    img = cv2.add(img, noise)

    # 添加一些结构
    for _ in range(20):
        # 随机圆
        center = (np.random.randint(0, 400), np.random.randint(0, 400))
        radius = np.random.randint(5, 30)
        color = np.random.randint(0, 255)
        cv2.circle(img, center, radius, color, -1)

    cv2.imwrite('resources/texture.jpg', img)
    print("纹理图像已创建")
    return img

def create_line_patterns():
    """创建线条图案用于霍夫变换测试"""
    print("创建线条图案...")

    # 创建400x400的灰度图像
    img = np.zeros((400, 400), dtype=np.uint8)

    # 绘制各种线条
    # 1. 水平线
    cv2.line(img, (50, 50), (350, 50), 255, 2)
    cv2.line(img, (50, 150), (350, 150), 255, 2)

    # 2. 垂直线
    cv2.line(img, (50, 50), (50, 350), 255, 2)
    cv2.line(img, (150, 50), (150, 350), 255, 2)

    # 3. 斜线
    cv2.line(img, (50, 250), (350, 50), 255, 2)
    cv2.line(img, (50, 350), (350, 250), 255, 2)

    # 4. 随机线条
    np.random.seed(123)
    for _ in range(10):
        pt1 = (np.random.randint(200, 400), np.random.randint(200, 400))
        pt2 = (np.random.randint(200, 400), np.random.randint(200, 400))
        cv2.line(img, pt1, pt2, 255, 1)

    cv2.imwrite('resources/line_patterns.jpg', img)
    print("线条图案图像已创建")
    return img

def create_object_and_scene():
    """创建物体和场景图像用于特征匹配测试"""
    print("创建物体和场景图像...")

    # 1. 创建物体图像
    object_img = np.zeros((200, 200, 3), dtype=np.uint8)

    # 绘制一个独特的物体
    # 主体矩形
    cv2.rectangle(object_img, (50, 50), (150, 150), (255, 255, 255), -1)

    # 添加特征标记
    cv2.circle(object_img, (100, 100), 20, (0, 0, 255), -1)
    cv2.line(object_img, (75, 75), (125, 125), (255, 0, 0), 3)
    cv2.line(object_img, (125, 75), (75, 125), (255, 0, 0), 3)

    # 添加一些随机特征点
    np.random.seed(456)
    for _ in range(30):
        x, y = np.random.randint(0, 200, 2)
        cv2.circle(object_img, (x, y), 1, (0, 255, 0), -1)

    cv2.imwrite('resources/object.jpg', object_img)

    # 2. 创建场景图像（包含变形的物体）
    scene_img = np.zeros((400, 500, 3), dtype=np.uint8)

    # 复制物体到场景中（添加变换）
    # 旋转
    M = cv2.getRotationMatrix2D((100, 100), 25, 0.9)
    rotated_object = cv2.warpAffine(object_img, M, (200, 200))

    # 放置到场景中
    scene_img[100:300, 150:350] = rotated_object

    # 添加背景噪声和干扰
    np.random.seed(789)
    noise = np.random.normal(0, 15, scene_img.shape).astype(np.uint8)
    scene_img = cv2.add(scene_img, noise)

    # 添加一些干扰物体
    for _ in range(5):
        center = (np.random.randint(0, 500), np.random.randint(0, 400))
        cv2.circle(scene_img, center, np.random.randint(10, 30),
                  (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), -1)

    cv2.imwrite('resources/scene.jpg', scene_img)

    print("物体和场景图像已创建")
    return object_img, scene_img

def create_all_samples():
    """创建所有示例图像"""
    print("开始创建所有示例图像...")
    print("=" * 50)

    # 确保目录存在
    os.makedirs('resources', exist_ok=True)

    # 创建各种示例图像
    basic_shapes = create_basic_shapes()
    chessboard = create_chessboard()
    gradient = create_gradient_pattern()
    texture = create_texture_image()
    lines = create_line_patterns()
    object_img, scene_img = create_object_and_scene()

    print("=" * 50)
    print("所有示例图像创建完成！")
    print("All sample images created!")

    # 显示创建的图像信息
    print("\n创建的图像文件:")
    print("- resources/basic_shapes.jpg - 基本形状")
    print("- resources/chessboard.jpg - 棋盘格")
    print("- resources/gradient.jpg - 渐变图案")
    print("- resources/texture.jpg - 纹理图像")
    print("- resources/line_patterns.jpg - 线条图案")
    print("- resources/object.jpg - 测试物体")
    print("- resources/scene.jpg - 场景图像")

if __name__ == "__main__":
    create_all_samples()
