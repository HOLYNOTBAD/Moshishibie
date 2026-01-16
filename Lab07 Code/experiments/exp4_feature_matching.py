#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验四：特征检测与匹配
Lab 4: Feature Detection and Matching

包含Harris角点检测、SIFT/SURF/ORB特征检测与匹配等功能
Includes Harris corner detection, SIFT/SURF/ORB feature detection and matching, etc.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_test_images():
    """创建测试图像"""
    print("创建测试图像...")

    # 1. 创建棋盘格图像用于Harris角点检测
    chessboard = np.zeros((300, 400), dtype=np.uint8)

    # 绘制棋盘格
    square_size = 40
    for i in range(0, 300, square_size):
        for j in range(0, 400, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                chessboard[i:i+square_size, j:j+square_size] = 255

    cv2.imwrite('resources/chessboard.jpg', chessboard)

    # 2. 创建一个简单的物体图像用于特征匹配测试
    object_img = np.zeros((200, 200, 3), dtype=np.uint8)

    # 绘制一个简单的形状
    cv2.rectangle(object_img, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(object_img, (100, 100), 25, (0, 0, 255), -1)
    cv2.line(object_img, (75, 75), (125, 125), (255, 0, 0), 3)

    # 添加一些纹理
    for _ in range(50):
        x, y = np.random.randint(0, 200, 2)
        cv2.circle(object_img, (x, y), 1, (0, 255, 0), -1)

    cv2.imwrite('resources/object.jpg', object_img)

    # 3. 创建场景图像（包含物体）
    scene_img = np.zeros((300, 400, 3), dtype=np.uint8)

    # 复制物体到场景中（添加变换）
    # 旋转和平移
    M = cv2.getRotationMatrix2D((100, 100), 30, 0.8)
    rotated_object = cv2.warpAffine(object_img, M, (200, 200))

    # 放置到场景中
    scene_img[50:250, 100:300] = rotated_object[0:200, 0:200]

    # 添加背景噪声
    noise = np.random.normal(0, 20, scene_img.shape).astype(np.uint8)
    scene_img = cv2.add(scene_img, noise)

    cv2.imwrite('resources/scene.jpg', scene_img)

    return chessboard, object_img, scene_img

def demonstrate_harris_corners():
    """演示Harris角点检测"""
    print("\n=== Harris Corner Detection ===")

    # 读取或创建棋盘格图像
    if os.path.exists('resources/chessboard.jpg'):
        img = cv2.imread('resources/chessboard.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        img, _, _ = create_test_images()

    # 应用Harris角点检测
    dst = cv2.cornerHarris(img, blockSize=2, ksize=23, k=0.04)

    # 结果归一化到0-255范围
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 阈值筛选角点
    threshold = 0.01 * dst_norm.max()
    corner_img = img.copy()

    # 在原图上标记角点
    corner_img[dst_norm > threshold] = 255

    # 也可以用彩色标记
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color_img[dst_norm > threshold] = [0, 0, 255]  # 红色标记角点

    # 保存结果
    cv2.imwrite('results/harris_corners.jpg', corner_img)
    cv2.imwrite('results/harris_corners_color.jpg', color_img)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Harris Corner Detection', fontsize=14)

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(dst_norm, cmap='hot')
    axes[1].set_title('Harris Response')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Detected Corners')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('results/harris_demo.png', dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形避免显示问题

    print("Harris corner detection completed")
    return color_img

def demonstrate_sift_surf_features():
    """演示SIFT/SURF特征检测"""
    print("\n=== SIFT/SURF Feature Detection ===")

    # 读取物体图像
    if os.path.exists('resources/object.jpg'):
        img = cv2.imread('resources/object.jpg')
    else:
        _, img, _ = create_test_images()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = {}

    try:
        # SIFT检测
        print("Trying SIFT feature detection")
        sift = cv2.SIFT_create()
        keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)

        # 在图像上绘制关键点
        sift_img = cv2.drawKeypoints(img, keypoints_sift, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        results['SIFT'] = {
            'keypoints': keypoints_sift,
            'descriptors': descriptors_sift,
            'image': sift_img
        }

        cv2.imwrite('results/sift_keypoints.jpg', sift_img)
        print(f"SIFT detected {len(keypoints_sift)} keypoints")

    except AttributeError:
        print("SIFT not available, may need to install opencv-contrib-python")

    try:
        # SURF检测 (可能会因专利限制不可用)
        print("Trying SURF feature detection")
        surf = cv2.xfeatures2d.SURF_create()
        keypoints_surf, descriptors_surf = surf.detectAndCompute(gray, None)

        # 在图像上绘制关键点
        surf_img = cv2.drawKeypoints(img, keypoints_surf, None,
                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        results['SURF'] = {
            'keypoints': keypoints_surf,
            'descriptors': descriptors_surf,
            'image': surf_img
        }

        cv2.imwrite('results/surf_keypoints.jpg', surf_img)
        print(f"SURF detected {len(keypoints_surf)} keypoints")

    except (AttributeError, cv2.error) as e:
        print(f"SURF not available (patented algorithm): {str(e)[:100]}...")

    # 可视化比较
    if results:
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 5))
        fig.suptitle('SIFT/SURF Feature Detection Comparison', fontsize=14)

        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original')
        axes[0].axis('off')

        for i, (name, data) in enumerate(results.items()):
            axes[i + 1].imshow(cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB))
            axes[i + 1].set_title(f'{name}\n({len(data["keypoints"])} keypoints)')
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.savefig('results/sift_surf_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形避免显示问题

    return results

def demonstrate_orb_features():
    """演示ORB特征检测"""
    print("\n=== ORB Feature Detection ===")

    # 读取物体图像
    if os.path.exists('resources/object.jpg'):
        img = cv2.imread('resources/object.jpg')
    else:
        _, img, _ = create_test_images()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ORB检测
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # 绘制关键点
    orb_img = cv2.drawKeypoints(img, keypoints, None,
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 保存结果
    cv2.imwrite('results/orb_keypoints.jpg', orb_img)

    print(f"ORB detected {len(keypoints)} keypoints")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ORB Feature Detection', fontsize=14)

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(orb_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'ORB keypoints\n({len(keypoints)} keypoints)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('results/orb_demo.png', dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形避免显示问题

    return keypoints, descriptors

def demonstrate_feature_matching():
    """演示特征匹配"""
    print("\n=== Feature Matching ===")

    # 读取物体和场景图像
    if os.path.exists('resources/object.jpg') and os.path.exists('resources/scene.jpg'):
        object_img = cv2.imread('resources/object.jpg', cv2.IMREAD_GRAYSCALE)
        scene_img = cv2.imread('resources/scene.jpg', cv2.IMREAD_GRAYSCALE)
    else:
        _, object_img, scene_img = create_test_images()
        object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
        scene_img = cv2.cvtColor(scene_img, cv2.COLOR_BGR2GRAY)

    # ORB特征检测
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(object_img, None)
    kp2, des2 = orb.detectAndCompute(scene_img, None)

    print(f"Object image: {len(kp1)} keypoints")
    print(f"Scene image: {len(kp2)} keypoints")

    # 1. 蛮力匹配
    print("\n1. Brute Force Matching")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_bf = bf.match(des1, des2)
    matches_bf = sorted(matches_bf, key=lambda x: x.distance)

    print(f"BF matching found {len(matches_bf)} matches")

    # 2. FLANN匹配 + KNN + 比率检验
    print("\n2. FLANN + KNN + Ratio Test")

    # FLANN参数
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                       table_number=6,
                       key_size=12,
                       multi_probe_level=1)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # KNN匹配 (k=2)
    matches_knn = flann.knnMatch(des1, des2, k=2)

    # 应用比率检验
    good_matches = []
    for m, n in matches_knn:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    print(f"{len(good_matches)} good matches after ratio test")

    # 绘制匹配结果
    # BF匹配结果
    bf_result = cv2.drawMatches(object_img, kp1, scene_img, kp2, matches_bf[:25], None,
                               flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 过滤后的匹配结果
    knn_result = cv2.drawMatches(object_img, kp1, scene_img, kp2, good_matches, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    # 保存结果
    cv2.imwrite('results/bf_matching.jpg', bf_result)
    cv2.imwrite('results/knn_ratio_matching.jpg', knn_result)

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Matching Comparison', fontsize=14)

    axes[0].imshow(bf_result)
    axes[0].set_title(f'BF Matching (Top 25)')
    axes[0].axis('off')

    axes[1].imshow(knn_result)
    axes[1].set_title(f'KNN+Ratio Test\n({len(good_matches)} matches)')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('results/matching_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()  # 关闭图形避免显示问题

    return good_matches, kp1, kp2

def demonstrate_homography():
    """演示单应性变换和物体定位"""
    print("\n=== Homography and Object Localization ===")

    # 读取图像
    if os.path.exists('resources/object.jpg') and os.path.exists('resources/scene.jpg'):
        object_img = cv2.imread('resources/object.jpg', cv2.IMREAD_GRAYSCALE)
        scene_img = cv2.imread('resources/scene.jpg', cv2.IMREAD_GRAYSCALE)
        scene_color = cv2.imread('resources/scene.jpg')
    else:
        _, object_img, scene_color = create_test_images()
        object_img = cv2.cvtColor(object_img, cv2.COLOR_BGR2GRAY)
        scene_img = cv2.cvtColor(scene_color, cv2.COLOR_BGR2GRAY)

    # 获取之前的匹配结果
    good_matches, kp1, kp2 = demonstrate_feature_matching()

    if len(good_matches) > 10:
        # 提取匹配点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 获取物体角点
        h, w = object_img.shape
        obj_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        # 变换到场景坐标
        scene_corners = cv2.perspectiveTransform(obj_corners, M)

        # 在场景图像上绘制边界框
        result_img = scene_color.copy()
        cv2.polylines(result_img, [np.int32(scene_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)

        # 也可以绘制匹配连线（只绘制inlier点）
        matches_mask = mask.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0),
                          singlePointColor=None,
                          matchesMask=matches_mask,
                          flags=2)

        final_result = cv2.drawMatches(object_img, kp1, result_img, kp2, good_matches, None, **draw_params)

        # 保存结果
        cv2.imwrite('results/homography_localization.jpg', result_img)
        cv2.imwrite('results/homography_matches.jpg', final_result)

        print("Homography computed, object localized successfully")

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Homography and Object Localization', fontsize=14)

        axes[0].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Object Bounding Box')
        axes[0].axis('off')

        axes[1].imshow(final_result)
        axes[1].set_title('Matches & Localization')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig('results/homography_demo.png', dpi=150, bbox_inches='tight')
        plt.close()  # 关闭图形避免显示问题

    else:
        print("Not enough matches for homography computation")

def create_comprehensive_demo():
    """创建综合演示"""
    print("\n=== 实验四综合演示 / Lab 4 Comprehensive Demo ===")

    # 确保有测试图像
    create_test_images()

    # 运行所有演示
    harris_result = demonstrate_harris_corners()
    sift_surf_results = demonstrate_sift_surf_features()
    orb_kp, orb_des = demonstrate_orb_features()
    demonstrate_homography()

    print("\n实验四所有功能演示完成！")
    print("All Lab 4 functions demonstrated!")

if __name__ == "__main__":
    print("实验四：特征检测与匹配 / Lab 4: Feature Detection and Matching")
    print("=" * 60)

    # 确保目录存在
    os.makedirs('resources', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # 运行综合演示
    create_comprehensive_demo()

    print("\n" + "=" * 60)
    print("实验四完成！请查看results文件夹中的结果。")
    print("Lab 4 completed! Check the results folder for outputs.")
