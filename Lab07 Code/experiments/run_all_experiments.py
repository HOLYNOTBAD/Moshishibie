#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行所有实验的脚本
Script to run all experiments
"""

import os
import sys
import subprocess

def run_experiment(exp_script, exp_name):
    """运行单个实验"""
    print(f"\n{'='*20} Running {exp_name} {'='*20}")

    try:
        # 确保results文件夹存在
        os.makedirs('../results', exist_ok=True)

        # 使用subprocess运行实验脚本
        result = subprocess.run([sys.executable, exp_script],
                              capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"{exp_name} completed successfully!")
        else:
            print(f"{exp_name} failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

    except subprocess.TimeoutExpired:
        print(f"{exp_name} timed out")
    except Exception as e:
        print(f"{exp_name} error: {e}")

def main():
    """主函数"""
    print("OpenCV实验系列 - 批量运行脚本")
    print("OpenCV Experiments Series - Batch Run Script")
    print("=" * 60)

    # 确保在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 创建示例图像
    print("首先创建示例图像...")
    run_experiment('create_sample_images.py', '创建示例图像')

    # 运行各个实验
    experiments = [
        ('exp1_setup_verification.py', '实验一：环境验证'),
        ('exp2_image_operations.py', '实验二：图像文件操作'),
        ('exp3_image_processing.py', '实验三：图像处理'),
        ('exp4_feature_matching.py', '实验四：特征检测与匹配')
    ]

    for script, name in experiments:
        if os.path.exists(script):
            run_experiment(script, name)
        else:
            print(f"警告: {script} 不存在")
            print(f"Warning: {script} not found")

    print("\n" + "=" * 60)
    print("所有实验运行完成！")
    print("请查看 results/ 文件夹中的输出结果。")
    print("All experiments completed!")
    print("Check the results/ folder for output.")

if __name__ == "__main__":
    main()
