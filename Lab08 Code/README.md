Lab08 Code — 第七章示例集合

包含：
- detect_people_hog.py  （HOG 行人检测示例）
- detect_car_bow_svm.py （BoW + SVM 车辆检测示例）
- detect_car_bow_svm_sliding_window.py （BoW + SVM + 金字塔 + 滑窗 + NMS）
- non_max_suppression.py （非极大值抑制实现）
- requirements.txt （依赖列表）
- setup_lab08.ps1 （在本目录准备数据与图片的 PowerShell 脚本）

运行前准备
1. 安装 Python 3.7+ 和依赖：
   在 PowerShell 中运行：

   pip install -r requirements.txt

2. 准备数据与示例图片：
   - 脚本需要 `CarData` 数据集（训练/测试样本）。请在本目录运行 `setup_lab08.ps1`，它会尝试下载并解压 CarData，如果下载/解压失败，会提示手动操作。
   - 示例还会读取仓库中的 `images` 目录下的若干图片；`setup_lab08.ps1` 会尝试将这些图片复制到本目录的 `images` 子目录。

如何运行
- 行人检测（HOG）：
  python detect_people_hog.py

- 车辆 BoW+SVM（单张测试）：
  python detect_car_bow_svm.py

- 车辆 BoW+SVM（滑窗金字塔检测）：
  python detect_car_bow_svm_sliding_window.py

注意事项
- SIFT: 请确保安装的 OpenCV 版本 >= 4.4.0（requirements 中已指定）。旧版本可能需要 opencv-contrib 或特殊构建。
- CarData: 数据集较大，下载需网络连接；脚本在找不到 `CarData` 时会退出并提示。
- 性能: 滑窗 + 金字塔 + SIFT 在 CPU 上会很慢。可缩小 `BOW_NUM_CLUSTERS`、增大滑窗步长或在小图上实验以加快速度。

故障排查
- 如果看到 `CarData folder not found`，请手动下载并解压到 `Lab08 Code\CarData`：
  https://github.com/gcr/arc-evaluator/raw/master/CarData.tar.gz

- 如果 SIFT 创建失败，尝试升级 opencv-python：
  pip install --upgrade opencv-python

- Windows PowerShell 解压 tar.gz 可能需要系统自带的 `tar` 或 7-Zip。README 内有替代说明。
