在 conda 下为 Lab08 创建 Python 3.9 环境（Windows PowerShell 指南）

建议：使用 conda-forge 的包来确保 OpenCV 包含 SIFT/额外模块（通常 conda-forge 的 opencv 包包含 contrib 功能）。下面给出创建、激活、验证的步骤。

1) 打开 PowerShell（管理员权限通常不需要，但确保 conda 已初始化到 PowerShell）：
   conda init powershell ; 新开一个 PowerShell 窗口以使更改生效

2) 在 `Lab08 Code` 目录下创建环境：

```powershell
cd "g:\root\Moshishibie\Lab08 Code"
conda env create -f environment.yml
```

3) 激活环境：

```powershell
conda activate lab08-py39
```

4)（可选）如果你偏好 pip 安装 requirements.txt 内的包：

```powershell
pip install -r requirements.txt
```

5) 验证 OpenCV 和 SIFT 是否可用：

```powershell
python - <<'PY'
import cv2
import sys
print('cv2.__version__ =', cv2.__version__)
# 尝试创建 SIFT
try:
    sift = cv2.SIFT_create()
    print('SIFT available')
except Exception as e:
    print('SIFT not available:', e)
# 简单检查能否导入 numpy
import numpy as np
print('numpy okay, version:', np.__version__)
PY
```

6) 运行示例（激活环境后）：

```powershell
python detect_people_hog.py
python detect_car_bow_svm.py
python detect_car_bow_svm_sliding_window.py
```

注意与建议：
- 如果 `conda env create` 报错找不到满足依赖的 opencv 版本，可把 `opencv>=4.4` 改为 `opencv=4.5` 或 `opencv=4.6`，或直接运行 `conda install -c conda-forge opencv`。
- 如果你更愿意完全用 pip（例如在虚拟环境或需要最新 pip 包），可以创建 conda 环境只指定 Python，然后用 `pip install -r requirements.txt`。但不要同时用 conda 安装一个 opencv 包又用 pip 安装另一个 opencv 版本（会产生冲突）。
- CarData 数据集仍需用 `setup_lab08.ps1` 下载/解压，或手动放置到 `Lab08 Code\CarData`。

故障排查提示：
- 如果 SIFT 报错：尝试 `conda install -c conda-forge opencv`，然后重试验证脚本；若仍然失败，可改用 `pip install opencv-contrib-python`（注意 pip 包会覆盖 conda 的 opencv，需要小心）。
- 若 PowerShell 中 `conda` 命令不可用，确认 Miniconda/Anaconda 已安装并且已运行过 `conda init powershell`，或直接用 Anaconda Prompt 来执行命令。

## 已知 pip/numpy 冲突：你刚才看到的错误说明

如果你在 `pip install -r requirements.txt` 时看到类似：

  "Successfully installed numpy-2.2.6 ... ERROR: pip's dependency resolver ... scipy 1.11.2 requires numpy<1.28.0,>=1.21.6, but you have numpy 2.2.6 which is incompatible."

说明当前（可能是 `base`）环境用 pip 安装了与已安装包（如 SciPy）不兼容的 numpy 版本。通常原因是：
- 你在系统/`base` 环境里直接运行了 `pip install -r requirements.txt`，而该 requirements.txt 包含 `opencv-python` 与 `numpy`，pip 会下载与当前 Python 版本相匹配的 wheel（例如 Python 3.12 下会安装 numpy 2.x），这可能与已安装的其他包产生冲突。

推荐的安全修复流程（优先级由高到低）：

1) 推荐做法 — 在干净的 conda 环境中创建并安装（不破坏 base）：

```powershell
cd "g:\root\Moshishibie\Lab08 Code"
conda env create -f environment.yml
conda activate lab08-py39
```

这会安装 `numpy=1.26`、`scipy` 和 conda-forge 的 `opencv`，避免 numpy 2.x 与 SciPy 的冲突。

2) 如果你已经在某个 conda 环境中并希望修复当前环境（不推荐在 base 上操作）：

```powershell
# 激活目标环境（例如 base 或 lab08-py39）
conda activate <env-name>
# 使用 conda 强制安装兼容的 numpy 版本（优于 pip）
conda install -c conda-forge numpy=1.26 scipy
# 然后用 conda 安装 opencv
conda install -c conda-forge opencv
```

3) 如果你已经用 pip 安装了不兼容的 numpy 到 base 环境，最简单稳妥的办法是不要尝试把 base 修复为开发环境，而是创建一个新环境（见步骤 1）。如果你确实要在 base 上降级：

```powershell
pip uninstall numpy
pip install "numpy<1.28,>=1.21.6"
```

注意：在 conda 环境中混用 conda 安装和 pip 安装 opencv/numpy 可能导致冲突，因此优先用 `conda install -c conda-forge opencv numpy`。仅当某个包必须通过 pip 安装时，再在激活的 conda 环境中使用 pip 安装单独包（但不要覆盖 numpy/opencv）。

---

如果你愿意，我现在可以在终端执行下面的其中一个操作并把输出贴回给你：
- A) 用 `environment.yml` 创建新的 `lab08-py39` 环境（推荐）。
- B) 在当前（你指定的）环境中用 conda 安装 `numpy=1.26` 和 `opencv`（需要你告诉我环境名）。
- C) 仅运行检查命令，报告当前 Python 环境里 cv2/numpy/SIFT 的状态（不改变任何东西）。

请选择 A / B / C，或告诉我你希望我先做哪一步。