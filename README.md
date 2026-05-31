# Face Classify Studio

Face Classify Studio 是一个基于 **Jupyter Notebook + PyTorch + ResNet18 迁移学习** 的 7 类人脸种族识别实验项目，并附带 macOS 本地桌面端流程，用于单张图片推理、概率展示、结果导出和测试人员错误反馈。

本仓库的源码已经统一改为 **Jupyter Notebook 实现**。主入口是：

```text
Face_Classify_Studio.ipynb
```

![Desktop feedback preview](reports/desktop_app_feedback_preview.png)

## 功能亮点

- 全部项目代码以 `.ipynb` Notebook 形式提供
- 7 分类输出：`Black`、`East Asian`、`Indian`、`Latino_Hispanic`、`Middle Eastern`、`Southeast Asian`、`White`
- ResNet18 ImageNet 预训练迁移学习
- 两阶段训练：先冻结主干训练分类头，再低学习率微调后层
- macOS MPS 自动加速，无法使用时回落 CPU
- Notebook 支持训练、测试评估、单图推理和反馈微调
- 桌面端 Notebook 支持图片预览、Top-1 预测、7 类概率条、CSV 导出和错误反馈

![Desktop app preview](reports/desktop_app_preview.png)

## 项目结构

```text
face_classify/
├── Face_Classify_Studio.ipynb       # Jupyter 主入口，完整实验流程
├── code/                            # 所有功能模块均为 Notebook
│   ├── train_resnet18.ipynb         # ResNet18 两阶段训练
│   ├── inference.ipynb              # 推理封装
│   ├── predict_one.ipynb            # 单图预测
│   ├── desktop_app.ipynb            # 桌面端界面
│   ├── train_feedback.ipynb         # 反馈样本微调
│   ├── import_fairface.ipynb        # FairFace 数据导入
│   ├── audit_dataset_quality.ipynb  # 数据质量检查
│   ├── supplement_fairface_quality.ipynb
│   └── sweep_resnet18.ipynb         # 参数实验记录
├── model/
│   ├── best_resnet18_faces.pth      # 当前默认模型
│   └── training_metrics.json        # 当前训练指标
├── reports/
│   ├── desktop_app_preview.png
│   └── desktop_app_feedback_preview.png
├── requirements.txt
├── run_jupyter.command              # 一键启动主 Notebook
├── setup_and_run.command            # 一键安装依赖并打开主 Notebook
└── run_desktop_app.command          # 打开桌面端 Notebook
```

## 一键部署 Notebook

macOS 用户可以直接双击：

```text
run_jupyter.command
```

这个脚本会自动完成：

1. 创建本地虚拟环境 `.venv`
2. 安装 `requirements.txt` 中的依赖
3. 打开 `Face_Classify_Studio.ipynb`

如果 macOS 提示脚本没有执行权限，可以在项目目录运行：

```bash
chmod +x run_jupyter.command setup_and_run.command run_desktop_app.command
```

## 手动启动

```bash
python3 -m pip install -r requirements.txt
python3 -m notebook Face_Classify_Studio.ipynb
```

## Notebook 内容

`Face_Classify_Studio.ipynb` 包含：

1. 环境与依赖配置
2. `ImageFolder` 数据集加载
3. 训练集数据增强与验证/测试标准化
4. ResNet18 迁移学习模型构建
5. 第一阶段冻结骨干训练分类头
6. 第二阶段低学习率微调后层
7. 测试集分类报告与混淆矩阵
8. 单张图片 Softmax 概率推理
9. 测试人员反馈样本微调
10. 从 Notebook 启动桌面端 Notebook

## 数据集格式

数据集使用 PyTorch `ImageFolder` 结构：

```text
data/
├── train/
│   ├── Black/
│   ├── East Asian/
│   ├── Indian/
│   ├── Latino_Hispanic/
│   ├── Middle Eastern/
│   ├── Southeast Asian/
│   └── White/
├── val/
└── test/
```

注意：本仓库不发布原始训练图片。请自行准备符合授权要求的数据集。

## 运行桌面端 Notebook

可以双击：

```text
run_desktop_app.command
```

或在 Jupyter 中打开：

```text
code/desktop_app.ipynb
```

桌面端 Notebook 默认加载：

```text
model/best_resnet18_faces.pth
```

## 反馈闭环

桌面端预测完成后，如果测试人员发现结果错误，可以点击 `预测错误`：

1. 选择正确标签
2. 可选填写备注
3. 点击保存反馈

反馈样本会保存到：

```text
feedback/images/<正确标签>/
feedback/feedback_log.csv
```

累计一定数量反馈样本后，可以在 `Face_Classify_Studio.ipynb` 或 `code/train_feedback.ipynb` 中运行反馈微调。

## 当前模型表现

当前版本使用小样本迁移学习训练，最终测试集整体准确率约为 `36.43%`，macro F1 约为 `35.32%`。该项目更适合作为完整机器学习流程、Notebook 实验记录和桌面端原型展示，而不是直接用于高风险真实场景。

## 说明

本项目仅用于课程、实验和原型开发。人脸属性识别涉及公平性、隐私和伦理风险，请勿用于身份判断、自动化决策或任何可能影响个人权益的场景。
