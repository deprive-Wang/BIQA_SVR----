# BCQI 在 LIVEC 上的复现

复现论文 [Toward a No-Reference Quality Metric for Camera-Captured Images](https://doi.org/10.1109/TCYB.2021.3128023) 中的低层感知特征、高层语义特征与 RBF SVR 方法。

当前只研究 LIVEC，不包含 CID2013 和跨数据库实验。已实现数据读取、7 维低层特征、1000 维 SqueezeNet 语义特征、1007 维拼接及 RBF SVR 训练评估。语义提取默认要求 CUDA，scikit-learn SVR 使用 CPU。

## 完成范围与后续工作

当前已实现特征提取、训练评估和结果绘图入口；尚无可作为完整实验结论的正式结果。单轮开发验证只用于检查流程，不能作为论文复现结果引用，生成产物不纳入 Git。

- 已实现：正式样本筛选与标签对齐、低层/语义特征提取、缓存一致性校验、训练集内部调参、模型保存、原始与 logistic 映射后四项指标、多轮均值汇总、单图原始分数推理、BRISQUE 基线提取及同划分显著性比较入口。
- 已提供：1000 次随机划分入口，以及低层、语义、拼接三种特征的消融入口；入口可用不代表对应实验已经执行。
- 尚未完成：PWRC 指标、1000 轮正式实验及消融结果，以及论文表 II 的完整基线方法集合。当前权重、预处理和数值选择以本项目记录的 Python 实现为准。
- 不在当前范围：CID2013、跨数据库实验、网络微调。单图预测已提供命令行入口；当前仅输出原始 SVR 分数，不提供利用测试集 MOS 拟合的部署校准分数。

后续顺序为：固定当前复现配置 → 在相同划分下运行主实验、消融和基线 → 补齐 PWRC → 汇总均值、有效轮数及与论文的差异。不要依据测试集表现挑选参数或随机种子。

## 代码入口

| 文件 | 职责 |
| --- | --- |
| `livec.py` | 官方标注读取、练习图排除、数据完整性检查 |
| `low_level.py` / `extract_features.py` | 7 维低层特征计算与缓存（CPU） |
| `semantic.py` / `extract_semantic.py` | 预训练 SqueezeNet 1000 维特征与拼接缓存（默认 CUDA） |
| `train_svr.py` | 划分、训练内交叉验证、RBF SVR、消融和多轮产物保存（CPU） |
| `metrics.py` | 四项质量指标和五参数 logistic 事后映射 |
| `plot_results.py` | 完成轮数与产物一致性检查、指标分布、固定轮次诊断、同划分消融图 |
| `predict_image.py` | 使用可信的已保存模型，对单张原始 RGB 图像输出未经测试集 logistic 映射的 MOS 预测 |
| `compare_baselines.py` | 提取 BRISQUE 分数，与 BCQI 在同一测试图像上比较并做逐轮配对显著性检验 |

从项目根目录执行下文命令。已有且通过一致性检查的特征缓存可直接用于 SVR，不必每次重新提取。修改提取器、权重或预处理后应重新生成受影响的缓存，不要绕过源码/内容哈希检查。

讲解代码时按 `livec.py`（名称与 MOS 对齐）→ `extract_features.py` / `low_level.py`（每图 7 维）→ `extract_semantic.py` / `semantic.py`（每图 1000 维，拼成 1007 维）→ `train_svr.py` / `metrics.py`（划分、调参、预测、评估）→ `predict_image.py`（单图推理）→ `compare_baselines.py`（同图像比较）→ `plot_results.py`（读取已保存结果画图）的顺序看。提取器缓存会核对源码全文哈希；注释变化也会使旧缓存失效。

## 作者代码

论文提供的入口为 [YT2015 的 GitHub 仓库列表](https://github.com/YT2015?tab=repositories)。2026-09-20 检查时，未找到可确认的 BCQI 仓库；当前无法取得作者源码。项目继续使用已经记录参数和预处理选择的 Python 实现，不等待作者源码，也不将本项目代码或结果称为作者官方实现或精确复现。

## 本地数据

数据根目录为 `data/ChallengeDB_release/`：

- `Images/`：1162 张正式实验图像。
- `Images/trainingImages/`：7 张用于主观实验的受试者练习图，应排除。
- `Data/AllImages_release.mat`：1169 个名称。
- `Data/AllMOS_release.mat`：按名称列表顺序对应的 MOS。
- `Data/AllStdDev_release.mat`：同序评分标准差。

根据 MAT 中的名称匹配图像，使用同一掩码过滤名称和标签，不依赖文件系统枚举顺序。原始数据保持不变，遵守数据包自带 README 的许可与引用要求。

## 环境准备

目标为独立的 `BIQA_SVR` 环境，Python 3.11。新机器从项目根目录运行：

```powershell
& 'E:\Miniforge\Scripts\conda.exe' env create -f environment.yml
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' -m pip check
```

已存在该环境时不必重复创建。其他机器使用其实际 Conda 路径。`environment.yml` 引用 `requirements.txt`。版本范围不是作者原始环境或精确锁文件。GPU 版 PyTorch 应依据本机驱动与 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择匹配的 torch/torchvision 构建。

本机 GPU 安装组合（Python 3.11、Windows、RTX 3070 系列）：

```powershell
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' -m pip install torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' -m pip check
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); print((torch.ones(2, device='cuda') + 1).cpu())"
```

显式的 `+cu126` 用来替换 CPU 构建；安装源只影响当前命令。无需安装 torchaudio。普通依赖清单不决定 GPU 构建，重建环境后仍需执行上面的 GPU 安装命令。版本配对见 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

## 数据检查

```powershell
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' livec.py
```

`livec.py --root <数据根目录>` 可检查其他位置的官方数据包。默认全量解码正式实验图像，打印数量和 MOS 范围，失败时返回非零退出码，不修改原始数据。

其他模块可通过 `from livec import load_livec` 调用；返回按 MAT 原始顺序排列的 `LivecSample` 元组，包含名称、绝对路径、MOS、评分标准差及原始标注索引。

## 低层特征

`low_level.py` 对照论文式 (3)-(16) 和表 I，输出顺序为：亮度、饱和度、J-S 对比度、噪声方差、小波锐度、GGD 的 alpha 和 beta，共 7 维。输入为原始 RGB uint8 图像，不缩放。

```powershell
# 全量提取；不覆盖已有文件，重复实验时指定不同输出名。
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' extract_features.py --output features/livec_low_level_v2.npz
```

NPZ 使用 `allow_pickle=False` 即可读取，包含 `features`、`names`、`mos`、`stddev`、`annotation_indices`、`image_sha256`、`feature_names` 和 JSON 字符串 `metadata`。缓存记录配置、源码 SHA-256、依赖版本和是否为调试子集，不进行全数据标准化。

实现边界：以下为本项目采用的 Python 数值选择，并非已核验的作者默认值；尤其是噪声估计。后续实验沿用这些已记录的选择，结果应注明配置。

- HSI 亮度采用 RGB 均值并归一化到 [0,1]，饱和度为 `1 - min(R,G,B)/mean(R,G,B)`；黑像素饱和度设为 0。其他属性采用浮点 BT.601 灰度 [0,255]，不执行 EXIF 旋转或 ICC 色彩转换。
- 对比度采用四舍五入灰度的 256 档概率直方图，与均匀分布计算自然对数 J-S **散度**，不取平方根。
- 噪声使用 7×7 正交 DCT-II 全部 48 个 AC 滤波器，只保留 valid 响应，使用总体中心矩。对式 (8) 的绝对残差目标，用加权中位数消去共同峰度，再做 129 点网格及局部有界优化。约束方差非负且小于最小响应方差、共同 Pearson 峰度至少为 1；网格搜索不保证任意输入的数学全局最优。输出为方差，不是标准差；不能视为相机真实噪声的可靠测量值。
- 锐度采用 `bior4.4` 的三级对称延拓分解（CDF 9/7），子带权重为 0.1/0.1/0.8，细到粗权重为 4/7、2/7、1/7；滤波器归一化和边界相位尚未与作者代码比对。参考 [PyWavelets 文档](https://pywavelets.readthedocs.io/en/latest/regression/wavelet.html)。
- MSCN 使用 7×7、sigma=7/6 的 Gaussian 核与 reflect 延拓，分母为局部标准差加 1。GGD 以零均值模型的绝对一阶矩及二阶矩拟合，alpha 搜索区间为 [0.2,10]；beta 为式 (15) 的尺度参数，不是方差。
- 零方差 DCT 响应的噪声输出约定为 0；全零 MSCN 的 GGD 约定为 alpha=2、beta=0，此时形状本来不可辨识。图像最小边长至少 72，小图直接报错，不悄悄缩放。

## 高层语义与特征拼接

`semantic.py` 使用 torchvision SqueezeNet v1.1 的 `IMAGENET1K_V1` 权重，载入前验证 SHA-256。权重存放于 `checkpoints/squeezenet1_1-b8a52dc0.pth`，来自 [PyTorch 官方权重地址](https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth)。运行时不自动下载，也不会以随机权重代替。

`extract_semantic.py` 默认 CUDA、float32、batch size 16；无 CUDA 时明确报错，不自动回退。CPU 仅可通过 `--device cpu` 显式选择，供诊断使用。GPU 路径使用 pinned memory 传输、`eval()` 和 `inference_mode()`，不微调、不做混合精度。前向计算关闭 cuDNN TF32 和 benchmark，启用确定性卷积，降低不同批大小的数值差异；这些设置和批大小写入缓存。低层特征继续使用 SciPy/PyWavelets 的 CPU 实现，拼接时复用现有缓存，不重复提取。

```powershell
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' extract_semantic.py --output features/livec_combined_v2.npz --device cuda
```

提取前检查现有低层缓存的配置、源码哈希、标注和顺序，逐图核验内容哈希；任何错配都会终止。输出 NPZ 的 `features[:, :7]` 是低层特征，`features[:, 7:]` 是语义特征，维度为 (1162,1007)。附带名称、MOS、标准差、索引、图像哈希、权重哈希、预处理及设备信息。`--limit` 仅截取已验证完整低层缓存的前若干样本供调试；输出不覆盖已有文件。

按论文直接从原图裁取中心 227×227 区域，起点为 `floor((size-227)/2)`，不先 resize、不填充小图。输入除以 255 后采用该 torchvision 权重的 ImageNet mean/std。输出取 classifier 末尾全局平均池化的 1000 维激活，不应用 softmax。参见 [torchvision 源码](https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py)。

[MATLAB 文档](https://www.mathworks.com/help/deeplearning/ref/squeezenet.html) 确认其 SqueezeNet 为 v1.1、输入为 227×227；本项目使用的 torchvision 权重和归一化尚未与作者 MATLAB 实现对齐，结果属于明确记录差异的 Python 复现。

## SVR 训练与评估

`train_svr.py` 首先按图像划分外层训练/测试集，随后在训练集内部使用 3 折交叉验证，以 MSE 选择 RBF SVR 参数。每个内层训练折都单独拟合 `StandardScaler`，选定后仅在外层训练集上重拟合；不缩放 MOS，也不对预测做 [0,100] 截断。初始搜索范围为 C={1,10,100}、gamma={scale,0.001,0.01}、epsilon={0.1,1}，这是本项目选择，非作者确认参数。

```powershell
# 单次端到端验证，929 张训练 / 233 张测试；输出目录必须尚不存在。
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' train_svr.py --output outputs/svr_seed42_new --seed 42
# 多次划分入口；每轮独立调参，计算时间随重复次数增长。
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' train_svr.py --output outputs/svr_1000 --repeats 1000 --seed 42 --jobs 2
```

`--feature-set combined|low|semantic` 支持消融；相同 seed 与 repeats 产生相同图像划分。`--jobs` 控制内部交叉验证并行数，默认 1。重复实验默认只保存首轮模型，全部轮次均保存预测、划分及评估；需要每轮模型时显式指定 `--save-models all`，注意磁盘占用。

例如在同一组划分下运行低层和语义消融（以下为待执行命令，不表示已完成）：

```powershell
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' train_svr.py --output outputs/svr_low_1000 --feature-set low --repeats 1000 --seed 42 --jobs 2
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' train_svr.py --output outputs/svr_semantic_1000 --feature-set semantic --repeats 1000 --seed 42 --jobs 2
```

三个特征设置均从同一份完整拼接缓存选列，因此仅低层消融也需要先准备拼接缓存。固定其他参数并使用不同输出目录；不要覆盖已有实验。

结果目录包含 `config.json`、每轮的 `metrics.json`、`predictions.npz`、`predictions.csv`，以及最终 `summary.json`。配置保存依赖版本、源码与缓存哈希、特征来源和所有随机种子；每轮保存内外层划分索引、图像名、候选参数的 CV 分数与 logistic 参数。仅最终汇总的 `status=complete` 表示全部请求轮次完成；异常中断时已完成轮次保留，不自动覆盖或续跑。

`model.pkl` 保存 StandardScaler+SVR，可对相同列顺序的特征调用 `predict`，返回原始 MOS 预测。仅加载自己生成且可信的 pickle 文件。模型不包含测试集 logistic 校准；该映射只用于论文式事后评估，不用于部署或参数选择。

`metrics.py` 同时提供原始预测和五参数 logistic 映射后的 SRCC、KRCC、PLCC、RMSE。logistic 拟合先使用三个起点直接优化；如果全部达到计算上限或未收敛，则消去三个线性参数，对余下两个参数做确定性的有界搜索。常量相关系数、不可识别或失败的拟合明确记为 null/失败状态，不伪造成功值。汇总保留每项有效轮次数，避免失败被静默忽略。相关系数保留符号，不取绝对值。

已完成的 `outputs/svr_1000` 有 38 轮仅 logistic 映射失败，原始预测完整。下列命令只用保存的测试预测与 MOS 修复这些映射，复制成新目录；原实验不变，不重新训练 SVR。新目录含 `recalibration.json` 记录来源与修复轮次，并再次逐轮核验。

```powershell
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' recalibrate_run.py --run outputs/svr_1000 --output outputs/svr_1000_logistic_fixed
```

论文主实验为 80%/20% 随机划分、1000 次重复并报告均值。程序保存每轮种子和图像 ID；预处理参数与 SVR 超参数仅从训练集确定。小轮数用于调试，不能当作完整复现结果。

主要指标为 SRCC、KRCC、PLCC、RMSE，表 II 另有 PWRC（当前未实现），因此不能宣称完整复现表 II。五参数 logistic 映射与原始预测分别保留；网络权重、数值约定和调参范围与作者实现尚未确认一致。后续开发约定见 [AGENTS.md](AGENTS.md)。

## 单图质量预测

使用本项目生成且可信的 `model.pkl`。输入须为 RGB 图像：`combined` 或 `semantic` 模型要求最小边长 227 像素，`low` 模型要求最小边长 72 像素。低层属性从原图计算，语义分支按当前 SqueezeNet 预处理。默认使用 CUDA，诊断时可显式指定 `--device cpu`。输出是 SVR 的原始 MOS 预测，不使用依赖测试集 MOS 拟合的 logistic 曲线，也不强制截断到 [0,100]。

```powershell
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' predict_image.py --run outputs/svr_seed42_current --image 'data/ChallengeDB_release/Images/10.bmp' --device cuda
```

示例中的 `svr_seed42_current` 是单轮开发验证模型；正式使用时将模型目录替换为自己已完成的训练结果。默认读取第 0 轮；只有保存了对应轮次模型时才能使用 `--round` 选择其他轮次。输入图像若不是 RGB、尺寸不够、无法解码，或模型的特征维度和预处理配置不兼容，会明确报错。

## 基线比较与显著性检验

`compare_baselines.py` 在正式 LIVEC 图像上提取 BRISQUE 分数（由 [PIQ](https://github.com/photosynthesis-team/piq) `0.8.0` 提供），并记录图片及 BRISQUE 权重哈希。输入 PIQ 前使用整张原图、不预先缩放；PIQ 内部仍按 BRISQUE 算法处理双尺度。原始分数越小表示质量越高，保存的比较分数会取负以统一方向。首次运行会将 PIQ 提供的 BRISQUE SVR 权重下载到 `checkpoints/piq/`；后续复用。该预训练 BRISQUE 是可运行的比较基线，**不是**论文表 II 中按每轮训练集重新训练的 BRISQUE，因此不可直接声称复现该行数值。

```powershell
# 仅在缓存尚不存在时提取；输出文件不允许覆盖。
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' compare_baselines.py extract-brisque --output features/livec_brisque_piq_v1.npz --device cuda
# 完成 1000 轮实验后，复用上述缓存进行正式比较。
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' compare_baselines.py compare --run outputs/svr_1000_logistic_fixed --baseline features/livec_brisque_piq_v1.npz --output outputs/compare_brisque.json
```

比较程序要求基线和 BCQI 使用同一份完整特征缓存、1162 个相同图像 ID、MOS 与图像哈希；每轮只取相同的 233 张测试图像。基线分数与 BCQI 一样分别报告原始和测试集事后 logistic 映射的四项指标。显著性检验使用**同一轮、同一图像**的映射后绝对残差做双侧配对 t 检验，多基线时对该轮 p 值做 Holm 校正，显著性水平为 0.05；各轮独立报告，不将重复出现的图像跨轮合并。论文只写了对预测残差作 t 检验，未说明配对形式和多重比较处理；本实现是明确记录的项目协议，不能宣称与表 III 的检验完全一致。某轮任一模型的 logistic 拟合失败时，该轮不做配对检验，并保留有效轮数。

默认只接受完成 1000 轮的实验；少轮调试需添加 `--allow-debug`，报告也会标记为调试结果。比较入口可一次接收多个符合相同 NPZ 字段和数据校验规则的基线档案；目前内置提取器只有 BRISQUE。

本地已有的 BRISQUE 全量缓存可直接复用。若只需检查比较流程，可将上例的 `--run` 换为已完成的单轮实验目录，另外指定尚不存在的输出文件，并加上 `--allow-debug`；所得结果只用于调试，不作论文结论。

开发测试使用项目环境中的 `pytest`，新环境可先按测试依赖清单安装，再运行已有的基线比较与单图推理测试：

```powershell
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' -m pip install -r requirements-dev.txt
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' -m pytest -q tests
```

## 实验结果绘图

`plot_results.py` 只读取已有实验产物，不训练模型、不重新拟合 logistic，不加载 `model.pkl`。默认要求请求并完成 1000 轮，验证每轮 929/233 划分、图像 ID、MOS、保存的映射参数、预测与指标以及汇总一致性。完成 1000 轮仅表示满足重复次数要求，不代表已完成作者实现对齐或 PWRC。

绘图命令示例（绘图尚未执行）：

```powershell
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' plot_results.py --runs outputs/svr_1000_logistic_fixed --output outputs/figures_main
# 消融对比：三份实验必须具有相同缓存来源、种子、内外层划分和调参协议。
& 'E:\Miniforge\envs\BIQA_SVR\python.exe' plot_results.py --runs outputs/svr_low_1000 outputs/svr_semantic_1000 outputs/svr_1000_logistic_fixed --output outputs/figures_ablation
```

输出为 PNG、PDF，以及精确统计表 `metrics_summary.csv` 和来源记录 `plot_manifest.json`。两张指标图分别展示原始与映射后 SRCC/KRCC/PLCC/RMSE 的跨轮分布、均值与总体标准差，并显示有效轮数；标准差不是置信区间。每个方法另输出散点/残差/误差分布组合图及保存的 logistic 曲线，默认固定第 0 轮，仅作诊断，不把该轮当作总体结果，不合并各轮重复出现的图像计算一个总相关系数。

少轮数调试必须显式添加 `--allow-debug`，所有图均标记 `DEBUG` 和实际轮数。`--round` 可显式指定诊断轮次，默认 0；不要根据测试指标选择最好轮次。输出必须为项目 `outputs/` 下尚不存在的子目录，拒绝覆盖。没有成功的 logistic 拟合时显示不可用，不伪造映射结果。

绘图依赖 `matplotlib>=3.8,<4`。

现有特征缓存的源码哈希包含注释；源码注释更新后，重新训练前需重新生成低层与拼接缓存。绘图读取已完成实验自己的配置和预测，不依赖当前特征缓存，也不会更改历史来源记录。
