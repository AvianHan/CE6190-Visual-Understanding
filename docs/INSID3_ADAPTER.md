# INSID3 adapter：原版与组件组合

本页描述 adapter 合约及相对上游的协议差异。实验验收以当前报告和 `results_summary/` 为准；代码中的 `full` 指锁定官方源码的主要算法流程，不表示复现了论文全部条件。

## 输入、输出与用途

- 输入：query RGB 图像，以及一个或多个 reference RGB 图像和对应 binary masks。
- 输出：query **原始分辨率**的 binary mask，真实执行处统计的 encoder 调用次数／耗时，以及组件配置。
- 模型不接收 query GT、类别名、数据集名；这些信息留在评测程序中。
- 主比较用于回答两种完整系统的表现；本 adapter 的组件开关用于回答 clustering、aggregation、debiasing 与 encoder 的作用。

## 锁定源码

INSID3 commit：`0c165a10cf52ab91f335883d06260de86854adbe`。
构造 adapter 时校验核心源文件 SHA256；如果源文件变动，停止运行，要求重新审计，不静默继续。
官方源码保持不变；完整流程调用原版类，消融使用显式 subclass。

## 纸面描述与代码的区别

1. 论文描述从 Gaussian noise 提取位置子空间；此 commit 的 `_build_positional_basis` 实际输入 **全黑图像**，ImageNet normalize 后提取特征，逐位置中心化再做 SVD。默认忠实保留源码。
2. 原文 candidate localization 重点描述 backward correspondence；代码同时包含 **forward-positive similarity gate AND backward gate**。默认保留。
3. 源码 aggregation 使用 candidate 命中面积比例作为 cluster 权重；seed 权重设为 1，但最终依然经过阈值，**不是无条件把 seed 保留**。默认保留。
4. 官方 CLI 使用 `resize_to_orig_size=False`，随后将 query GT 缩放到预测尺寸。公共评测协议要求原始 query 分辨率，因此这里使用 `resize_to_orig_size=True`。这属于明确记录的 **统一评测适配**，不能将其分数直接声称为完全相同的作者评测协议。
5. `simple_similarity` 的 debiasing 是独立显式开关。不能未经核对就把其数值对应到论文 Table 3 的某一行。

## 配置

`create_adapter(config)` 接收 flat settings。路径必须是已下载的本地文件／源码目录。adapter 不联网下载模型。

| 字段 | 默认／用途 |
|---|---|
| `upstream_dir` | 必填：INSID3 本地 checkout |
| `checkpoint` | 必填：当前 encoder 的本地权重 |
| `encoder` | `dinov3`、`dinov2`、`sam3` |
| `dinov3_source` / `dinov2_source` / `sam3_source` | 当前 encoder 的本地源码；SAM3 使用 `CG-ICS/sam3`，其中包含 `sam3/model_builder.py` |
| `variant` | `full`、`no_debias`、`no_aggregation`、`cross_only`、`simple_similarity` |
| `use_debias` | 可独立组合；默认 `true` |
| `aggregation` | `self_cross`、`cross_only`、`seed_only` |
| `prediction_mode` | `clusters` 或 `similarity` |
| `image_size` | DINOv3 默认 1024；DINOv2／SAM3 默认 1008；三者共同分辨率可使用 1008 |
| `model_size` | `small`、`base`、`large`；SAM3 仅支持其固定的 `large` trunk |
| `svd_components` | 默认 500；`0` 取消 debiasing，报告实际有效维数 |
| `tau` | 默认 0.6，clustering granularity |
| `merge_threshold` | 默认 0.2，aggregation threshold |
| `similarity_threshold` | 默认 0.55，仅用于 similarity baseline；不是经我们验证得到的最优值 |
| `mask_refiner` | `bilinear`（默认）或另装依赖后的 `crf` |
| `device` | 默认 `cuda`；必须在实际 allocation 中使用 |

`variant` 是预设；明确提供的独立组件开关会覆盖预设。结果 metadata 保存全部 effective settings，不能仅依据 variant 名称比较。

## Encoder × processing 的语义

| Encoder | 特征定义与适配 | 能支持什么结论 |
|---|---|---|
| DINOv3 | 官方 `get_intermediate_layers(..., n=1, reshape=True)` | 原版与各处理组件的条件效果 |
| DINOv2 | 官方最后一层 patch features，float32 用于后处理 | 相同处理在另一 self-supervised 表示中的效果；不是作者 Table 8 调参配置的逐项复现 |
| SAM3 | **ViT trunk 最后一层、neck 之前**的 feature map；仅构造并加载 trunk；ImageNet normalize 撤销后应用 SAM3 的 mean/std=0.5 | SAM3 表示与 INSID3 处理的组合效果；不含 SAM3 text encoder、mask decoder 或 MLLM |

SAM3 features 变体不再具有“仅 self-supervised 预训练”的属性；下游仍不更新参数。为了复用 INSID3 的共同 resize，SAM3 使用该 resize 后的图像与正确 SAM3 normalization，而非声称完全复刻 SAM3Processor 的 uint8 resize 路径。

DINOv3 patch size=16，DINOv2／SAM3 patch size=14。配置非法分辨率直接报错，避免静默 crop。若比较原版默认分辨率与共同 1008 设置，分别记录，两者不能混成一个实验。

阈值／特征层／SVD 维数不能根据测试集最优成绩选择。首先运行冻结默认配置；若做调参，为每个 encoder 使用独立验证集和相同预算。小 encoder 的特征维数可能小于默认 SVD=500，必须检查报告的 `effective_svd_rank`；删除全部特征子空间没有合理语义，应在模型规模分析中单独选择验证配置。

## 成本与失败

- `encoder_calls` 来自真正的 `get_intermediate_layers` 入口，`encoder_images` 表示该调用处理的图像数。多个 reference 与 query 合成一个 batch，因此两者不同。
- 一次性的 positional basis 构建单独记为 `basis_build_encoder_calls`，不计入逐 episode 调用。
- `encoder_seconds` 包含在 `segmentation_seconds` 内，不应相加。
- 每个 episode 清空 reference/target 状态；不跨 episode 缓存 embeddings。
- 空 reference、非法 mask、源文件变化、权重不匹配均报错，由 runner 记录失败；不伪造空预测作为成功结果。
- GPU 显存与端到端耗时由公共 runner 测量，CPU synthetic tests 不能验证真实 GPU 数值或性能。

参考：INSID3 https://github.com/visinf/INSID3 ；SAM3 feature 来源为 CG-ICS 锁定仓库中的 SAM3 实现。
