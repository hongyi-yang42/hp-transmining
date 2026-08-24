# 对齐嵌入模型选型与更换记录（e5-base → LaBSE）

**状态**：已执行，LaBSE 自 2026-08-18 起为 canonical。
**决策日期**：2026-08-18（评估于 2026-08-16–18 完成）。
**本文目的**：完整记录本次换模的动机、评估协议、结果与执行内容，供方法学审查时查阅；
不承载随管线重跑而变化的生产统计数字。

## 背景

句对齐是本管线下游全部标注工作的原料：每个德语 PP 数据点要靠 DE–EN / DE–ZH
的句子对齐找到对应译文上下文。对齐器（`src/hp_corpus/align.py`）= 多语句子
嵌入模型 + 全局带状动态规划（DP）。嵌入模型给出的跨语言相似度是 DP 的唯一
信号源，其区分度直接决定对齐质量上限——这是模型选型成为关键变量的原因。

初版默认模型为 e5-base（`gap_penalty=0.1`）。

## 换模动机

对 e5-base 输出的诊断显示其存在**余弦压缩**：跨语言相似度分数整体挤在窄区间，
真配对与假配对的分数差距小，DP 难以区分。这构成启动正式对照评估的动因，
而非先有结论再找证据。

## 评估协议

评估分两轮金标准，第二轮为消除第一轮抽样偏差的扩容验证。

**第一轮（6 章归因层金标准）**：DE–ZH 前 6 章，按归因层（引语/叙述）分层抽样
182 例（ZH→DE 方向）+ 58 例可窗口例（DE→ZH 方向）。

**第二轮（gold2，全书随机金标准）**：为排除第一轮归因层抽样可能带来的偏袒，
改为 Ch.1–17 全书按章比例分层、固定 seed 的纯随机抽样，无归因过滤——
ZH→DE 150 例 + DE→ZH 100 例。候选窗取三模型（e5 / LaBSE / bge-m3）配对
结果的并集 ±2 句，对三个模型同等公平。

**判官与人工校验**：多个互相不可见对方裁决的 LLM 判官独立盲评每条候选是否
构成可接受的翻译对应；另做人工目视抽检 20 例（含 ZH 一句合并两 DE 句、
DE 一句拆 ZH 三句等难例），与判官裁决 20/20 一致，确认判官可信。

**预设决策门槛（先于看结果定好）**：DE→ZH 严格 ±0 精度提升 ≥ +8pp，且
ZH→DE acceptable 率回退 ≤ 2pp，方准入换模。

## 结果

**gold2（全书随机金标准，主证据）**：

| 臂 | ZH→DE acceptable (full/wrong) n=150 | ZH→DE 覆盖 | DE→ZH 严格 ±0 n=100 | ±1 |
|---|---|---|---|---|
| e5-base（旧默认） | 64.7% | 58.4% | 70% | 88% |
| **LaBSE@gap0.18（现行）** | **96.7%** | **90.6%** | **96%** | **98%** |
| bge-m3@gap0.18 | 89.3% | 82.5% | 90% | 98% |

分层细节：LaBSE 在引语层 13/13、叙述层 132/137 全优；DE→ZH quote-strict 35/36。
无偏随机样本上 LaBSE 对 e5 的差距比第一轮归因层样本更大（e5 在叙述层更弱）。

**第一轮（6 章归因层，佐证）**：

| 臂 | ZH→DE acceptable | 覆盖 | DE→ZH 严格 ±0 |
|---|---|---|---|
| e5-base@0.1 | 80.2% | 62.1% | 65% |
| LaBSE@0.1 | 92.3% | 79.0% | 77% |
| bge-m3@0.1 | 89.6% | 74.6% | 77% |
| **LaBSE@0.18** | **94.0%** | **82.9%** | **84%** |
| bge-m3@0.18 | 89.6% | 76.1% | 81% |

两轮结论同向。附加发现：gap 罚分必须随模型的余弦尺度重标定——LaBSE 分数
分布偏乐观，需要更高的 gap 罚分（0.18）强制配对，否则 DP 倾向插 gap 逃配。

## 决定与执行

- **采用 LaBSE + `gap_penalty=0.18`**：唯一在预设门槛上以最大余量通过且全面
  最优的臂。bge-m3 同样过门槛但所有指标居次，保留在 `models/bge-m3` 作备选。
- `AlignmentConfig` 默认值切换为 `model_name="models/LaBSE"`、`gap_penalty=0.18`
  （e5 运行需显式设 0.1）。
- 嵌入缓存 v3 按模型 + 模型指纹内容寻址（`config/embedding_models.yaml`），
  多模型向量在 `data/embeddings/` 下共存，换模不毁旧缓存。
- DE–EN / DE–ZH 全 17 章生产对齐以 LaBSE 重跑（`scripts/run_alignments_v2.py`），
  下游机器 master TSV 与标注 CSV 在新对齐上重建。
- 模型获取自 ModelScope 镜像（HuggingFace 自 CN 不可达），本地路径
  `models/LaBSE`（gitignored，身份由指纹锁定）。

## 局限

- 对照实验覆盖 DE–ZH 方向；DE–EN 方向未做独立金标准评估（两方向共用同一
  对齐器与模型，DE–EN 为印欧语对，跨语距离更近，预期不劣于 DE–ZH）。
- LLM 判官虽经 20 例人工抽检一致，仍非全量人工金标准。

## 后续决策：标注交付物按机器置信度切分（2026-08-24）

LaBSE 落地后的错误画像审计（双向 DP 交叉验证：ZH→DE 反向 17 章重跑，
1165 个含 PP 段双向一致率 98.6%）确认错误为**系统性**（引语/叙述框架的
切分粒度失配产生确定性 off-by-one），非随机噪声——双向同错，一致性本身
不能当作正确性证据。据此对标注交付物做置信度切分：

- **规则**：EN 或 ZH 任一侧机器对齐置信度 **< 0.40** 的行不进标注 CSV，
  移入 companion CSV（`annotation_pairs_low_confidence.csv`，附机器置信度
  与触发侧诊断列）供后续分析、eyeball 与人工裁决；机器 master 保持全量，
  eligible-pool join 不受影响。validator 从 master 重推切分做闭环校验。
- **0.40 的依据**：人工复核确认的错例全部落在 <0.40 带内（最高一例
  0.39）；0.35–0.50 带 126 段中 123 段双向一致，说明 0.50 会过度切除本来
  正确的行；切分行 contracted 占比 48.9% vs 保留行 52.2%，无形式偏倚。
- **companion 是待裁决 backlog 而非研究排除**：候选修复路径为人工对齐或
  EN 枢纽传递对齐（DE→EN→ZH；mini 验证 3 例人工确认真错中 2 例精确救回，
  全局信号分化：直连 ≥0.7 行 pivot 一致率 98%、<0.35 行分歧率 57%），
  pivot 的 50 例分歧段需判官验证后方可接线。

## 后续决策：切分规则扩展 + 上下文窗口收紧（2026-08-24，交接清理）

对齐交接前的两项小改动，不触碰 LaBSE/DP 对齐本身：

- **切分规则扩展**：在 <0.40 置信度规则之外，EN 或 ZH 任一侧
  `*_context_provenance` ∈ {`manual_review`, `neighbor_fallback`}
  （机器无正常可靠锚点/上下文）的行同样转入 companion CSV。实现于
  `hp_corpus.annotation_csv.split_low_confidence`（builder 与 validator
  共用，validator 仍从 master 重推切分闭环校验），不加新列、不加新
  reason 分类；`machine_low_conf_sides` 语义 = 触发切分的侧。现行量：
  1303 保留 / 89 切分（置信度规则单独为 88，provenance 规则净增 1）。
  切分行只是当前阶段分离待检，不是研究排除。
- **上下文窗口收紧**：routine ±1 不变；加宽（merge / 疑似 DE 欠切分 /
  机器置信度 <0.50）由 ±3 收紧为**每侧至多 ±2，且加宽窗口总长 ≤5 个
  目标句**（`candidates.budgeted_window`；锚组完整保留，预算只限制新增
  邻句，不再为长尾难例扩到 ±3/±4/±5）。重建后 26 个回归 case 侧全部
  通过（25 in-context + 1 manual_review），加宽窗口长度由 4–11 句降为
  ≤5 句；窗口覆盖不到的难例走 fallback plan → `manual_review` →
  companion 路径，而非加窗救援。

## 产物索引

| 内容 | 位置（gitignored） |
|---|---|
| 对照表（本文数字来源） | `data/derived/alignment_metric_review/model_comparison.md` |
| gold2 金标准条目/裁决 | `data/derived/alignment_metric_review/gold2/` |
| 第一轮判官逐条结果 | `data/derived/alignment_metric_review/judge_results_182.csv` |
| 人工 review 记录 | `data/derived/alignment_metric_review/review50.csv` |
| 生产对齐 manifest（含每章配置与统计） | `data/derived/alignment_v2/` |
