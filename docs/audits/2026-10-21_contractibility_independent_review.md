# INDEPENDENT_REVIEW — Bremmers contractibility 审计独立验收（2026-10-21）

> **发布说明（2026-11-05 收尾）**：本文为 2026-10-21 独立验收记录的入库版，随审计测试一并提交。
> 文中 `tmp/research/…` 为本地研究备忘（gitignored），不随仓库分发；引用其小节号仅供本地对照。
> 2026-11-05 复验：审计脚本退出码 0、52/52 锚点 PASS、全仓 464 passed、Ruff clean，关键数量与本记录一致。

验收对象：`acda90f`（`scripts/audit_contractibility.py`）、本地备忘 `tmp/research/PAPER_96_FORENSICS.md` §7、`tmp/research/contractibility_audit/`。验收约束已遵守：未修改生产筛选规则（`step4.py` / `sampling.py` / `build_eligible_pool.py` 全程只读），未自动填充人工标签，未删除反例。

**推送状态**：`acda90f` 于 2026-10-21 推送失败（网络不可达）。2026-11-05 收尾：确认远程 HEAD 未前进（仍为 `61e4867`）后，`acda90f` + 审计测试 + 本报告一并推送 `origin/main`（最终 SHA 见 git 历史）。

---

## 1. 交付真实性 — VERIFIED

| 命令 | 退出 | 摘要 |
|---|---|---|
| `git show --stat --oneline acda90f` | 0 | 仅 1 文件：`scripts/audit_contractibility.py`，701 行。commit 不含 §7（`tmp/` gitignored，符合设计） |
| `git status --short` | 0 | 验收开始时 clean；验收期间新增本报告 + 测试文件（验收时点未暂存） |
| `uv run python scripts/audit_contractibility.py` | 0 | 全部锚点 PASS；5 个输出文件重新生成于 `tmp/research/contractibility_audit/` |

- §7 存在（1 处 `^## 7`）。
- 输入 SHA-256（2026-10-21 验收时点；派生文件摘要，非源 PDF 指纹）：

| 文件 | SHA-256 |
|---|---|
| `data/derived/step4/full_novel_annotation_master.tsv` | `90d11bf6…1571a` |
| `data/derived/annotation/annotation_pairs.csv` | `f878d6cb…1f625` |
| `data/derived/annotation/annotation_pairs_low_confidence.csv` | `daf21722…5d0ce` |
| `data/derived/annotation/HP_annotation_pairs_001_150_integrated.xlsx` | `cb358272…55f1c2` |
| `vendor/.../prepositions/data.py` | `d5eaad98…47cb2` |
| 34 个抽取 TSV 聚合 | `b2ac49d4…8903`（逐文件清单：`tmp/research/contractibility_audit/input_extraction_tsvs.sha256`） |

- **新增测试**：`tests/test_audit_contractibility.py`（17 个合成正反例：39 格 (prep,det) 映射完备性/标准格 12 格精确集/口语格精确集/阴性 der 仅 zu/9 介词集推导/分类边界/strict_norm 四例/内联串+sharedStrings+空单元格 xlsx 解析/Anchors 行为/ID 集不相交）。全仓 464 passed，Ruff clean。

## 2. 数量独立复核 — VERIFIED

独立实现（`tmp/research/contractibility_audit/independent_recheck.py`，与被验脚本零共享代码：独立分类规则、独立 xlsx 解析、vendor 清单经 exec 原文加载）：

| 断言 | 独立复算 | 判定 |
|---|---|---|
| U 668 / 9 介词 512 / 4 介词 156（aus 109/hinter 17/unter 20/gegen 10） | 一致 | ✓ |
| strict 清单外 390 = 281 阴性 der（X≠zu）+ 105 口语 dem + 4 残余标准 | 一致（残余 4 行 ID 逐一相同） | ✓ |
| 标准类 121（668 内）= 121（512 内）；4 介词贡献 0 | 一致（121 行 ID 清单落盘 `recheck_standard_121_ids.tsv`） | ✓ |
| Group A 75（44 incl/31 excl）；U9 include 17 → **11 非标准**；44 → **15** | 一致（11 条 datapoint ID 全列出，见 `recheck_includes_15_ids.tsv`） | ✓ |
| Group B 75 行 EN/ZH 已填 | 75/75 | ✓ |

**385 与 390 的 5 行差（已查明）**——四种匹配口径的完整阶梯（9 介词 U 域，清单外出现次）：

| 口径 | 清单外 | 相邻差 | 行级归因 |
|---|---|---|---|
| raw 精确 | 395 | — | — |
| strict（ß→ss＋去句点＋小写） | 390 | −5 | 拼写改革行：`ch06_p0112_s004`(zu dem **Schloß**)、`ch09_p0022_s003`(auf das **Schloß**)、`ch13_p0037_s015`(über das **Schloß**)、`ch16_p0095_s008`(in das **Schloß**)、`ch17_p0043_s003`(zu dem **Schluß**)——ß→ss 后入册 |
| strict＋前缀（≥4，一方为另一方前缀） | **385** | −5 | `ch06_p0065_s004`+`ch08_p0008_s005`（**Klasse**↔Klassenzimmer）、`ch06_p0076_s006`（**Haufen**↔Haufe，唯一真变体）、`ch09_p0015_s003`+`ch13_p0037_s009`（**Sonne**↔Sonnenlicht）——即 385−390=5；除 Haufen 外均为假前缀命中 |
| strict＋前 4 字符相等（§6.4 变体） | 381 | 再 −4 | 追加 Stadt↔Stadion、Schlange↔Schloss/Schlafzimmer、Schalter↔Schaufenster 假命中 |

（ß→ss-only 无句点/小写 = 390，与 strict 等值：FILTER_PP 无带句点/大小写差异条目。）

**抽取层 vs master（分别报告，未合并）**：U 侧逐键**全等**（extraction-only 0 / master-only 0）；C 侧 extraction=726、master=724，extraction-only = 2（`ch04_p0030_s002` t11-12 **ums Leben**、`ch15_p0021_s003` t4-5 **ums Herz**，um 不在 13 介词，被池构建的 inventory 门剔除），master-only = 0。

## 3. 流程阶段判断 — VERIFIED（含更正）

- `step4.py`（机器候选生成）：**无形态门，无配对门**——docstring 明确暂缓，仅预计算 `minimal_pair_group`。
- `src/hp_corpus/sampling.py:24-28`（正式合格池规则）：**ch4+ 配对门已实现**——C ch4–17 仅当 `(canonical_prep, head_lemma)` 亦出现在 **reviewed-include 的 U 集**（修正词元优先）；另含 13 介词 inventory 门（um/vor 剔除，解释 ums×2）与 U 侧 `det_xpos==ART` 结构门。由 `scripts/build_eligible_pool.py` 执行。
- **全量真实执行未发生**：`data/derived/eligible_pool/` 不存在（Group B de_valid 未终判，fail-closed 拒收空白）。
- 更正已落地：本地备忘 §7.1/§7.5 已改写；"配对限制在任何环节都未实施"说法**撤回**。注意：池阶段配对用的是 reviewed-include 集 + 修正词元，与 §7.5 的机器口径量化（593→60）**落点会不同**。

## 4. 反向检查 — VERIFIED

- **清单外 4 条标准行逐一验证**：`ch03_p0003_s019`（in dem grauen **Wasser**，im-able，真遗漏）；`ch10_p0048_s002`（zu der offenen **Tür**，zur-able，真遗漏）；`ch12_p0002_s006`（surface 即 `zu mit dem Vorschlag`，游离 zu 伪影，span 污染在词形中可见）；`ch06_p0076_s006`（**Haufen**，清单实有 Haufe，词元约定差）。
- **清单内非标准行**：5 条口语行的同型 das 框架全部真实存在（Sofa ch04×2；Spielfeld ch11×2+ch13；Feuer ch14；Spiel ch11）；清单内不可能类 = 0。
- **(zu,der)→zur 边界**：全书共 5 行 = 4 在册（Katze ch1/Schlange ch2/Frau ch6/Tribüne ch11）+ **1 清单外**（Tür 真遗漏）——并非"全部在册"。**(auf,dem)→aufm 边界**：9 介词内 (auf,dem)×101＋(über,dem)×9＝110（105 清单外＋5 在册）；(für,dem)=0。
- **ART 证据级别**：281 行的 `det_xpos=ART`/`deprel=det` 为 Stanza 机器证据，仅作分析支撑；`sampling.py` 的 ART 结构门同样是机器判据，**非人工 gold**（25 条 not_definite 人工排除才是人工层证据）。
- **Group B 三分**：① 原始状态 = 75 行 de_valid 空白（德语悬置、EN/ZH 已填 75/75）；② 隐含 include 约定 = **不存在**（`docs/ANNOTATION_CSV.md:105` "Every row must get one of the two values"；validator 对空白报 `DE_VALID_INVALID`，fail-closed）；③ 正式要求 = 每行显式 include/exclude 方可回传建池。

## 5. "会议 83 条" — UNVERIFIED

`GLAC25.txt`（AC2019 全文提取）**不存在 83**；该版明确 96（"The number of contexts we ended up with for German-English-Mandarin amounts to 96"）。最接近的实数：96 个数据点中 80 个含定指成分（MDS 图注）；EN "the" 30/40＋50/56＝80。若 83 另有出处（如某次演示、某表），需研究者提供来源方可评价。

## 6. 结论分级

| 问题 | 结论 |
|---|---|
| 可用于机器诊断？ | **是** — 脚本可复现、全锚点过、独立复算一致、规则面被 17 个合成测试钉住 |
| 可供下一步人工审核？ | **是（有限）** — Group B 75 行可继续德语终判；`group_b_advisory.csv`（35 行在论文语法读法空间外）可作参考信息，但不得预填或据此跳过人工判断 |
| analysis-ready？ | **否** — 人工审核未完成（Group B 悬置）、eligible pool 从未构建、回传 gate 未通过 |
| 足以解释论文 96？ | **结构性解释成立、数值复现不成立（也不应成立）** — 96⊆224 已证；U 侧差异 99% 形态不可缩合（本轮独立复核确认）；C 侧为语境级抽样；224→96 为人工挑选。机器层无法产出 96 属预期 |
| 足以解释"会议 83"？ | **无法评价** — 83 出处未明（§5） |

> **口径限定（2026-11-05 补，防止误读）**：本记录及上游本地备忘中出现的 **724:121（≈86% 缩合）是混合筛选口径下的德语候选数量比**——C 侧 724 仅过 13 介词 inventory 门（未过 ch4+ 配对门，过门后为 191），U 侧 121 为 9 介词域内形态标准类的结果（668→512→121）。它**不是**统一筛选规则下的最终样本构成，**更不是**汉语形式（裸名词/指示词/其他/省略）对应比例；任何场合不得表述为"已复现论文比例"。论文侧 40:56 为人工挑选样本，与本口径不可直接相比。

## 7. 产物归属

入库（随本记录提交）：

- `tests/test_audit_contractibility.py`（17 测试；全仓 464 passed）
- 本报告（`docs/audits/2026-10-21_contractibility_independent_review.md`）

本地保留（不入库）：

- `tmp/research/contractibility_audit/`：`independent_recheck.py`、`recheck_standard_121_ids.tsv`、`recheck_includes_15_ids.tsv`、`input_extraction_tsvs.sha256` 及审计再生输出
- `tmp/research/PAPER_96_FORENSICS.md`（§7.1/§7.5 更正在内）
- `uv.lock` 工作区改动：仅 PyPI 镜像地址变化（109 个包的 name-version 集合逐行一致），保留本地、不入提交

## 8. 收尾状态

2026-10-21 验收停止交审后，2026-11-05 完成 Git 发布收尾：复验测试与审计、整理本报告入库、推送 `origin/main`（普通推送，无 force）。未合并其他分支、未更新外部系统、未改动生产筛选规则与人工标签。
