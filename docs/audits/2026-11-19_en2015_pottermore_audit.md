# SOURCE AUDIT — Pottermore 2015 英文七册合集（七册库 EN 底本候选）（2026-11-19）

> **发布说明**：本文为审计记录的入库版。完整报告、可重跑脚本（`scan_structure.py` / `verify_integrity.py` / `compare_hp1.py`）、机器可读产物（`chapters.jsonl` / `markers.tsv` / `hp1_diff_pairs.tsv`）与解包原件均在本地 `data/derived/en2015_epub_audit/`（gitignored，不随仓库分发）。源文件指纹（SHA-256）按仓例不入库，登记于本地 `CLAUDE.local.md` 与 `EN_2015_volume_manifest.csv`。
> 审计对象与四份 HP1 源 PDF 无关：本件为**规划中七册平行语料库**的英文底本候选，不属于本仓 Bremmers 复现的工作流。

## 1. 文件身份（两层，均已确认）

| 层 | 身份 | 证据 |
|---|---|---|
| 第 1 层（未获取原件） | Pottermore 2015 **US** 合集 Kindle 发行件：ASIN `B01B3DIPMW`，合集 ISBN `978-1-78110-647-1`，发行日 2015-12-14 | OPF 保留 `MOBI-ASIN` + `dc:date`；外部书目（Amazon/Google Books，§7）确认该 ASIN/ISBN ↔ "Sorcerer's Stone through Deathly Hallows" US 合集 |
| 第 2 层（审计对象） | 本地 EPUB = 上述 Kindle 件经 **calibre 2.35.0** 转制的副本（转制时间戳 2016-02-17） | OPF `dc:contributor`；`META-INF/calibre_bookmarks.txt` 阅读器残留；对第 1 层的逐字保真性未验证（无原件对照） |

- 本地文件 `data/raw/hp_en_2015_pottermore_1-7.epub`（8,080,147 B，EPUB 2，无 DRM）。获取时原始长文件名与 SHA-256 记入本地 manifest；改名后 SHA 复验一致。
- **文件名 ISBN 纠偏**：获取渠道自动命名的 `isbn13 9781781106464` 经查为姊妹产品（UK "Philosopher's" 合集，ASIN `B01B3DKROQ`）的 ISBN——文件名错挂，以内证（七册版权页 + OPF）为准。
- 七册版权页同一模板：`Published in print in the U.S.A. by Arthur A. Levine Books, an imprint of Scholastic Inc.`；文本版权年 1997/1998/1999/2000/2003/2005/2007（= US 初版年序）；每册只印合集 ISBN 一个。

## 2. 结构完整性 — 全部通过

`scan_structure.py`（可重跑，本地）：TOC 216 锚点核对 + 逐册章节号严格递增，17/18/22/37/38/30/36 + 尾声 = 198 章章骨架全齐；每册书名页→献词→目次→正文→版权页结构齐全（HP7 另有题词页）；七册首段/末段全为 canonical 文本（HP7 尾声终于 "All was well."）；无截断、无跳章、无重复。

| 册 | 章 | 词数 | 段落 |
|---|---|---|---|
| HP1 | 17 | 79,186 | 2,937 |
| HP2 | 18 | 87,408 | 3,164 |
| HP3 | 22 | 111,229 | 4,131 |
| HP4 | 37 | 197,886 | 6,036 |
| HP5 | 38 | 265,955 | 8,773 |
| HP6 | 30 | 174,475 | 5,683 |
| HP7 | 36+尾声 | 202,214 | 6,825 |
| 合计 | 198+尾声 | 1,118,353 | 37,549 |

## 3. 转换完整性与文字质量

`verify_integrity.py`（可重跑，本地）：

| 检查 | 结果 |
|---|---|
| spine/manifest | 486 项 / 250 条 / **0 悬空引用**、0 缺文件 |
| 图片 | 232 文件 = 232 引用，**0 悬空、0 孤儿**；逐册章头插画 alt 与章题一一对应 |
| 内部链接锚点 | 100% 解析（0 失败）；本版无尾注系统（US 原书无译注） |
| 乱码类 | 0 U+FFFD / 0 控制字符 / 0 HTML 残留 / 0 软连字符 / 0 连字残留 |

初扫 ~5,000 条"异常"逐类复核后归为版式本体（间隔省略号 `. . .`、对话式 `— ?`——与印刷语料逐位一致，HP1 双侧 `— ?` 均恰 9 处）或误报（美式多段引语、悬挂连字符、对话重复词）。**真实缺陷仅 3 处**（全库 1.12M 词）：斜体跨段边界丢空格 ×2（HP2 ch11、HP3 ch16）+ 直引号 ×1（HP5 ch25 报纸标题内嵌）。入库修复规则一条：`([a-z][.!?])([A-Z])` 补空格。

## 4. 版本判定

**= US（Scholastic/Arthur A. Levine）文本体系，2015 Pottermore 数字重制态；拼写层全 US，词汇层为 "US 术语 + mum 全保" 混合态。**

1. **US 体系（确认，内证 A 级）**：书名 *Sorcerer's Stone*（philosopher:sorcerer = 0:34）；47 对拼写标记全 US 侧（colour/color、grey/gray 149、centre/center 60、favourite/favorite 92、duelling/dueling 62、pyjamas/pajamas 52、haemorrhage/hemorrhage…）；lemon drop 8:0、soccer 主导；`Mr./Mrs./Dr.` 带点（无点式 0）。
2. **mum/mom 全 UK 态（确认）**：七册 `mum` 246 : `mom` 0；本仓 Scholastic 1998 印刷 HP1 同维度 = `mom` 23 : `mum` 6（后者全为海格方言）。即数字版未继承印刷版的 mom 本土化；HP1–4 的 sweater/sneakers/lemon drop/soccer 又是 US 本土化。HP5–7 出现 jumper/trainers UK 侧形式，与 Scholastic 后期印次本土化减弱方向一致（外部通行说法，B/C 级，仅作解释）。
3. **对应印次 UNVERIFIED**：HP1 与 1998 印刷版存在 mom 系 ~29 处 + ~5 处微修订（§5），证明晚于 1998 初版的修订态；是否等同某具体 US 再印/重排版，无对照件，不猜测。

## 5. 与本仓现有 HP1 英文语料（Scholastic 1998 印刷转写）比对

`compare_hp1.py`（可重跑，本地），章内对齐 + 归一化后逐段分类，印刷段为分母（n = 2,053）：

| 类别 | 段数 | 占比 |
|---|---|---|
| 逐字相同 | 1,373 | 66.9% |
| 仅标点微差 | 70 | 3.4% |
| 含文字差异 | 382 | 18.6%（分解见下） |
| 切分边界差异 | 221 | 10.8% |
| 无 ≥0.80 对应段 | 7 | 0.3% |

18.6% 的词级分解（231 条差异对全分类）：段切分差异投影 169 + 印刷侧章首下沉大写字母缺失 16（EPUB 反而更完整，如 arry→Harry、OOM→BOOM）+ 大小写/连字/标点 27 + **真实措辞差异仅两类**——mom→mum 系（~29 处）与微修订 ~5 处（ch2 Tibbies→Tibbles、ch6 Woodcraft→Woodcroft、ch9 sanity→saintly、ch11 GRYFFINDORS→GRYFFINDOR、ch16 I→I'll（末条或为对齐伪差））。

**结论：同一 US 文本体系的两个不同状态（数字修订态 vs 1998 印刷态）——同源、不同态、不得混用。** 对本仓 Bremmers 复现无影响（DE↔EN 继续用印刷源 `hp1_en`）。

## 6. 可用性判定

**有条件可用作七册平行语料库英文底本**，条件：

1. 版本字段按本报告精度登记，表述模板："Pottermore 2015 US digital edition (Sorcerer's text, calibre 2.35.0 EPUB conversion of Kindle ASIN B01B3DIPMW); later-corrected print state, exact printing UNVERIFIED; mum-form throughout"。不填猜测印次。
2. 不得与本仓 `hp1_en`（印刷态）混用或互替；七册库 HP1 起统一用本 EPUB。
3. 入库加一条斜体边界空格修复规则（§3）；其余标点风格与印刷版一致，现有 `clean.py` 规则可吸收。
4. 与 ZH 底本（`hp_zh_2016_pottermore_1-7.epub`，见 `docs/audits/2026-11-16_zh2016_pottermore_audit.md`）同源同代（Pottermore 数字件、calibre 转制），版本声明粒度对齐。

UNVERIFIED 清单：对 2015 原始发行件的逐字保真性；数字态对应的具体 US 印次；mum 全保是转制策略还是所据印刷态如此。

## 7. 信息来源与证据等级

等级定义：A 权威编目/一手；B 多源二手；C 单一二手；D 弃用。

| 来源 | 等级 | 用于 |
|---|---|---|
| EPUB 自身版权页 ×7 / OPF | A（对文件本身） | US 谱系声明、合集 ISBN、ASIN、calibre 转制元数据 |
| Amazon US 合集条目（ASIN B01B3DIPMW / ISBN 978-1-78110-647-1，2015-12-14） | A/B | ASIN↔ISBN↔US 版本三方绑定 |
| Amazon / Google Books（ISBN 9781781106464 ↔ ASIN B01B3DKROQ） | B | 判定文件名 ISBN = 姊妹 UK 产品（错挂） |
| 本仓 `data/text_clean/hp1_en_*`（Scholastic 1998 印刷转写） | A（对印刷态） | §5 比对分母与标记对照 |
| "Scholastic 后期印次本土化减弱" 通行说法 | B/C | 仅解释 §4.2，不参与判定 |
| 弃用：文件名自带 isbn13 字段 | D | 经证错挂 |

判定纪律：文本体系归属只依据内文词汇证据 × 权威编目；文件名/渠道标签不作依据。
