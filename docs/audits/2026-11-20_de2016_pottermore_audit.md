# SOURCE AUDIT — Pottermore 2016 德语七册合集（七册库 DE 底本）（2026-11-20）

> **发布说明**：本文为审计记录的入库版。完整报告、可重跑脚本（`de2016_structure.py` / `de2016_compare.py` / `de2016_probes.py`）、机器可读产物（`hp1_epub_vs_pdf1998.json`）与解包原件均在本地 `data/derived/de2016_epub_audit/`（gitignored，不随仓库分发）。源文件指纹（SHA-256）按仓例不入库，登记于本地 `CLAUDE.local.md`。
> 审计对象与 Bremmers 复现工作流的 HP1 德语源 PDF 无关：本件为**规划中七册平行语料库**的德语底本，不属于 HP1 复现工作流（DE 侧继续用 1998 Carlsen 初版 PDF——论文引用的正是 Carlsen 1997/98, trans. Klaus Fritz）。

## 1. 文件身份（零售 EPUB，非转制件）

| 层 | 身份 | 证据 |
|---|---|---|
| 发行层 | Pottermore 2016 德语电子合集 "Harry Potter: Die Gesamtausgabe (1-7)"，ISBN `978-1-78110-651-8`，2016-12 上架 | OPF `dc:identifier`；外部书目（Pottermore Publishing，Olly Moss 封面，与版权页吻合）；文件名 ISBN 三方一致 |
| 本地层 | 上述合集的**零售态 EPUB**（EPUB 2，无 DRM），非 calibre←Kindle 转制 | mimetype 首位 + `OEBPS/content.opf` + `toc.ncx` 自足结构；资产名全带 PTMR 水印；Pottermore 家族样式表；iBooks display-options；内嵌时间戳全 2016-11-03 |

- 本地文件 `data/raw/hp_de_2016_pottermore_1-7.epub`（5,863,548 B）。获取时原始长文件名（含 `isbn13 9781781106518`）记于本地完整报告；改名后 SHA 复验一致（仓例同 EN 2015 件）。
- 唯一非零售痕迹：`META-INF/calibre_bookmarks.txt` 阅读器残留（OPF 无 calibre 转制元数据）→ 判定零售 Pottermore EPUB（或经 calibre 一次性重打包）；对 2016 发行原件的逐字保真 UNVERIFIED。
- **与 EN/ZH 两件获取链不同**（那两件是 calibre←Kindle 转制）；三件版本声明粒度对齐但链路如实分开记录。
- OPF `dc:language` 标为 `nl` —— **元数据缺陷**（内容为德语，§3 内文证据否决该信号）。

## 2. 结构完整性 — 全部通过

`de2016_structure.py`（可重跑，本地）：zip 零错误；manifest 251 项 0 缺文件、spine 232 引用 0 悬空、ncx 217 nav 点 0 未解析；乱码类五项全零（U+FFFD / C0 控制符 / 软连字符 / ﬁﬂ 连字 / 文本裸 `<`）；未引用资产仅 7 个备用封面图（cosmetic）。七册章骨架 = Carlsen 德文版全齐（17/18/22/37/38/30/37 + 每册 contents/dedication + HP7 题词页），章号严格递增，七册首章章题均德文。

| 册 | 章 | 词数 |
|---|---|---|
| HP1 | 17 | 80,503 |
| HP2 | 18 | 87,101 |
| HP3 | 22 | 109,966 |
| HP4 | 37 | 194,563 |
| HP5 | 38 | 260,919 |
| HP6 | 30 | 174,253 |
| HP7 | 37 | 206,008 |
| 合计 | 199 | 1,113,313 |

七册版权页 **7/7**：译者 **Klaus Fritz**；"Diese Digitale Edition wurde 2015 zum ersten Mal von Pottermore Limited veröffentlicht"；**"Copyright für die Deutsche Ausgabe © Carlsen Verlag GmbH 1998"**；原题 *Harry Potter and the Philosopher's Stone*。

## 3. 文字态（数字态 vs 印刷态）

- 对话标记 = **»…«（guillemets，内指）**，非德文印刷惯用的 „…"（HP1 任意章 0 个 „"；ch2 = 64 对 guillemets）。
- 复合词连写变体类：可分动词复合词在数字态连写（*leidtun*、*wiedersehen*、*hierbleiben*…，印刷态分写）——对 PP 抽取无实质影响（名词词元/介词不受此类变体影响）。
- 无正字法现代化：dass 473/472、muss 56/56、daß/muß 双侧全零。

## 4. 与本仓现有 HP1 德语语料（1998 Carlsen 初版 PDF 转写）比对

`de2016_compare.py` / `de2016_probes.py`（可重跑，本地）：

| 指标 | 值 |
|---|---|
| 逐章词数差 | +0.2–0.5%（EPUB 含 "Kapitel N" 章题行） |
| 句级 loose 归一存活（粗分句） | 88.2%（残差为段界/对话切分差；未匹配句引号占比 0%） |
| 词袋 vocab Jaccard / 频次余弦 / token 重合 | 0.961 / 0.9999 / 98.9% |
| EPUB 独有词形 | 163 型（复合词连写 + 章题词 + PDF 抽取伪影） |

**结论：同一 Klaus Fritz 1998 译文的两个状态（2015/16 数字态 vs 1998 印刷态）——同源、不同态、不得混用。** 对 Bremmers 复现无影响：HP1 的 DE 侧继续用 1998 PDF（论文引用 Carlsen 1997/98 Fritz，本仓 METHODS 已记录版本对应）。

## 5. 可用性判定

**有条件可用作七册平行语料库德语底本**，条件：

1. 版本字段按本报告精度登记，表述模板："Pottermore 2016 German digital edition (*Die Gesamtausgabe* 1-7, retail EPUB, ISBN 978-1-78110-651-8); Fritz translation © Carlsen 1998; later digital state, exact printing UNVERIFIED; guillemet dialogue"。不填猜测印次。
2. 不得与本仓 1998 PDF 混用或互替；七册库 HP1 起统一用本 EPUB。
3. EPUB 摄取前端需适配：guillemet 感知分句、"Kapitel N" 章题入元数据（现有 `segment.py` 德语规则面向印刷引号体系）。
4. 与 EN 底本（`docs/audits/2026-11-19_en2015_pottermore_audit.md`）、ZH 底本（`docs/audits/2026-11-16_zh2016_pottermore_audit.md`）版本声明粒度对齐；获取链差异如实记录（EN/ZH = calibre←Kindle，DE = 零售 EPUB）。

UNVERIFIED：对 2016 发行原件的逐字保真（一条 calibre 阅读器残留）；数字态对应的具体印次；guillemet 是否 Pottermore 德语电子版通例。

## 6. 信息来源与证据等级

等级定义：A 权威编目/一手；B 多源二手；C 单一二手；D 弃用。

| 来源 | 等级 | 用于 |
|---|---|---|
| EPUB 版权页 ×7 / OPF | A（对文件本身） | Fritz 译文声明、© Carlsen 1998、合集 ISBN、数字版 2015 首发 |
| 外部书目（Pottermore Publishing 德语电子合集，ISBN 978-1-78110-651-8，2016-12） | A/B | ISBN ↔ 德语合集绑定；与 Carlsen 2023 纸质 Gesamtausgabe（ISBN 3551557489）区分 |
| 本仓 `data/text_clean/hp1_de_*`（1998 初版 PDF 转写） | A（对印刷态） | §4 比对分母与正字法对照 |
| OPF `dc:language=nl` | D | 经内文证据否决（元数据缺陷） |

判定纪律：文本体系归属只依据内文证据 × 权威编目；文件名/渠道标签不作依据。
