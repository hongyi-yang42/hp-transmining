Status: evidence reconstruction for report drafting; not a methodological result and not a paper conclusion.

# How the researcher worked with the coding agent — first-hand interaction evidence

Working notes for the research-methods report. Six representative interactions, each reconstructed
from Claude Code session transcripts, git history, repo docs, and local artifacts. All timestamps in
transcripts are UTC; commit timestamps are local (UTC+8).

**Evidence base.** 30 session transcripts (2026-08-07 → 2026-08-28, the full retained history);
87 commits (2026-08-05 → 2026-08-28); `docs/ALIGNMENT_MODEL_DECISION.md`, `docs/ANNOTATION_CSV.md`,
`docs/ANALYSIS_SAMPLE_DESIGN.md`, `docs/FULL_NOVEL_SAMPLING.md`, `docs/METHODS.md`; project memory
files; gitignored local artifacts cited by path only.

**Quote discipline.** Every quote below labeled *verified exact* was re-checked character-for-character
against the decoded transcript text (user-typed messages only; tool results and system blocks excluded).
`[...]` marks editorial omissions; each contiguous fragment around an omission is itself verified.
Sessions are cited by the first 8 hex chars of the session UUID (full IDs in the index at the end).

**Known recovery gaps** (things that cannot be quoted from transcripts):

1. Repo history starts 2026-08-05 but transcripts start 2026-08-07 — the scaffolding era
   (EN pipeline end-to-end `f21c532`, first DP alignment `4791936`, etc.) has no recoverable prompts.
2. The decision to extend the reproduction to the full novel (Ch.1–17) was made by the researcher
   offline while reading the paper; the earliest in-session record is the researcher pasting their own
   decision summary as the opening prompt of session `64168855` (2026-08-09). The reasoning itself
   happened outside any agent session.
3. The full-table annotation-CSV review that produced the 23-case regression list (Interaction 4/5)
   was run by the researcher with an LLM outside these transcripts; only its corrected results,
   pasted into the agent session, are on record.
4. The Ch.15 "bei dem Aufruhr" case is evidenced by the regression manifest + master TSV + commits,
   not by any transcript prompt naming it (verified 2026-08-28: manifest case
   `dp_ch15_hp1_de_ch15_p0059_s001#b001_t15-17`, PP surface "bei dem Aufruhr", uncontracted,
   zh provenance `neighbor_fallback`).
5. Some design gates were authored by the agent inside plan files and approved by the researcher
   through Claude Code's plan-approval UI ("User has approved your plan") rather than typed text —
   noted where it matters (Interaction 2).

---

## Interaction 1 — Plan-before-implementation: read-only root-cause investigation of the German source switch (Aug 7–8)

**Research problem.** The 1998 first-edition German PDF (the edition the paper cites) had just replaced
the 2013 reissue as extraction source. Contracted-PP match rate against the paper's filter dropped from
83% to 69%. Before touching anything, the researcher commissioned a read-only root-cause investigation
with an explicit stop gate.

**Actual user instruction** (session `69b58362`, L654, 2026-08-07T14:28:36Z — verified exact):

```
Task: Investigate root cause of PP match rate drop (83%→69%) between
2013 and 1998 DE sources. Do NOT switch source or modify existing files.

[...]
- 2013: 898 sent, 15098 tok, 132 contracted PP, 110 matched (83%)
- 1998: 819 sent, 14129 tok, 111 contracted PP, 77 matched (69%)
- 33 PPs lost, 14pp drop — not OCR noise

[...]
Constraints:
- Read-only: do NOT modify any config, code, or data file
- Do NOT re-run alignments
- Do NOT write novel text to terminal (copyright)
- Do NOT commit

Acceptance criteria:
- All 33 missing PPs categorized
- Per-category counts reported
- 3-5 worked examples with sentence IDs

Stop gate: Before any config modification or source switch decision
```

**Agent response / proposed plan.** Followed the prescribed 5-step analysis: set-difference of
(prep, head_lemma, sentence_id) triples between the two extraction TSVs, trace each missing PP to its
source sentence, categorize. No code was written; the investigation report landed 9 minutes later.

**What was actually implemented.** Nothing at investigation time (stop gate honored). After a
two-stage authorization — first "fix the concat artifacts" only (2026-08-07T14:42Z), then the full
switch (2026-08-08T03:59Z) — four commits on 2026-08-08: `be80ec7` (text_split concat-artifact splitter,
22 tests), `c0dab3f` (integrate into cleaning), `3ebfdf7` (switch `hp1_de*.yaml` to the 1998 source),
`57746ed` (docs).

**Evidence returned.** All 33 lost matches categorized: 8 (33%) 1998 text-layer concat artifacts
(fixable), 8 (33%) parser divergence, 5 (21%) 2013 PaddleOCR false positives (i.e. the old 83% was
inflated; corrected ~81%), 2 genuine wording differences, 1 unclear; system-wide token evidence
(tokens ≥15 chars: 2.29% vs 0.66%). Analysis saved to
`data/derived/source_switch_1998/pp_drop_root_cause.md`. Before the switch was authorized, the agent
verified 1998+fix extraction equals 2013 (110/132 matched), 140 tests green.

**Human decision.** Two-stage acceptance. First: fix path only —
*verified exact* (L762, 2026-08-07T14:42Z):

```
fix the concat artifacts. then check after that the 1998+ 修复链接是否可以更好一点，我选择1998是因为这样和原论文更加符合
```

Second (2026-08-08T03:59Z, excerpt — verified fragments): full authorization to commit `text_split`,
integrate the fix, make 1998 the sole DE source, re-run the affected alignments, rebuild the Step-4
pack — with constraints (keep both PDFs; do NOT re-run EN/ZH pipelines; do NOT modify step4.py logic;
no novel text to terminal; clean staging per commit) and a new stop gate before extending to Ch.4–17.

**Evidence locations.** Sessions `69b58362` (L654, L762, L1073), `cbf32935` (L2138, originating
context: the researcher deposited the 1998 PDF in `data/raw/` on Aug 6 and re-prioritized toward paper
reproduction); commits `be80ec7` `c0dab3f` `3ebfdf7` `57746ed` (all 2026-08-08);
`data/derived/source_switch_1998/`.

---

## Interaction 2 — Controlled alignment-model comparison and the LaBSE switch (Aug 17–19)

**Research problem.** Sentence alignment feeds every downstream annotation. Diagnostics showed the
default e5-base model suffered cosine compression (true and false pairs barely separable). The question
became: fix the algorithm, or replace the embedding model — and how to evaluate either claim honestly.

**Actual user instructions** (session `11beee98`; all verified exact):

Challenge the proxy metric and demand honest reporting (L1078, 2026-08-18T02:25:54Z):

```
所以整体位移的原因你找到了吗？修复完成了？状态有什么好转？作为1+2的实验之后哪些措施是有效的？多有效？你要汇报这些不是吗？另外一致率的golden
  standard是如何计算的？你怎么知道最好的alignment如何发生？需要提供给你一个单独的对照译本？还是你有什么其他思路？告诉我你整体的思考的过程
```

Reject the evaluation itself, mandate LLM-subagent judging (L1488, 2026-08-18T07:29:47Z):

```
对的 显然你的代理不准确，能不能这个判别的任务你每次就交给一个LLM subagent来干？可靠吗？想想看如何prompt？然后你先自己看一下这个review 50所暴露出来的我们alignment上面的问题还有我们alignment评测方面的问题，显然被低估了我觉得？？
```

Pose the base-model hypothesis, require plan-first (L1817, 2026-08-18T10:53:43Z):

```
或者你的基座模型能不能替换呢？是不是基座模型的问题？另外零和博弈的局面如何解决？我们实际上用alignment只不过希望从找到语法现象的german转到对应的中文，是不是也不需要强制要求中文之间相互独立？反而应该可以给一定的上下文？先试试看算法层面，再判断是否是基座模型需要替换？如果要替换需要怎么替换？有什么更好的选择？check more  repo  or  做法
```

```
同意都做 但是先做好规划
```
(L1845, 11:06:08Z)

Challenge the sample behind the percentages (L2210, 12:03:45Z):

```
你这个百分比是如何计算的？是通过你目视检查的还是只有在50个句子上面做的？
```

Commission the unbiased gold standard (L2221, 12:05:52Z):

```
你先补全一下金标准吧，至少包括我们关心的ch1-17随机抽样，样例也可以适当扩充一点，subout agents if needed for 目视
```

Acceptance (L2457, 12:29:05Z):

```
同意切换 LaBSE，开始 canonical 重跑。
```

**Agent response / proposed plan.** After the proxy-metric challenge, the agent built an LLM-judge
evaluation: a 50-row file for external inspection, then mutually-blind judge subagents with an
exemption list for known artifacts (pronoun attribution, name-form variation, quote/attribution split
asymmetry). For the model comparison (plan `misty-seeking-plum.md`, approved via plan mode), it
pre-registered the decision gate **before running**: DE→ZH strict ±0 must improve ≥ +8pp over e5-base
AND ZH→DE acceptable rate may not regress > 2pp; below gate → keep e5-base, report only. Canonical
switch was explicitly out of scope until separately approved.

**What was actually implemented.** Round 1 (6-chapter attribution-layer gold, 182 + 58 items, 5 blind
+ 10 judge subagents): e5 80.2%/65%; LaBSE@0.1 92.3% but 1:0/0:1 pairs exploded (159/70) — gap penalty
must be recalibrated per cosine scale; LaBSE@0.18 94.0%/84%; bge-m3@0.18 89.6%/81%. After the human's
sample challenge, gold2: 250 items, Ch.1–17 chapter-proportional fixed-seed random draw, candidate
windows = union of all three models' pairs ±2 (fair to all arms), 19 blind judge subagents:
e5 64.7% ZH→DE acceptable / 70% DE→ZH strict ±0; **LaBSE@0.18 96.7% / 96%**; bge-m3 89.3% / 90%.
Then the canonical rerun: `023c76e` (defaults → `models/LaBSE`, gap 0.18), `326b052` (candidate
windows), `9e6ed21` (arity-5 N:M grouping), `11819f1`, merged `991f473`; post-switch audit fixes
`3868726`–`52ba80d` (honest src/tgt schema, cache v3 model fingerprints, full-config manifests);
decision record `450b396` (`docs/ALIGNMENT_MODEL_DECISION.md`).

**Evidence returned.** Both rounds' tables archived at
`data/derived/alignment_metric_review/model_comparison.md` and `gold2/`; per-item judge verdicts in
`judge_results_182.csv`; canonical rerun QC: DE-EN 236/5708 = 4.1% needs_review, DE-ZH 676/5004 =
13.5%; 386 tests at merge.

**Human decision.** Accepted LaBSE + gap 0.18 (the only arm passing the pre-set gate with maximum
margin); bge-m3 kept as a local backup (passed the gate but second everywhere); e5 kept runnable for
historical runs (must set gap 0.1 explicitly); ratio scoring and two-pass anchor DP stayed
**default-off** — they had failed their own pre-declared gate earlier that week (+1.1pp proxy gain
falsified by the judge gold standard as actually −3.3pp). EN–ZH was later demoted to advisory
`--diagnostics` (the researcher's own audit, `7605c87`).

**Accuracy caveat (important for the report).** The gold standard was judged by LLM subagents, not by
human annotators: the researcher explicitly deprioritized human alignment annotation ("我不准备针对
alignment展开人工标注" — L1090) and delegated eyeballing to subagents (L2221). The 20-item
spot-check that calibrated the judges (20/20 agreement) was performed by the agent itself, not the
researcher — the decision record's phrase "人工目视抽检 20 例" should not be read as human gold
annotation, and the one artifact designed for direct human judgment (`review50.csv`) has 0/50 judgment
columns filled. The pre-registered +8pp/−2pp gate was authored by the agent inside the plan and
approved by the researcher via the plan-approval UI. No full human gold annotation of alignment exists.

**Evidence locations.** Session `11beee98` (L1078, L1090, L1488, L1817, L1845, L2210, L2221, L2457);
sessions `12f7c4eb`, `72dc250e` (post-switch audit, decision-record commission); commits `ea974b3`
`d6ece16` `fe4d7e4` `9e6ed21` `326b052` `023c76e` `11819f1` `991f473` `3868726` `8592787` `7605c87`
`55d1128` `52ba80d` `450b396`; `docs/ALIGNMENT_MODEL_DECISION.md`;
`data/derived/alignment_metric_review/`.

---

## Interaction 3 — Corpus-wide implementation of an already-decided requirement: full-novel German mainline, Ch.4–17 (Aug 14)

**Research problem.** The full-novel scope (Ch.1–17, all three languages) was decided by the researcher
on Aug 9 while reading the paper — outside any agent session. What remained was a bounded engineering
task: extend the German machine stage from Ch.1–3 to Ch.1–17 without fabricating anything.

**Actual user instruction** (session `716b7948`, L21, 2026-08-14T07:51:09Z — verified fragments):

```
Track A local-agent prompt: Ch.4–17 execution

你是 Harry Potter Translation Mining 项目的 Track A execution agent。你的任务是核查并推进 Ch.4–17 的真实 German pipeline：输入检查 → parse → paper-faithful extraction → provisional/full-novel sampling。你不是 annotation agent，不得替人类写 gold。

[...]

本轮目标

尽可能完成第 4–17 章的机器阶段。如果输入仍然缺失，必须给出精确的缺失清单并 fail closed；不要创建伪造的 PDF、config、parsed file、candidate count 或 research result。

[...]

不要把未实际运行的结果写成 completed；不要把计划值、旧 pilot 数字或 allow-missing 诊断数写成 full-novel reproduction result。
```

And the page-location rule (L155, 2026-08-14T08:24:11Z — verified fragment):

```
不要猜页码。如果 PDF 文本层无法明确确认某一章的起止页，请列出歧义并停止该章，不能用猜测配置继续运行。
```

**Agent response / proposed plan.** Round 1 refused to fabricate: the input gate found Ch.4–17
config/segmented/parsed all missing, reported a precise 28-item missing list, ran a --allow-missing
diagnostic explicitly labeled diagnostic, and a Ch.1–3 positive control (row counts identical to the
old extractor). Only then did it locate the 14 chapter page ranges from the PDF text layer (fixed
vertical offset of title blocks, y0≈83.1/97.2; 16 offset pages + Ch.1 = 17 chapters; four known anchor
pages all hit; pp. 7–337 contiguous) and propose a manifest-gated full-novel master builder.

**What was actually implemented.** Commit `1f971f3` (17 files, +1176/−8): 14 new configs
`config/hp1_de_ch04..ch17.yaml`; `zero_hits_ok` manifest blessing for legitimately header-only
chapters; `scripts/build_full_novel_annotation.py`; 8 focused tests. Follow-on `d17dd85` (EN Ch.4–17 +
ZH Ch.4–6 configs, +817).

**Evidence returned.** Full-novel extraction manifest 34/34 entries status=ok (per-entry sha256,
sentence counts); totals 726 contracted + 668 uncontracted = 1,394 occurrences; Ch.1–3 row counts
identical to the prior extractor; the new builder correctly **refused** to emit a partial master
(MISSING_INPUTS: 56 files = the EN/ZH Ch.4–17 gap); 351 tests passed, Ruff clean.

**Human decision.** Accepted ("这轮已经完成了 German machine mainline，结果有效"), ordered exactly one
scoped commit of the four delivered file groups, no push — then re-scoped the remainder into
PHASE A/B/C with aggregate-only handoff reports between phases and explicit prohibitions
("不要运行 German review，不要填写 gold annotation，也不要把 1,394 个 machine occurrences 称为论文
final sample") — L424, 2026-08-14T09:08:02Z (verified fragments); commit landed 56 seconds later.

**Evidence locations.** Session `716b7948` (L21, L155, L424); session `64168855` L4 (Aug 9, the
offline scope decision pasted in); commits `1f971f3`, `d17dd85`; `docs/FULL_NOVEL_SAMPLING.md`;
`data/extracted/full_novel/manifest.json`.

---

## Interaction 4 — Debugging by tracing the pipeline: ZH Ch.17 tail truncation (Aug 19)

**Research problem.** After the researcher's own full-table review of the annotator CSV (conducted
with an LLM outside the transcripts), 23 datapoints had at least one side whose context window missed
the true counterpart locus. The researcher brought the corrected failure table to the agent as a task
specification — verification-first, model changes explicitly excluded.

**Actual user instruction** (session `e7469420`, L7, 2026-08-19T12:58:49Z — verified fragments;
the 7-row failure table between the opening and the fix section is elided, PP identifiers only:
Ch.3 `aus dem Kamin`/EN, Ch.6 `zum Aufprall`, Ch.12 `im Kopf`, Ch.13 `in der Hausmeisterschaft`,
Ch.14 `im Schrank`, Ch.17 `beim bloßen Anblick` and `im Sommer`, all ZH):

```
你质疑得对。**我重新按 1,392 个 German PP occurrence 全表逐章检查后，需要修正前面的数字：non-empty but wrong 不是 5 条，而是目前确认的 7 条。**

判定标准严格按你的研究需求：不是“上下文主题差不多”，而是**这个具体 German PP occurrence 对应的译文位置必须在 context 中；否则不能可靠标 counterpart / omission。**

[...]

## 最小正确修法

我不会改 LaBSE。

[...]

我认为这才是现在最值得交给 coding agent 的下一步。
 check if you understand the problem? is it real? what is your proposal?
```

(The elided middle contains the per-side failure counts — EN 11 failed → 1381/1392 = 99.21%, ZH 15
failed → 1377/1392 = 98.92%, both-sides-pass 1369/1392 = 98.35% — and the researcher's own epistemic
guard on that number, verbatim:

"它依旧属于 **LLM-assisted QC / machine-ready evidence，不是 human-validated 98.35% gold accuracy**"

as well as "我不会说“1,369 条被真人证明 100% 正确”" — and the bounded fix specification: rescue the
7 local-window misses to ±2 only when needed, neighbor-anchor fallback with explicit provenance for
the anchorless empties, and "把当前 **23 条失败全部做成 regression cases**" plus
"**不重新做 alignment model experiment**" — all verified exact.)

**Agent response / proposed plan.** Opened with verification, not implementation: "我先对照代码和
实际 CSV 数据验证你的三个判断…再给方案". Traced CSV → master TSV → `candidates.py`
(`if not anchor_ids: return []` — confirming the anchorless→empty mechanism and the exact counts
EN=10/ZH=9/overlap=2) → aligned JSONL → segmented JSONL. Corrected two of the researcher's claims
(`#NAME?` is Excel parsing `- `-prefixed cells as formulas, not an export bug; fixed ±2 widening
cannot rescue Ch.14 `im Schrank`, whose true ZH locus sits at anchor +5). During per-case tracing it
found the source-data bug: the ZH Ch.17 segmented file ends at p0205_s002 → "这不是检索窗口问题，
是语料截断。查 config 的页码范围" (14:23Z) → `config/hp1_zh_ch17.yaml` end_page 231 with a comment
claiming p.232 is a colophon → direct PyMuPDF read of pp.230–232 proved p.232 carries the novel's
final paragraph → "根因确认——**是 config bug，不是检索问题**" (14:24:45Z).

**What was actually implemented.** `b691f9e`: end_page 231→232 + corrected provenance comment;
re-run render→clean→segment; re-align DE–ZH Ch.17 only. `f3d4ee4` (13 files, +837): provenance columns
`en/zh_context_provenance` ∈ {anchor_window, merge_widened, neighbor_fallback, manual_review},
widening for N:M-merged anchors, neighbor-bracket fallback, excel-safe CSV writer, and the regression
gate `scripts/check_context_regression.py` + `data/derived/annotation/context_regression.json`
(23 cases / 26 sides, IDs only, gitignored by the researcher's explicit choice over the agent's
tracked-fixture recommendation). `9fa3a79` followed the researcher's L793 question: heuristic widening
for under-segmented DE segments and low-confidence anchors.

**Evidence returned.** Before/after: ZH Ch.17 330→346 sentences with existing IDs stable; the two
reviewed Ch.17 rows auto-repaired into clean 1:1 anchors (conf 0.93/0.69); empty-context rows
17→0; regression gate 26 sides = 21 in-context + 5 manual_review (later 25+1 after `9fa3a79`);
408→414 tests; honest leftover disclosed (ZH Ch.17 Stanza parse lacks the 16 new tail sentences —
unused by the annotation pipeline).

**Human decision.** Accepted ("good 先commit吧" — L766, verified exact) but immediately demanded an
eyeball-level root-cause walkthrough of the 5 manual_review rows before being satisfied; then approved
heuristic widening (fixes 2+3) while deferring DE re-segmentation ("我觉得2和3可以实现，而这个De对话切分
你也思考一下…导致了产生+5的这样偏差" — L793, verified exact). The follow-up analysis showed Ch.14's +5
offset was DP 3:1 member mis-anchoring plus the ZH translation folding the idiom into the next sentence
— not a segmentation bug.

**Companion case (same trace-first pattern).** 2026-08-17, session `a1f40aac` L7 — verified exact:
"i now give you a ocred version of the german text by gpt please help me check how does it compare
with our own ocr or proceed raw text version and which method do you think is better in terms of
accuracy?" The agent built `tmp/compare_gpt_ocr.py`, traced every difference to a specific `clean.py`
function/line (99.97% identical; 0 word substitutions; our 184 un-joined hyphenations vs GPT's 16;
our bbox-outlier filter was dropping the centered Ch.3 envelope block). Verdict: keep the
deterministic text layer as source of record; fix the two `clean.py` bugs instead — fixes landed in
`db83cd0`/PR #6 and `2cd15d8`. The researcher then narrowed branch policy: "如果你这个缺陷不大，可以
直接在main上面做修正，不要单独开分支了" (L514, verified exact), and deferred the full re-run.

**Evidence locations.** Sessions `e7469420` (L7, L766, L793; trace at L536–L564), `a1f40aac` (L7,
L191, L514); commits `b691f9e`, `f3d4ee4`, `9fa3a79`, `db83cd0`, `c6d679a`, `2cd15d8`;
memory `project_context_regression_gate.md`, `project_gpt_ocr_comparison.md`;
`config/hp1_zh_ch17.yaml`, `data/derived/annotation/context_regression.json`.

---

## Interaction 5 — Regression-driven convergence of the annotation-handoff context policy (Aug 19–24)

**Research problem.** The annotator-facing CSV must give a human enough local context to mark the
EN/ZH counterpart of each German PP — without rows becoming unreadably long, and without unreliable
machine contexts masquerading as reliable ones. Four commits over five days converged on the final
policy: routine ±1; bounded widening ≤ ±2 inside a 5-segment reading budget; low-confidence or
unreliable-provenance rows deferred to a companion CSV.

**Actual user instruction — the final policy, with an explicit no-scope-creep clause**
(session `31c41314`, L7, 2026-08-24T10:16:30Z — verified exact, complete):

```
Work from the latest main of hongyi-yang42/hp-transmining.

This is a small alignment-handoff cleanup. Do not redesign the pipeline, change LaBSE/DP, add new schemas, or expand the scope.

1. Bound the annotator-facing context

Current behavior:

routine anchor → ±1 target segment;
merge / suspected DE under-segmentation / machine confidence <0.50 → ±3.

Change this to:

routine case → keep ±1;
widened/suspicious case → at most ±2;
when widening, use a simple 5-target-segment reading budget;
always preserve the strict anchor group; the budget limits added neighbouring context, not the anchor itself;
do not extend to ±3/±4/±5 just to rescue difficult tail cases.

The goal is not to improve the embedding or strict DP alignment. It is only to give the annotator enough local context without making each row unnecessarily long.

2. Route unresolved machine pairs to the companion CSV

The companion CSV is simply the set of pairs that, at the current machine-alignment stage, we do not want in the clean teacher-facing CSV.

Keep the existing <0.40 confidence rule, but also defer a row when either EN or ZH retrieval provenance indicates that the machine does not have a normal reliable anchor/context:

manual_review
neighbor_fallback

Do not describe these rows as permanently excluded from the study. They are just separated at the current stage for later inspection.

Do not add a new elaborate reason taxonomy unless required by the existing validator. check and think about the plan. can you understand?
```

Earlier links in the chain (session `3e10af2d`, L222, 2026-08-23T16:36:13Z — verified exact):

```
先不动了，因为我看这个pivot好像也需要人工验证。我们决定你先不要把低confidence的alignment内容放进最终的标注csv了，思考一下如何加入这个代码逻辑和应当设定的限额。并更新一版新的annotation汇报区别，可以把低置信度的alignment单独输出一个csv方便我们后续分析和eyeballing保存，最后给我file dir
```

**Agent response / proposed plan.** For the confidence split: tested thresholds empirically against
the 5 eyeball-confirmed bad rows and form balance — either<0.35 → 55 rows but **misses the Ch.16 row
at conf 0.39**; either<0.40 → 88 rows, 5/5 known-bad covered, contracted share 48.9% vs 52.2% (no
form bias); either<0.50 → 201 rows, over-cutting correct rows. Chose 0.40. For the bounded window:
simulated the new rule against the 23-case regression manifest **before** coding ("zero breakage —
the one at-risk side is already manual_review"); the simulation matched the post-implementation
rebuild exactly (1303 kept / 89 deferred).

**What was actually implemented.** `2bb6eb8` (Aug 24): `split_low_confidence()` in
`src/hp_corpus/annotation_csv.py`; builder/validator paired flags; the validator re-derives the split
from the master so the two files cannot redefine their own row sets. `85a1e01` (Aug 24):
`budgeted_window()` in `candidates.py`; `WIDENED_CONTEXT_WINDOW=2`, `WIDENED_CONTEXT_BUDGET=5`;
deferral = confidence <0.40 OR provenance ∈ {manual_review, neighbor_fallback}; no new columns or
taxonomy. Widened windows dropped from 4–11 segments (mostly 7) to ≤5.

**Evidence returned.** Validator 0 violations (1303 kept + 89 deferred); context regression gate
unchanged at 25 in-context + 1 manual_review; 421→432 tests; 55/88 deferred rows carry
merge_widened provenance, matching the granularity-mismatch diagnosis.

**How specific regression cases constrained the implementation.**

- **Ch.14 `im Schrank`** (`dp_ch14_...`, zh expected locus at anchor +5): proved ±2 rescue impossible
  end-to-end on Aug 19 → routed to `manual_review`; under the final policy, manual_review rows defer
  to the companion CSV instead of inflating windows.
- **Ch.15 `bei dem Aufruhr**' (`dp_ch15_hp1_de_ch15_p0059_s001#b001_t15-17`, zh
  `neighbor_fallback`): same routing — unreliable provenance defers rather than pretends.
- **Ch.16 conf 0.39**: ruled out the 0.35 threshold (a known-wrong row would have stayed in the main
  CSV).
- The Aug-19 ±3 widening the researcher had approved was **reversed** on Aug 24 by the same
  researcher's own instruction ("do not extend to ±3/±4/±5 just to rescue difficult tail cases").

**Did the researcher explicitly forbid broadening?** Yes — twice, verbatim: "This is a small
alignment-handoff cleanup. Do not redesign the pipeline, change LaBSE/DP, add new schemas, or expand
the scope." (above) and on Aug 19: "我不会改 LaBSE" / "不重新做 alignment model experiment" (Interaction 4).

**Human decision.** Accepted; then "sync up to the origin/main. and then regenerate the final
annotation csv by the new pipeline? and lead me to where they are stored and things changed" (L369,
verified exact) — agent pushed, regenerated both CSVs and both gates, and walked the researcher to the
files. Note the deliberate framing the researcher imposed: deferred rows are "just separated at the
current stage for later inspection", not excluded from the study — encoded in the commit message
("The machine master keeps every row").

**Evidence locations.** Sessions `e7469420` (Aug 19), `3e10af2d` (L222, L443, L475), `31c41314`
(L7, L369); commits `f3d4ee4`, `9fa3a79`, `2bb6eb8`, `85a1e01`; `docs/ANNOTATION_CSV.md`,
`docs/ALIGNMENT_MODEL_DECISION.md` (split + window-tightening sections);
`data/derived/annotation/annotation_pairs*.csv`, `context_regression.json`.

---

## Interaction 6 — A rejected over-broad design: "direction convergence" deletes the sampling framework and 12 validation scripts (Aug 17)

**Research problem.** The agent had built (and the researcher had earlier tolerated) an elaborate
sampling/annotation apparatus: Mode A/B/C sampling framework, S0–S6 selection waterfall, noun-only
counterfactual projection, 12 multi-stage annotation scripts (batches, dual-annotator comparison,
adjudication ledger, gold merge, template refresh, ...). The researcher decided the design had
over-grown the actual research need and cut it wholesale.

**Actual user instruction** (session `5c714abf`, L2816, 2026-08-17T00:49:51Z — verified fragments):

```
继续处理仓库 hongyi-yang42/hp-transmining 的 Draft PR #5，但先确认远端 head 仍为 42a55446d5a9f6c1a5b9d93960f32aa3b15e68d6。保持 Draft、不得 merge。不要访问 Notion、agent memory 或任何凭据。

这是一次方向收敛，不是给现有设计继续加补丁。用户已经做出以下决定：

1. noun-only counterfactual 完全删除。删除 264、1,062、delta 204，以及所有计算代码、summary 字段、测试、文档和 PR body 引用。
2. 删除 Mode A/B/C 体系。Mode C 的 n=400、frame freeze、strata、group floor、support closure、witness ladder、reserve、replacement、power heuristic 全部删除。
3. 删除 production all-include projection。未完成 German review 时不能生成虚拟 target、projection ledger 或投影数字。
4. German review 将对机器候选池全量进行。
5. German review 完成后，正式选择只运行一次。
[...]
本 PR 只完成“German review 后的正式 eligible-pool 规则”，不要实现 Excel、annotation batches、比例抽样或 EN/ZH 标注。
```

And on the deliverable itself (L3194, 2026-08-17T03:06:57Z — verified exact):

```
那么我同意你对於这三个问题地解决方案。然后其实从根本上来讲，我觉得都不需要这么多验证脚本？或者说我只需要你做到标注出德文-中文-英文的pair csv就好了，给到我们的标注人员。另外也不需要提供过多的冗杂的内部代号信息（对标注帮助很小的列不需要放进来），另外改成csv来存储这个最终交给标注人员的文档。你能够理解我说的吗？你propose什么样的改变？
```

**Agent response / proposed plan.** Acknowledged the core demand ("最终交付物就是一个给标注人员的
德-英-中三语 pair CSV，过程性的验证脚本和内部代号列都是噪音，能砍就砍"), inventoried the 12
legacy scripts via a subagent, verified the machine master's 1,392 rows are 100% EN+ZH-aligned so a
single CSV could cover the pool, and asked four structured questions (single-pass CSV? delete all?
keep row_hash? minimal fields?) — the researcher picked the recommended options and customized the
fourth to "最小字段+alignment置信度".

**What was actually implemented.** Commit `fcfa9f0` (43 files, +2,908/−10,943; PR #5, merged
`39c18e0`): deleted the Mode A/B/C framework, the noun-only counterfactual, the projection mode, the
12 annotation-machinery scripts (~2,876 test lines and 3 docs with them), and the dual lemma columns
(single `de_corrected_head_lemma`); added `src/hp_corpus/sampling.py` + `build_eligible_pool.py` with
13 fail-closed gates, and `annotation_csv.py` + `build_annotation_csv.py` + `validate_annotation_csv.py`
for the single trilingual pair CSV (21 columns, BOM+CRLF for Excel).

**Evidence returned.** 440 local tests; CI green (438 passed / 2 skipped); leakage guard 13 passed;
real-data extraction↔master set consistency (1,394/1,392/2 um-vor difference); CLI exits 2 with zero
output when review is incomplete; the real deliverable generated (1,392 rows, 724 contracted / 668
uncontracted), validator 0 violations in template state.

**Human decision.** Rejected the over-broad design wholesale and re-centered it on one deliverable —
then personally code-reviewed the amended PR and blocked the merge on three fail-open defects it found
(L3065, verified fragments: the CLI accepted arbitrary `--chapters` subsets; the selector failed open
on incomplete review; silent master-duplicate overwrite) — "你看一下这些问题是否存在？如果存在，
是不是大问题？如何修正？" All three confirmed and fixed before merge. The narrowed design stuck: all
later rounds (Aug 19 context columns, Aug 24 confidence split) refined this single CSV rather than
re-growing machinery.

**Evidence locations.** Session `5c714abf` (L2816, L3065, L3194, L3210); commits `fcfa9f0`,
`39c18e0`; `docs/ANALYSIS_SAMPLE_DESIGN.md`, `docs/ANNOTATION_CSV.md`; memory
`project_sampling_mode_decision.md`.

---

## Interaction pattern (inferred only from the six interactions above)

The recurring loop, visible in all six:

```
research question / constraint (often decided offline by the researcher)
  → written task spec: goal + explicit constraints + acceptance criteria + stop gate
  → agent verifies/reads first (read-only investigation, input gates, refusal to fabricate)
  → human scope decision (approve / narrow / reject — sometimes in two stages)
  → bounded implementation: scoped commits, synthetic-fixture tests, no data files in git
  → concrete evidence returned: counts, manifests, gates, test totals — aggregate only
  → human acceptance keyed to the numbers, often with a follow-up "why does this fail?" question
  → failures codified as regression cases / gates
  → next bounded revision
```

Human authority concentrated at four points: **what the task is** (specs were typed by the researcher,
often with fail-closed and no-fabrication clauses), **what counts as acceptable** (acceptance criteria
and stop gates in the prompt; approval gates; the personal audit that caught six schema/manifest
defects on Aug 19), **what is deliberately excluded** (LaBSE untouched during retrieval fixes; no gold
writing by the agent; deferred ≠ excluded), and **when accumulated machinery gets deleted** (Inter-
action 6). The agent's distinctive contributions were breadth of execution (17-chapter runs, 34/34
manifests, blind judge fleets) and honest bookkeeping (needs_review flags, provenance columns,
pre-registered gates it authored inside human-approved plans).

Important variations, all evidenced above:

- **Direct small fix on main, no branch** — the researcher explicitly relaxed its own earlier
  branch/PR rule when the change was small ("如果你这个缺陷不大，可以直接在main上面做修正，不要单独
  开分支了", Aug 17), and deferred the full re-run because alignment work was in flight.
- **Stopping an over-broad plan before implementation** — the Aug 11 plan-mode rejection (session
  `c4eab9ff` L152: "这份计划总体可以实施，但不建议原样提交" followed by six mandatory corrections,
  delivered through the plan-rejection dialog; revised plan then approved and implemented as `4dde93d`);
  the Aug 17 direction-convergence deletion; the Aug 23 pivot shelving ("先不动了").
- **Re-running only what is affected** — ZH Ch.17 re-segment + DE–ZH Ch.17 re-align only;
  "不重跑无关部分" in the Source Option A approval; the researcher correcting the agent's framing that
  "9 alignments" meant Ch.1–3 × 3 pairs, not Ch.4–17.
- **Rejected alternatives preserved as diagnostics, not deleted** — ratio scoring + two-pass anchor
  DP landed as default-off switches after failing their pre-declared gate; EN–ZH kept under
  `--diagnostics`; bge-m3 kept as a local backup model; the EN-pivot rescue shelved pending judge
  validation; deferred CSV rows kept in a companion file for later adjudication.
- **A human reversal of an earlier acceptance** — the ±3 widening approved on Aug 19 was bounded back
  to ±2/5-budget by the same researcher on Aug 24.

---

## Evidence index

**Sessions** (first 8 chars; full UUIDs are the `.jsonl` filenames in the Claude Code project dir):

| Session | Date span | Used for |
|---|---|---|
| `69b58362` | Aug 7–8 | Interaction 1 (investigation + authorization) |
| `cbf32935` | Aug 7 | Interaction 1 context (1998 PDF deposit) |
| `64168855` | Aug 9 | Full-novel scope decision pasted in (Interaction 3 layer 1) |
| `c4eab9ff` | Aug 11 | Plan-mode six-corrections rejection (pattern section) |
| `716b7948` | Aug 14–15 | Interaction 3 (Track A prompts); ZH PDF arrival |
| `8ecd4b3d` | Aug 15 | ZH Source Option A memo → approval → execution |
| `3ee7a490` | Aug 15 | Machine-corpus freeze + parity-audit mandate |
| `5c714abf` | Aug 15–17 | Interaction 6; GPT-OCR era context |
| `a1f40aac` | Aug 17 | GPT-OCR audit (Interaction 4 companion) |
| `11beee98` | Aug 17–18 | Interaction 2 (proxy challenge → LaBSE) |
| `12f7c4eb`, `72dc250e` | Aug 19 | Post-switch six-point audit; decision record |
| `e7469420` | Aug 19 | Interactions 4 & 5 (ch17 trace, rescue) |
| `3e10af2d` | Aug 23–24 | Pivot shelving; confidence split |
| `31c41314` | Aug 24 | Interaction 5 final policy |

**Key commits.** Interaction 1: `be80ec7` `c0dab3f` `3ebfdf7` `57746ed` · Interaction 2: `ea974b3`
`d6ece16` `fe4d7e4` `9e6ed21` `326b052` `023c76e` `11819f1` `991f473` `3868726` `8592787` `7605c87`
`55d1128` `52ba80d` `450b396` · Interaction 3: `1f971f3` `d17dd85` (`19f6819` machine master, `eaaac1d`
ZH source switch) · Interaction 4: `b691f9e` `f3d4ee4` `9fa3a79` `db83cd0` `c6d679a` `2cd15d8` ·
Interaction 5: `2bb6eb8` `85a1e01` · Interaction 6: `fcfa9f0` `39c18e0` (plan-rejection follow-up:
`4dde93d` `00efed0`).

**Docs.** `docs/ALIGNMENT_MODEL_DECISION.md` (model switch + split + window-tightening records);
`docs/ANNOTATION_CSV.md`; `docs/ANALYSIS_SAMPLE_DESIGN.md`; `docs/FULL_NOVEL_SAMPLING.md`;
`docs/METHODS.md`.

**Local (gitignored) artifacts.** `data/derived/alignment_metric_review/` (model_comparison.md,
gold2/, judge_results_182.csv, review50.csv); `data/derived/annotation/context_regression.json`
(23 cases); `data/derived/annotation/annotation_pairs.csv` + `annotation_pairs_low_confidence.csv`;
`data/derived/source_switch_1998/`; `data/derived/zh_source_audit/` (SOURCE_DECISION_MEMO.md,
manifests, legacy_archive/); `data/extracted/full_novel/manifest.json`;
`data/derived/step4/full_novel_annotation_master.tsv`.
