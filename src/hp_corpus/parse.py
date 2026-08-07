"""UD parsing via Stanza → CoNLL-U.

Input: segmented JSONL (one Segment per line).
Output: CoNLL-U file with one UD tree per Segment.

Each UD sentence carries two comment lines:
    # sent_id = <segment_id>
    # text = <segment text>

Stanza is lazy-imported so unit tests for the rest of the package don't
require the ~1.2 GB of model files or the stanza/torch deps.

Model setup
-----------
Models are NOT bundled in the repo (per .gitignore policy). The expected
layout under ``~/stanza_resources/`` is::

    resources.json
    <lang>/<processor>/<package>.pt

For our 3 languages the default_processors from Stanza 1.14 are:

    de        tokenize=combined, mwt=combined, pos=combined_charlm,
              lemma=combined_nocharlm, depparse=combined_charlm
    en        tokenize=combined_nocharlm, mwt=combined,
              pos=combined_charlm, lemma=combined_nocharlm,
              depparse=combined_charlm
    zh-hans   tokenize=gsdsimp, pos=gsdsimp_charlm,
              lemma=gsdsimp_nocharlm, depparse=gsdsimp_charlm

The modelscope mirror (modelscope.cn/stanfordnlp/stanza-<lang>) hosts the
same model files as Hugging Face but is reachable from CN where HF is
blocked. As of 2026-08, modelscope's recent uploads use dicts_version=3
(JSON-in-gzip) for the lemma files while Stanza 1.14.0 only knows
versions 1 (legacy tuple) and 2 (pickle-in-gzip). The fix is a one-time
in-place rewrite of the lemma .pt files; see
``scripts/patch_stanza_lemma_format.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Per-language Stanza processor config. Chinese has no MWT (multi-word
# token expansion); German and English do.
_PROCESSORS = {
    "de": "tokenize,mwt,pos,lemma,depparse",
    "en": "tokenize,mwt,pos,lemma,depparse",
    "zh": "tokenize,pos,lemma,depparse",
}


def _stanza_resources_dir() -> Path:
    """Resolve the stanza_resources dir, honoring $STANZA_RESOURCES_DIR."""
    env = os.environ.get("STANZA_RESOURCES_DIR")
    if env:
        return Path(env)
    return Path.home() / "stanza_resources"


def parse_segments(
    segments_path: str | Path,
    lang: str,
    output_path: str | Path,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Parse segmented JSONL → CoNLL-U file. Returns metadata dict.

    Parameters
    ----------
    segments_path : input JSONL of Segment records
    lang : 'de' | 'en' | 'zh'
    output_path : where to write the .conllu file
    model_dir : optional override for ~/stanza_resources
    """
    # Lazy import — stanza + torch are heavy.
    import stanza

    resources_dir = Path(model_dir) if model_dir else _stanza_resources_dir()
    # Force offline mode so stanza never tries to reach the blocked HF host.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("STANZA_RESOURCES_DIR", str(resources_dir))

    processors = _PROCESSORS.get(lang)
    if processors is None:
        raise ValueError(f"unsupported lang {lang!r}; expected one of {list(_PROCESSORS)}")

    nlp = stanza.Pipeline(
        lang=lang,
        processors=processors,
        dir=str(resources_dir),
        download_method=None,  # never reach network
        verbose=False,
    )

    segments: list[dict] = []
    with open(segments_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                segments.append(json.loads(line))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_sentences = 0
    n_tokens = 0
    n_mwt = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for seg in segments:
            text = seg["text"]
            sid = seg["id"]
            doc = nlp(text)
            for sentence in doc.sentences:
                # CoNLL-U comment lines carry the segment ID + original text.
                fout.write(f"# sent_id = {sid}\n")
                fout.write(f"# text = {text}\n")
                # Iterate tokens (not words) so we can emit the CoNLL-U MWT
                # range line (e.g. "5-6	im	_	_	…") when a token was
                # split by the MWT processor. This preserves the contracted
                # surface form ("im") alongside its components ("in"+"dem"),
                # which the Bremmers et al. extraction needs.
                for token in sentence.tokens:
                    words = token.words
                    if len(words) == 1:
                        w = words[0]
                        feats = w.feats or "_"
                        xpos = w.xpos or "_"
                        deps = w.deps or "_"
                        misc = w.misc or "_"
                        fout.write(
                            f"{w.id}\t{w.text}\t{w.lemma}\t{w.upos}\t"
                            f"{xpos}\t{feats}\t{w.head}\t{w.deprel}\t"
                            f"{deps}\t{misc}\n"
                        )
                    else:
                        # MWT range line: id-id <surface> _ _ ... _
                        first_id = words[0].id
                        last_id = words[-1].id
                        # Stanza token.text is the original surface form
                        # ("im"); words[].text are the expanded pieces.
                        fout.write(
                            f"{first_id}-{last_id}\t{token.text}\t_\t_\t_\t_\t_\t_\t_\tMWT=Yes\n"
                        )
                        n_mwt += 1
                        for w in words:
                            feats = w.feats or "_"
                            xpos = w.xpos or "_"
                            deps = w.deps or "_"
                            misc = w.misc or "_"
                            fout.write(
                                f"{w.id}\t{w.text}\t{w.lemma}\t{w.upos}\t"
                                f"{xpos}\t{feats}\t{w.head}\t{w.deprel}\t"
                                f"{deps}\t{misc}\n"
                            )
                    n_tokens += len(words)
                fout.write("\n")
                n_sentences += 1

    return {
        "input_segments": len(segments),
        "output_sentences": n_sentences,
        "output_tokens": n_tokens,
        "n_mwt": n_mwt,
        "lang": lang,
        "path": str(out_path),
    }
