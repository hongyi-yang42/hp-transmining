"""Unified OCR: PaddleOCR for image-only scans, PyMuPDF for embedded text layers.

Output is OCRBlock JSONL regardless of engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .render import extract_text_layer_blocks, has_text_layer
from .schema import OCRBlock

_console = Console()


def _make_paddleocr(lang: str):
    """Construct a PaddleOCR instance. Imported lazily so unit tests don't require it.

    PaddleOCR 3.x dropped ``use_angle_cls`` and ``show_log``; the modern equivalent
    is ``use_textline_orientation``. We also disable the doc-preprocessor
    (orientation classify + unwarping) since our scans are already upright.
    """
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


def _poly_to_bbox(poly) -> list[float]:
    """Convert a PaddleOCR polygon (nx2 array/list of [x, y]) to [x0, y0, x1, y1]."""
    if poly is None:
        return [0.0, 0.0, 0.0, 0.0]
    try:
        # PaddleOCR 3.x returns numpy arrays.
        arr = __import__("numpy").asarray(poly)
        if arr.size == 0:
            return [0.0, 0.0, 0.0, 0.0]
        xs = arr[:, 0]
        ys = arr[:, 1]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    except Exception:  # noqa: BLE001 — fall back to flat-list interpretation
        if len(poly) == 0:
            return [0.0, 0.0, 0.0, 0.0]
        if len(poly) == 4 and not hasattr(poly[0], "__len__"):
            return [float(c) for c in poly]
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def ocr_image(paddle_instance, image_path: str | Path, page: int) -> list[OCRBlock]:
    """Run PaddleOCR on one image; return OCRBlocks with bbox + confidence.

    Handles PaddleOCR 3.x: ``result[0]`` is an OCRResult dict-like with
    ``rec_texts``, ``rec_scores``, ``rec_polys``.
    """
    result = paddle_instance.predict(str(image_path))
    if not result:
        return []
    page_result = result[0]
    texts = list(page_result.get("rec_texts", []) or [])
    scores = list(page_result.get("rec_scores", []) or [])
    polys = list(page_result.get("rec_polys", []) or [])

    blocks: list[OCRBlock] = []
    for block_idx, (text, score, poly) in enumerate(zip(texts, scores, polys, strict=False)):
        blocks.append(
            OCRBlock(
                page=page,
                block_idx=block_idx,
                text=str(text),
                bbox=_poly_to_bbox(poly),
                confidence=float(score),
                engine="paddleocr",
            )
        )
    return blocks


def ocr_pages_paddleocr(
    image_paths: list[Path],
    lang: str,
) -> list[OCRBlock]:
    """Run PaddleOCR over a list of pre-rendered page images."""
    paddle = _make_paddleocr(lang)
    all_blocks: list[OCRBlock] = []
    for img in image_paths:
        page_no = int(img.stem.split("_p")[-1])
        all_blocks.extend(ocr_image(paddle, img, page_no))
    return all_blocks


def ocr_pdf_text_layer(
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
) -> list[OCRBlock]:
    """Extract blocks via PyMuPDF's text layer."""
    return extract_text_layer_blocks(pdf_path, start_page, end_page)


def write_jsonl(blocks: list[OCRBlock], output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(b.model_dump_json() + "\n")


def read_jsonl(input_path: str | Path) -> list[OCRBlock]:
    blocks: list[OCRBlock] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            blocks.append(OCRBlock.model_validate_json(line))
    return blocks


def confidence_summary(blocks: list[OCRBlock]) -> dict[str, float | int]:
    """Aggregate statistics. Returns only numbers — no text content."""
    if not blocks:
        return {"count": 0}
    confs = [b.confidence for b in blocks]
    sorted_c = sorted(confs)
    n = len(sorted_c)
    return {
        "count": n,
        "mean": sum(confs) / n,
        "median": sorted_c[n // 2],
        "p10": sorted_c[max(0, int(n * 0.10) - 1)],
        "below_0p6": sum(1 for c in confs if c < 0.6),
    }


def print_confidence_table(blocks: list[OCRBlock], label: str) -> None:
    s = confidence_summary(blocks)
    if s["count"] == 0:
        _console.print(f"[yellow]{label}: no blocks[/]")
        return
    table = Table(title=f"OCR confidence — {label}", show_header=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in s.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
        else:
            table.add_row(k, str(v))
    _console.print(table)


def run_ocr(config: dict[str, Any], pages_dir: str | Path) -> list[OCRBlock]:
    """Dispatch OCR based on config. Uses pre-rendered PNGs for paddleocr."""
    engine = config["ocr"]["engine"]
    ocr_lang = config["ocr"]["lang"]  # PaddleOCR's lang code (e.g. "ch")
    book_lang = config["lang"]  # top-level lang code used in filenames (e.g. "zh")
    start = config["chapter"]["start_page"]
    end = config["chapter"]["end_page"]
    pdf_path = config["pdf_path"]

    if engine == "pymupdf":
        return ocr_pdf_text_layer(pdf_path, start, end)

    if engine == "paddleocr":
        return ocr_pages_paddleocr(
            _select_page_images(pages_dir, config["book"], book_lang, start, end),
            ocr_lang,
        )

    if engine == "auto":
        import fitz

        with fitz.open(pdf_path) as doc:
            return (
                ocr_pdf_text_layer(pdf_path, start, end)
                if has_text_layer(doc)
                else ocr_pages_paddleocr(
                    _select_page_images(pages_dir, config["book"], book_lang, start, end),
                    ocr_lang,
                )
            )

    raise ValueError(f"Unknown ocr.engine: {engine}")


def _select_page_images(
    pages_dir: str | Path,
    book: str,
    lang: str,
    start: int,
    end: int,
) -> list[Path]:
    pages = Path(pages_dir)
    wanted = set(range(start, end + 1))
    images = sorted(pages.glob(f"{book}_{lang}_p*.png"))
    selected = [p for p in images if int(p.stem.split("_p")[-1]) in wanted]
    if not selected:
        raise FileNotFoundError(
            f"No rendered PNGs found in {pages} for pages {start}-{end}. "
            "Run `hp-corpus render` first."
        )
    return selected


__all__ = [
    "OCRBlock",
    "ocr_pdf_text_layer",
    "ocr_pages_paddleocr",
    "write_jsonl",
    "read_jsonl",
    "confidence_summary",
    "print_confidence_table",
    "run_ocr",
]
