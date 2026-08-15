"""PDF rendering and source validation."""

from __future__ import annotations

import hashlib
import statistics
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import yaml
from pydantic import BaseModel

from .schema import OCRBlock, OCRLine, OCRSpan


class ValidationError(Exception):
    """Raised when a source file fails validation."""


class SourceReport(BaseModel):
    """Metadata-only report for one source. No text snippets."""

    document_id: str
    lang: str
    path: str
    exists: bool
    size_bytes: int
    expected_sha256: str | None = None
    actual_sha256_prefix: str
    sha256_ok: bool
    expected_total_pages: int
    actual_total_pages: int
    page_count_ok: bool
    expected_has_text_layer: bool
    actual_has_text_layer: bool
    text_layer_ok: bool


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_text_layer(doc: fitz.Document, sample_pages: int = 5) -> bool:
    """Return True iff the PDF has a usable embedded text layer.

    Samples `sample_pages` evenly-spaced pages; returns True iff the median
    extracted-text length exceeds 50 chars. Median (not min/max) tolerates
    occasional blank pages like title/section dividers.
    """
    if doc.page_count == 0:
        return False
    indices = [
        int(i * (doc.page_count - 1) / max(sample_pages - 1, 1)) for i in range(sample_pages)
    ]
    lengths = [len(doc[i].get_text().strip()) for i in indices]
    return statistics.median(lengths) > 50


def validate_source(config: dict[str, Any]) -> SourceReport:
    """Validate one source PDF against its manifest entry. Raises ValidationError on mismatch.

    `expected_sha256` is optional: when absent, the hash check is skipped
    (file-existence, page-count, and text-layer checks still run). This
    lets public configs ship without source-PDF fingerprints; users who
    want hash verification can populate the field in a gitignored local
    override (e.g., `config/hp1_de.local.yaml`) and pass `--config` to it.
    """
    path = config["pdf_path"]
    expected_sha = config.get("expected_sha256")
    expected_pages = config["total_pages"]
    expected_layer = config["has_text_layer"]
    document_id = f"{config['book']}_{config['lang']}"

    p = Path(path)
    exists = p.exists()
    size_bytes = p.stat().st_size if exists else 0

    if not exists:
        raise ValidationError(f"Source file not found: {path}")

    if expected_sha:
        actual_sha = sha256_file(p)
        actual_sha_prefix = actual_sha[:12]
        sha_ok = actual_sha == expected_sha
        sha_problem = (
            f"sha256 mismatch (expected {expected_sha[:12]}…, got {actual_sha_prefix}…)"
            if not sha_ok
            else ""
        )
    else:
        actual_sha_prefix = ""
        sha_ok = True
        sha_problem = ""

    page_count_ok = False
    actual_pages = 0
    layer_ok = False
    actual_layer = False

    with fitz.open(p) as doc:
        actual_pages = doc.page_count
        page_count_ok = actual_pages == expected_pages
        actual_layer = has_text_layer(doc)
        layer_ok = actual_layer == expected_layer

    report = SourceReport(
        document_id=document_id,
        lang=config["lang"],
        path=str(p),
        exists=exists,
        size_bytes=size_bytes,
        expected_sha256=expected_sha,
        actual_sha256_prefix=actual_sha_prefix,
        sha256_ok=sha_ok,
        expected_total_pages=expected_pages,
        actual_total_pages=actual_pages,
        page_count_ok=page_count_ok,
        expected_has_text_layer=expected_layer,
        actual_has_text_layer=actual_layer,
        text_layer_ok=layer_ok,
    )

    problems = [
        f
        for f in (
            sha_problem,
            not page_count_ok
            and f"page count mismatch (expected {expected_pages}, got {actual_pages})",
            not layer_ok and f"text-layer mismatch (expected {expected_layer}, got {actual_layer})",
        )
        if f
    ]
    if problems:
        raise ValidationError(f"Validation failed for {document_id}: " + "; ".join(problems))

    return report


def render_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    book: str,
    lang: str,
    dpi: int = 300,
    start_page: int | None = None,
    end_page: int | None = None,
) -> tuple[int, bool]:
    """Render the requested page range to PNG.

    Page indices in args are 1-indexed inclusive, matching the config convention.
    Returns (pages_rendered, has_text_layer).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        layer_present = has_text_layer(doc)
        start = (start_page or 1) - 1
        end = end_page or doc.page_count
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        count = 0
        for i in range(start, end):
            pix = doc[i].get_pixmap(matrix=matrix)
            pix.save(out / f"{book}_{lang}_p{i + 1:04d}.png")
            count += 1
        return count, layer_present


def extract_text_layer_blocks(
    pdf_path: str | Path,
    start_page: int,
    end_page: int,
) -> list[OCRBlock]:
    """Extract blocks from the embedded text layer for a 1-indexed page range.

    Used directly by ocr.py when engine == 'pymupdf'. Page numbers in the
    returned records are 1-indexed.

    Span-level metadata (font size, flags) is preserved on ``OCRBlock.lines``
    so downstream cleaning can separate footnote markers (superscript digits)
    and note bodies (small-print spans) from body text. PaddleOCR blocks
    carry no such metadata; their ``lines`` stay empty.
    """
    blocks: list[OCRBlock] = []
    with fitz.open(pdf_path) as doc:
        for page_idx in range(start_page - 1, min(end_page, doc.page_count)):
            page = doc[page_idx]
            page_no = page_idx + 1
            d = page.get_text("dict")
            block_idx = 0
            for b in d.get("blocks", []):
                if b.get("type") != 0:  # 0 == text block
                    continue
                bbox = list(b.get("bbox", [0, 0, 0, 0]))
                lines = b.get("lines", [])
                if not lines:
                    continue
                text_parts: list[str] = []
                line_meta: list[OCRLine] = []
                for line in lines:
                    spans = line.get("spans", [])
                    line_text = "".join(s.get("text", "") for s in spans)
                    if line_text:
                        text_parts.append(line_text)
                        line_meta.append(
                            OCRLine(
                                spans=[
                                    OCRSpan(
                                        text=s.get("text", ""),
                                        size=float(s.get("size", 0.0)),
                                        flags=int(s.get("flags", 0)),
                                    )
                                    for s in spans
                                    if s.get("text", "")
                                ]
                            )
                        )
                if not text_parts:
                    continue
                text = "\n".join(text_parts)
                blocks.append(
                    OCRBlock(
                        page=page_no,
                        block_idx=block_idx,
                        text=text,
                        bbox=bbox,
                        confidence=1.0,
                        engine="pymupdf",
                        lines=line_meta,
                    )
                )
                block_idx += 1
    return blocks
