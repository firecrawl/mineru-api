"""Warm-up utilities for CUDA kernel pre-compilation.

OCR-det groups detected text regions by their padded resolution (64-pixel
boundaries).  Each unique (height, width) pair triggers a CUDA kernel
compilation (~2 s per shape on first encounter).

Two warm-up strategies are provided:

1. **Synthetic PDF inference** (`create_warmup_pdf`) — runs the full pipeline
   to warm Layout, MFD, MFR, Table, and OCR models.  However, the OCR-det
   resolution groups produced by synthetic text rarely overlap with those
   from real-world PDFs.

2. **Direct OCR-det tensor warm-up** (`warmup_ocr_det_shapes`) — feeds dummy
   images at every common 64-px-padded resolution directly into the OCR-det
   model, guaranteeing kernel coverage regardless of document content.
"""

import io
import time


def warmup_ocr_det_shapes(lang="en"):
    """Pre-compile CUDA kernels for OCR-det across all common input resolutions.

    The synthetic PDF warm-up produces OCR-det resolution groups that don't
    overlap with real documents (observed: 28 warm-up shapes vs 37 production
    shapes with zero overlap).  This function bypasses the PDF pipeline and
    feeds dummy images directly into the OCR-det model at every 64-px-padded
    resolution in the typical text-region range.

    Grid: heights 64-512 (8 values) x widths 64-1280 (20 values) = 160 shapes.
    Shapes already compiled by the PDF warm-up run in ~0.02 s each (cached).
    New shapes compile at ~2 s each.  Typical additional time: 2-4 minutes.
    """
    import numpy as np
    from mineru.backend.pipeline.model_init import AtomModelSingleton
    from mineru.backend.pipeline.model_list import AtomicModel

    atom_model_manager = AtomModelSingleton()
    ocr_model = atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.3,
        lang=lang,
    )
    text_detector = ocr_model.text_detector

    # Build grid: every 64-px multiple in the common text-region range.
    # Heights 64-512 cover single lines through large paragraphs.
    # Widths 64-1280 cover narrow labels through full page-width blocks.
    shapes = [
        (h, w)
        for h in range(64, 513, 64)
        for w in range(64, 1281, 64)
    ]

    total = len(shapes)
    print(f"Pre-compiling OCR-det CUDA kernels for {total} resolution shapes...")
    start = time.time()

    for i, (h, w) in enumerate(shapes):
        dummy = np.ones((h, w, 3), dtype=np.uint8) * 255
        try:
            text_detector.batch_predict([dummy], 1)
        except Exception:
            pass
        if (i + 1) % 40 == 0:
            elapsed = round(time.time() - start, 1)
            print(f"  OCR-det warmup: {i + 1}/{total} shapes ({elapsed}s)")

    elapsed = round(time.time() - start, 1)
    print(f"OCR-det kernel pre-compilation complete: {total} shapes in {elapsed}s")


def _make_page_ops(page_idx):
    """Return PDF content-stream bytes for one warm-up page.

    Eight layout strategies rotate by *page_idx % 8* so that an 8-page
    document covers titles, columns, grids, scattered words, dense body
    text, and many font sizes — maximising the number of distinct
    OCR-det resolution groups produced.
    """
    ops = ["BT"]
    kind = page_idx % 8

    if kind == 0:
        # Large heading + subtitle + small body + footer
        ops.append("/F1 42 Tf 1 0 0 1 50 720 Tm (Document Title Heading) Tj")
        ops.append("/F1 20 Tf 1 0 0 1 50 670 Tm (A subtitle line in medium font) Tj")
        y = 610
        for i in range(12):
            ops.append(f"/F1 10 Tf 1 0 0 1 50 {y} Tm (Body text line {i} with enough words to span the page width nicely) Tj")
            y -= 14
        ops.append("/F1 7 Tf 1 0 0 1 50 80 Tm (Footer: page number and small disclaimer text) Tj")

    elif kind == 1:
        # Two-column layout — narrow region widths
        y = 750
        for i in range(30):
            ops.append(f"/F1 9 Tf 1 0 0 1 40 {y} Tm (Left column line {i} text) Tj")
            ops.append(f"/F1 9 Tf 1 0 0 1 320 {y} Tm (Right column line {i} text) Tj")
            y -= 12
            if y < 50:
                break

    elif kind == 2:
        # Scattered snippets at many different font sizes
        items = [
            (50, 740, 48, "HUGE"), (50, 680, 36, "Large Text"),
            (50, 630, 24, "Medium-Large"), (350, 740, 6, "tiny six pt"),
            (350, 720, 7, "seven pt text"), (350, 700, 8, "eight pt line"),
            (50, 580, 18, "Eighteen point"), (300, 580, 14, "Fourteen pt"),
            (50, 520, 11, "Eleven point body text line"),
            (300, 520, 10, "Ten pt right side"),
            (50, 460, 20, "Twenty pt block"), (350, 460, 9, "Nine pt note"),
            (50, 380, 16, "Sixteen"), (200, 380, 12, "Twelve"),
            (400, 380, 8, "Eight"), (50, 300, 32, "Big"),
            (50, 250, 7, "small footer note at bottom"),
        ]
        for x, y, s, t in items:
            ops.append(f"/F1 {s} Tf 1 0 0 1 {x} {y} Tm ({t}) Tj")

    elif kind == 3:
        # Alternating large/small blocks with big gaps
        y = 750
        for block in range(6):
            size = 28 if block % 2 == 0 else 8
            lines = 2 if block % 2 == 0 else 6
            for j in range(lines):
                text = f"Block {block} line {j} sample text content here"
                ops.append(f"/F1 {size} Tf 1 0 0 1 50 {y} Tm ({text}) Tj")
                y -= size + 3
            y -= 30  # gap between blocks
            if y < 50:
                break

    elif kind == 4:
        # Grid layout (table-like) — many small uniform crops
        y = 750
        for row in range(20):
            for col in range(5):
                x = 30 + col * 115
                ops.append(f"/F1 7 Tf 1 0 0 1 {x} {y} Tm (R{row}C{col} data) Tj")
            y -= 11
            if y < 40:
                break

    elif kind == 5:
        # Single isolated words scattered (many tiny regions)
        positions = [
            (60, 750), (200, 750), (400, 750), (500, 750),
            (80, 650), (250, 650), (450, 650),
            (60, 550), (180, 550), (330, 550), (480, 550),
            (100, 450), (300, 450), (500, 450),
            (60, 350), (220, 350), (400, 350),
            (80, 250), (250, 250), (450, 250),
            (60, 150), (200, 150), (370, 150), (500, 150),
        ]
        sizes = [8, 10, 12, 14, 16, 18, 9, 11, 13, 15, 20, 7,
                 22, 8, 10, 24, 6, 14, 9, 28, 11, 8, 16, 10]
        for (x, y_pos), s in zip(positions, sizes):
            ops.append(f"/F1 {s} Tf 1 0 0 1 {x} {y_pos} Tm (Word{s}) Tj")

    elif kind == 6:
        # Dense small text (many lines, forces long narrow crops)
        y = 770
        for i in range(55):
            ops.append(f"/F1 7 Tf 1 0 0 1 30 {y} Tm "
                       f"(Line {i:03d} dense text the quick brown fox jumps over the lazy dog abcdef 0123456789) Tj")
            y -= 10
            if y < 20:
                break

    else:
        # Full-width lines cycling through 15 different font sizes
        y = 760
        sizes = [6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 9, 11, 13, 15]
        for i, s in enumerate(sizes):
            indent = 40 + (i % 4) * 30
            ops.append(f"/F1 {s} Tf 1 0 0 1 {indent} {y} Tm "
                       f"(Size {s}pt sample line {i} with varied indent and content width) Tj")
            y -= s + 8
            if y < 40:
                break

    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def create_warmup_pdf(num_pages=8):
    """Build a synthetic PDF with *num_pages* diverse layouts.

    Returns raw PDF bytes ready to be fed into the MinerU pipeline.
    """
    page_obj_nums = []
    page_content_pairs = []
    obj_num = 4  # 1=Catalog, 2=Pages, 3=Font

    for p in range(num_pages):
        content_bytes = _make_page_ops(p)

        content_obj_num = obj_num
        page_obj_num = obj_num + 1
        page_content_pairs.append((content_obj_num, page_obj_num, content_bytes))
        page_obj_nums.append(page_obj_num)
        obj_num += 2

    total_objs = obj_num
    buf = io.BytesIO()
    offsets = {}

    def write(data):
        if isinstance(data, str):
            data = data.encode()
        buf.write(data)

    def start_obj(num):
        offsets[num] = buf.tell()
        write(f"{num} 0 obj\n")

    def end_obj():
        write(b"endobj\n")

    write(b"%PDF-1.4\n")

    start_obj(1)
    write(b"<</Type/Catalog/Pages 2 0 R>>\n")
    end_obj()

    kids = " ".join(f"{n} 0 R" for n in page_obj_nums)
    start_obj(2)
    write(f"<</Type/Pages/Kids[{kids}]/Count {num_pages}>>\n")
    end_obj()

    start_obj(3)
    write(b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\n")
    end_obj()

    for content_obj_num, page_obj_num, content_bytes in page_content_pairs:
        start_obj(content_obj_num)
        write(f"<</Length {len(content_bytes)}>>\nstream\n")
        buf.write(content_bytes)
        write(b"\nendstream\n")
        end_obj()

        start_obj(page_obj_num)
        write(f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
              f"/Contents {content_obj_num} 0 R"
              f"/Resources<</Font<</F1 3 0 R>>>>>>\n")
        end_obj()

    xref_offset = buf.tell()
    write(f"xref\n0 {total_objs}\n")
    write(b"0000000000 65535 f \r\n")
    for i in range(1, total_objs):
        write(f"{offsets[i]:010d} 00000 n \r\n")

    write(f"trailer<</Size {total_objs}/Root 1 0 R>>\n"
          f"startxref\n{xref_offset}\n%%EOF\n")

    return buf.getvalue()
