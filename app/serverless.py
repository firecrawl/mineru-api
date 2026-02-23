import base64
import os
import time
import asyncio
import tempfile
import copy
import io

import runpod

# New mineru imports
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.pipeline.pipeline_analyze import ModelSingleton

from pypdf import PdfReader, PdfWriter
from pypdfium2._helpers.misc import PdfiumError

class TimeoutError(Exception):
    pass

def _trim_pdf_to_max_pages(pdf_bytes: bytes, max_pages: int) -> bytes:
    """Return a new PDF bytes object with at most the first max_pages pages."""
    if max_pages is None or max_pages <= 0:
        return pdf_bytes

    input_buffer = io.BytesIO(pdf_bytes)
    reader = PdfReader(input_buffer)

    writer = PdfWriter()
    pages_to_write = min(max_pages, len(reader.pages))
    for page_index in range(pages_to_write):
        writer.add_page(reader.pages[page_index])

    output_buffer = io.BytesIO()
    writer.write(output_buffer)
    return output_buffer.getvalue()

def _repair_pdf(pdf_bytes: bytes) -> bytes:
    """Re-write the PDF through pypdf to fix structural issues (e.g. broken xref tables)."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()
    except Exception:
        # If repair itself fails, return original bytes and let the pipeline report the error
        return pdf_bytes

def _create_warmup_pdf(num_pages=4):
    """Create a synthetic multi-page PDF with dense text for CUDA kernel warm-up.

    Generates pages with many text lines at varied font sizes so the layout model
    detects numerous text regions, forcing OCR-det to process diverse tensor shapes
    and pre-compile CUDA kernels for them.
    """
    page_obj_nums = []
    page_content_pairs = []
    obj_num = 4  # 1=Catalog, 2=Pages, 3=Font

    for p in range(num_pages):
        ops = ["BT"]
        y = 760
        for i in range(40):
            size = 8 + (i % 5) * 2  # cycle 8, 10, 12, 14, 16 pt
            x = 40 + (i % 3) * 10
            text = f"P{p+1} L{i+1} The quick brown fox jumps over the lazy dog 0123456789"
            ops.append(f"/F1 {size} Tf 1 0 0 1 {x} {y} Tm ({text}) Tj")
            y -= size + 3
            if y < 40:
                break
        ops.append("ET")
        content_bytes = "\n".join(ops).encode("latin-1")

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


def _warmup_with_inference():
    """Run a full inference pass on a synthetic PDF to pre-compile CUDA kernels.

    The first time GPU models encounter new tensor shapes, CUDA JIT-compiles
    optimized kernels (~2s per shape). This warm-up forces compilation for
    common shapes before real requests arrive, eliminating the cold-start
    penalty (observed as ~70s vs ~0.7s for OCR-det in production).
    """
    print("Running warm-up inference to pre-compile CUDA kernels...")
    start = time.time()
    pdf_bytes = _create_warmup_pdf(num_pages=4)
    try:
        _do_convert(
            pdf_bytes, lang="en", parse_method="ocr",
            formula_enable=True, table_enable=True,
            max_pages=None, start_time=time.time(),
        )
        elapsed = round(time.time() - start, 1)
        print(f"Warm-up inference complete in {elapsed}s")
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"Warm-up inference finished in {elapsed}s (non-critical error: {e})")


def _do_convert(pdf_bytes, lang, parse_method, formula_enable, table_enable, max_pages, start_time):
    """Core conversion logic — separated so convert_to_markdown can retry with repaired bytes."""
    # Optionally limit to first N pages
    if max_pages is not None:
        try:
            max_pages_int = int(max_pages)
        except Exception:
            raise Exception("Invalid max_pages value; must be an integer")
        pdf_bytes = _trim_pdf_to_max_pages(pdf_bytes, max_pages_int)

    # Analyze the PDF
    infer_results, all_image_lists, all_pdf_docs, lang_list_result, ocr_enabled_list = pipeline_doc_analyze(
        [pdf_bytes],
        [lang],
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable
    )

    # Process results
    model_list = infer_results[0]
    images_list = all_image_lists[0]
    pdf_doc = all_pdf_docs[0]
    page_count = len(pdf_doc)
    _lang = lang_list_result[0]
    _ocr_enable = ocr_enabled_list[0]

    # Create temporary image directory for any image processing
    with tempfile.TemporaryDirectory() as temp_dir:
        image_writer = FileBasedDataWriter(temp_dir)

        # Convert to middle JSON format
        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_doc, image_writer,
            _lang, _ocr_enable, formula_enable
        )

        # Generate markdown
        pdf_info = middle_json["pdf_info"]
        md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, "images")

        processing_time_ms = round((time.time() - start_time) * 1000)
        metadata = {
            "pages": page_count,
            "ocr": _ocr_enable,
            "processing_time_ms": processing_time_ms,
        }
        return md_content, metadata

def convert_to_markdown(pdf_bytes, lang="en", parse_method="auto", formula_enable=True, table_enable=True, max_pages=None):
    """Convert PDF bytes to markdown - returns the markdown string and processing metadata"""

    start_time = time.time()
    try:
        return _do_convert(pdf_bytes, lang, parse_method, formula_enable, table_enable, max_pages, start_time)
    except PdfiumError as first_error:
        # PDFium can't parse the PDF (e.g. broken xref table) — repair and retry
        repaired = _repair_pdf(pdf_bytes)
        if repaired is pdf_bytes:
            raise Exception(f"Error converting PDF to markdown: {first_error}")
        try:
            return _do_convert(repaired, lang, parse_method, formula_enable, table_enable, max_pages, start_time)
        except Exception:
            raise Exception(f"Error converting PDF to markdown: {first_error}")
    except Exception as e:
        raise Exception(f"Error converting PDF to markdown: {e}")

async def async_convert_to_markdown(pdf_bytes, timeout_seconds=None, **kwargs):
    """Async wrapper with timeout support"""
    loop = asyncio.get_running_loop()
    
    if timeout_seconds and timeout_seconds > 0:
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: convert_to_markdown(pdf_bytes, **kwargs)),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"PDF processing timed out after {timeout_seconds} seconds")
    else:
        return await loop.run_in_executor(None, lambda: convert_to_markdown(pdf_bytes, **kwargs))


async def handler(event):
    """Main serverless handler - returns only markdown"""
    try:
        input_data = event.get("input", {})
        base64_content = input_data.get("file_content")
        filename = input_data.get("filename")
        timeout = input_data.get("timeout")
        created_at = input_data.get("created_at")
        max_pages = input_data.get("max_pages")
        
        # Processing options
        lang = input_data.get("lang", "en")
        parse_method = input_data.get("parse_method", "auto")
        formula_enable = input_data.get("formula_enable", True)
        table_enable = input_data.get("table_enable", True)

        # Calculate remaining timeout
        timeout_seconds = None
        if timeout:
            timeout_seconds = int(timeout) / 1000
            if created_at:
                elapsed = time.time() - (created_at / 1000)
                if elapsed >= timeout_seconds:
                    return {"error": "Request timed out before processing", "status": "TIMEOUT"}
                timeout_seconds = max(0, timeout_seconds - elapsed)
                if timeout_seconds < 1:
                    return {"error": "Insufficient time remaining", "status": "TIMEOUT"}

        # Validate input
        if not base64_content or not filename:
            return {"error": "Missing file_content or filename", "status": "ERROR"}

        if not filename.lower().endswith('.pdf'):
            return {"error": "Only PDF files supported", "status": "ERROR"}

        # Validate max_pages if provided
        if max_pages is not None:
            try:
                max_pages = int(max_pages)
                if max_pages <= 0:
                    return {"error": "max_pages must be a positive integer", "status": "ERROR"}
            except Exception:
                return {"error": "Invalid max_pages; must be an integer", "status": "ERROR"}

        # Process PDF
        pdf_bytes = base64.b64decode(base64_content)
        
        md_content, metadata = await async_convert_to_markdown(
            pdf_bytes=pdf_bytes,
            timeout_seconds=timeout_seconds,
            lang=lang,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            max_pages=max_pages
        )

        return {"markdown": md_content, "status": "SUCCESS", **metadata}
        
    except TimeoutError as e:
        return {"error": str(e), "status": "TIMEOUT"}
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}

if __name__ == "__main__":
    # Warm up models and pre-compile CUDA kernels (both debug and production)
    print("Warming up pipeline models...")
    ModelSingleton().get_model(
        lang="en",
        formula_enable=True,
        table_enable=True
    )
    print("Pipeline models warmed up")
    _warmup_with_inference()

    if os.environ.get("DEBUG_SERVER", "false").lower() == "true":
        import uvicorn
        from fastapi import FastAPI, Request

        app = FastAPI()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.post("/run")
        async def debug_endpoint(request: Request):
            input_data = await request.json()
            # Simulate RunPod event structure
            event = {"input": input_data}
            return await handler(event)

        print("Starting Debug Server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})