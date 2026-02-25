import base64
import os
import time
import asyncio
import concurrent.futures
import tempfile
import copy
import io
import threading

import torch
torch.backends.cudnn.benchmark = False

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

from app.warmup import create_warmup_pdf, warmup_ocr_det_shapes

# --- Configuration -----------------------------------------------------------
NUM_GPU_WORKERS = int(os.environ.get("NUM_GPU_WORKERS", "3"))
MIN_CHUNK_PAGES = int(os.environ.get("MIN_CHUNK_PAGES", "3"))

# Thread pool for all GPU work.  cuDNN algorithm caches are per-thread
# (per cuDNN handle), so warmup and real inference MUST run in the same
# thread for compiled kernels to be reused.
_gpu_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=NUM_GPU_WORKERS, thread_name_prefix="gpu"
)

class TimeoutError(Exception):
    pass

# --- PDF utilities -----------------------------------------------------------

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

# --- Chunk splitting ---------------------------------------------------------

def _get_chunk_count(total_pages: int) -> int:
    """Determine how many chunks to split a PDF into.

    Returns 1 (no chunking) when the document is too small to benefit.
    """
    if total_pages < NUM_GPU_WORKERS * MIN_CHUNK_PAGES:
        return 1
    return min(NUM_GPU_WORKERS, total_pages // MIN_CHUNK_PAGES)


def _split_pdf_into_chunks(pdf_bytes: bytes, num_chunks: int) -> list:
    """Split *pdf_bytes* into *num_chunks* smaller PDFs (as bytes).

    Pages are distributed as evenly as possible; larger chunks come first
    when there is a remainder.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)

    base_size = total_pages // num_chunks
    remainder = total_pages % num_chunks

    chunks = []
    page_idx = 0
    for i in range(num_chunks):
        chunk_size = base_size + (1 if i < remainder else 0)
        writer = PdfWriter()
        for _ in range(chunk_size):
            writer.add_page(reader.pages[page_idx])
            page_idx += 1
        buf = io.BytesIO()
        writer.write(buf)
        chunks.append(buf.getvalue())

    return chunks

# --- Core conversion ---------------------------------------------------------

def _warmup_with_inference():
    """Run a full inference pass on a synthetic PDF to pre-compile CUDA kernels.

    The first time GPU models encounter new tensor shapes, CUDA JIT-compiles
    optimized kernels (~2s per shape). This warm-up forces compilation for
    common shapes before real requests arrive, eliminating the cold-start
    penalty (observed as ~70s vs ~0.7s for OCR-det in production).
    """
    thread = threading.current_thread().name
    print(f"[{thread}] Running warm-up inference to pre-compile CUDA kernels...")
    start = time.time()
    pdf_bytes = create_warmup_pdf(num_pages=8)
    try:
        _do_convert(
            pdf_bytes, lang="en", parse_method="ocr",
            formula_enable=True, table_enable=True,
            max_pages=None, start_time=time.time(),
        )
        elapsed = round(time.time() - start, 1)
        print(f"[{thread}] Warm-up inference complete in {elapsed}s")
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"[{thread}] Warm-up inference finished in {elapsed}s (non-critical error: {e})")


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


def _convert_single(pdf_bytes, start_time, lang="en", parse_method="auto",
                    formula_enable=True, table_enable=True, max_pages=None):
    """Convert a single PDF with PdfiumError retry logic."""
    try:
        return _do_convert(pdf_bytes, lang, parse_method, formula_enable,
                           table_enable, max_pages, start_time)
    except PdfiumError as first_error:
        repaired = _repair_pdf(pdf_bytes)
        if repaired is pdf_bytes:
            raise Exception(f"Error converting PDF to markdown: {first_error}")
        try:
            return _do_convert(repaired, lang, parse_method, formula_enable,
                               table_enable, max_pages, start_time)
        except Exception:
            raise Exception(f"Error converting PDF to markdown: {first_error}")
    except Exception as e:
        raise Exception(f"Error converting PDF to markdown: {e}")


def convert_to_markdown(pdf_bytes, lang="en", parse_method="auto",
                        formula_enable=True, table_enable=True, max_pages=None):
    """Convert PDF bytes to markdown - returns the markdown string and processing metadata"""
    return _convert_single(pdf_bytes, time.time(), lang, parse_method,
                           formula_enable, table_enable, max_pages)

# --- Chunk conversion wrappers -----------------------------------------------

def _do_convert_chunk(chunk_bytes, chunk_index, num_chunks, start_time,
                      lang, parse_method, formula_enable, table_enable):
    """Run _do_convert for a single chunk with per-chunk logging."""
    thread = threading.current_thread().name
    chunk_start = time.time()
    print(f"[{thread}] Chunk {chunk_index + 1}/{num_chunks} starting")
    result = _do_convert(chunk_bytes, lang, parse_method, formula_enable,
                         table_enable, None, start_time)
    elapsed = round(time.time() - chunk_start, 1)
    pages = result[1]["pages"]
    print(f"[{thread}] Chunk {chunk_index + 1}/{num_chunks} done in {elapsed}s ({pages} pages)")
    return result


def _convert_chunk_with_retry(chunk_bytes, chunk_index, num_chunks, start_time,
                              lang, parse_method, formula_enable, table_enable):
    """Convert a chunk with PdfiumError retry."""
    thread = threading.current_thread().name
    try:
        return _do_convert_chunk(chunk_bytes, chunk_index, num_chunks,
                                 start_time, lang, parse_method,
                                 formula_enable, table_enable)
    except PdfiumError as first_error:
        print(f"[{thread}] Chunk {chunk_index + 1}/{num_chunks} PdfiumError, retrying with repair")
        repaired = _repair_pdf(chunk_bytes)
        if repaired is chunk_bytes:
            raise Exception(f"Chunk {chunk_index + 1} error: {first_error}")
        try:
            return _do_convert_chunk(repaired, chunk_index, num_chunks,
                                     start_time, lang, parse_method,
                                     formula_enable, table_enable)
        except Exception:
            raise Exception(f"Chunk {chunk_index + 1} error: {first_error}")
    except Exception as e:
        raise Exception(f"Chunk {chunk_index + 1} error: {e}")

# --- Async orchestration -----------------------------------------------------

async def async_convert_to_markdown(pdf_bytes, timeout_seconds=None, **kwargs):
    """Split a PDF into chunks and process them concurrently on the GPU thread pool.

    Orchestration (page counting, splitting, merging) runs on the asyncio
    thread so that all pool threads are available for GPU work.
    """
    loop = asyncio.get_running_loop()
    start_time = time.time()

    lang = kwargs.get("lang", "en")
    parse_method = kwargs.get("parse_method", "auto")
    formula_enable = kwargs.get("formula_enable", True)
    table_enable = kwargs.get("table_enable", True)
    max_pages = kwargs.get("max_pages")

    # Apply max_pages trimming upfront (CPU-only, fine on asyncio thread)
    if max_pages is not None:
        try:
            max_pages_int = int(max_pages)
        except Exception:
            raise Exception("Invalid max_pages value; must be an integer")
        pdf_bytes = _trim_pdf_to_max_pages(pdf_bytes, max_pages_int)

    # Count pages and determine chunking strategy
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
    except Exception:
        total_pages = 0

    num_chunks = _get_chunk_count(total_pages)

    async def _run():
        nonlocal pdf_bytes

        if num_chunks <= 1:
            # Single chunk — use existing conversion path
            return await loop.run_in_executor(
                _gpu_executor,
                lambda: _convert_single(
                    pdf_bytes, start_time, lang, parse_method,
                    formula_enable, table_enable, max_pages=None,
                ),
            )

        # Split into chunks (CPU-only, on asyncio thread)
        try:
            chunks = _split_pdf_into_chunks(pdf_bytes, num_chunks)
        except Exception as e:
            print(f"Chunk splitting failed ({e}), falling back to single-chunk")
            return await loop.run_in_executor(
                _gpu_executor,
                lambda: _convert_single(
                    pdf_bytes, start_time, lang, parse_method,
                    formula_enable, table_enable, max_pages=None,
                ),
            )

        print(f"Processing {total_pages} pages in {len(chunks)} chunks")

        # Submit chunks with a stagger to avoid concurrent pypdfium2
        # document loading (not thread-safe).  Once past the loading
        # phase, inference runs in parallel via GIL-releasing CUDA ops.
        futures = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                await asyncio.sleep(1.0)
            fut = loop.run_in_executor(
                _gpu_executor,
                lambda idx=i, c=chunk: _convert_chunk_with_retry(
                    c, idx, len(chunks), start_time,
                    lang=lang, parse_method=parse_method,
                    formula_enable=formula_enable, table_enable=table_enable,
                ),
            )
            futures.append(fut)

        results = await asyncio.gather(*futures)

        # Merge markdown in page order
        md_content = "\n\n".join(r[0] for r in results)
        total_processing_time_ms = round((time.time() - start_time) * 1000)
        total_page_count = sum(r[1]["pages"] for r in results)
        metadata = {
            "pages": total_page_count,
            "ocr": results[0][1]["ocr"],
            "processing_time_ms": total_processing_time_ms,
            "chunks": len(chunks),
        }
        return md_content, metadata

    if timeout_seconds and timeout_seconds > 0:
        try:
            return await asyncio.wait_for(_run(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TimeoutError(f"PDF processing timed out after {timeout_seconds} seconds")
    else:
        return await _run()

# --- Request handler ---------------------------------------------------------

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

# --- Warmup ------------------------------------------------------------------

def _full_warmup():
    """Primary thread warmup: model loading + inference + OCR-det shapes.

    Executed inside _gpu_executor so that cuDNN algorithm caches live in the
    same thread that will later handle real inference requests.
    """
    thread = threading.current_thread().name
    print(f"[{thread}] Warming up pipeline models...")
    ModelSingleton().get_model(
        lang="en",
        formula_enable=True,
        table_enable=True
    )
    print(f"[{thread}] Pipeline models warmed up")
    _warmup_with_inference()
    warmup_ocr_det_shapes(lang="en")
    print(f"[{thread}] Primary warmup complete")


def _secondary_warmup():
    """Secondary thread warmup: inference + OCR-det shapes.

    Models are already loaded (singletons).  This populates the per-thread
    cuDNN algorithm caches.  Should be fast thanks to the process-level
    CUDA kernel cache populated by the primary thread.
    """
    thread = threading.current_thread().name
    start = time.time()
    print(f"[{thread}] Secondary warmup starting...")
    _warmup_with_inference()
    warmup_ocr_det_shapes(lang="en")
    elapsed = round(time.time() - start, 1)
    print(f"[{thread}] Secondary warmup complete in {elapsed}s")


if __name__ == "__main__":
    print(f"GPU thread pool: {NUM_GPU_WORKERS} workers, min {MIN_CHUNK_PAGES} pages/chunk")

    # Primary warmup on first thread (model loading + inference + OCR-det)
    _gpu_executor.submit(_full_warmup).result()

    # Secondary warmup on remaining threads (inference + OCR-det only)
    if NUM_GPU_WORKERS > 1:
        print(f"Warming up {NUM_GPU_WORKERS - 1} secondary GPU threads...")
        futs = [_gpu_executor.submit(_secondary_warmup)
                for _ in range(NUM_GPU_WORKERS - 1)]
        for f in concurrent.futures.as_completed(futs):
            f.result()  # propagate exceptions
        print("All GPU threads warmed up")

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
