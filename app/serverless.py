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

from app.warmup import create_warmup_pdf, warmup_ocr_det_shapes

# Thread pool for GPU work.  cuDNN algorithm caches are per-thread
# (per cuDNN handle), so warmup and real inference MUST run in the same
# thread for compiled kernels to be reused.
_gpu_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="gpu"
)

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

def convert_to_markdown(pdf_bytes, lang="en", parse_method="auto", formula_enable=True, table_enable=True, max_pages=None):
    """Convert PDF bytes to markdown - returns only the markdown string"""
    
    try:
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
            
            # Generate and return markdown
            pdf_info = middle_json["pdf_info"]
            return pipeline_union_make(pdf_info, MakeMode.MM_MD, "images")
            
    except Exception as e:
        raise Exception(f"Error converting PDF to markdown: {str(e)}")

async def async_convert_to_markdown(pdf_bytes, timeout_seconds=None, **kwargs):
    """Async wrapper with timeout support.

    Runs inference on _gpu_executor so it shares the same thread (and
    cuDNN algorithm caches) that was warmed up at startup.
    """
    loop = asyncio.get_running_loop()

    if timeout_seconds and timeout_seconds > 0:
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_gpu_executor, lambda: convert_to_markdown(pdf_bytes, **kwargs)),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"PDF processing timed out after {timeout_seconds} seconds")
    else:
        return await loop.run_in_executor(_gpu_executor, lambda: convert_to_markdown(pdf_bytes, **kwargs))

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
        
        md_content = await async_convert_to_markdown(
            pdf_bytes=pdf_bytes,
            timeout_seconds=timeout_seconds,
            lang=lang,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            max_pages=max_pages
        )

        return {"markdown": md_content, "status": "SUCCESS"}
        
    except TimeoutError as e:
        return {"error": str(e), "status": "TIMEOUT"}
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}

def _warmup_with_inference():
    """Run a full inference pass on a synthetic PDF to pre-compile CUDA kernels.

    The first time GPU models encounter new tensor shapes, CUDA JIT-compiles
    optimized kernels (~2s per shape). This warm-up forces compilation for
    common shapes before real requests arrive.
    """
    thread = threading.current_thread().name
    print(f"[{thread}] Running warm-up inference to pre-compile CUDA kernels...")
    start = time.time()
    pdf_bytes = create_warmup_pdf(num_pages=8)
    try:
        convert_to_markdown(
            pdf_bytes, lang="en", parse_method="ocr",
            formula_enable=True, table_enable=True,
            max_pages=None,
        )
        elapsed = round(time.time() - start, 1)
        print(f"[{thread}] Warm-up inference complete in {elapsed}s")
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        print(f"[{thread}] Warm-up inference finished in {elapsed}s (non-critical error: {e})")


def _full_warmup():
    """Load models + run inference + warm OCR-det shapes on the GPU thread.

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
    print(f"[{thread}] Warmup complete")


if __name__ == "__main__":
    # Run all warmup on the GPU thread so cuDNN caches are reused at inference time
    _gpu_executor.submit(_full_warmup).result()

    if os.environ.get("DEBUG_SERVER", "false").lower() == "true":
        import uvicorn
        from fastapi import FastAPI, Request

        app = FastAPI()

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
