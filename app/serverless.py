import base64
import os
import time
import tempfile
import copy
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

import runpod

# New mineru imports
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make

class TimeoutError(Exception):
    pass

def convert_to_markdown(pdf_bytes, lang="en", parse_method="auto", formula_enable=True, table_enable=True):
    """Convert PDF bytes to markdown - returns only the markdown string"""
    
    try:
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

def convert_to_markdown_vlm(pdf_bytes, backend="vlm-sglang-engine", server_url=None):
    """Convert PDF bytes to markdown using VLM backends; returns markdown string.
    Only server/engine backend is supported as requested.
    """
    # Normalize backend to what vlm_doc_analyze expects
    normalized_backend = backend[4:] if backend.startswith("vlm-") else backend
    with tempfile.TemporaryDirectory() as temp_dir:
        image_writer = FileBasedDataWriter(temp_dir)
        middle_json, _ = vlm_doc_analyze(
            pdf_bytes,
            image_writer=image_writer,
            backend=normalized_backend,
            server_url=server_url,
        )
        pdf_info = middle_json["pdf_info"]
        return vlm_union_make(pdf_info, MakeMode.MM_MD, "images")


def convert_to_markdown_dispatch(pdf_bytes, **kwargs):
    """Dispatch to pipeline or VLM engine based on env MINERU_BACKEND.
    Defaults to pipeline without changing existing behavior.
    """
    backend_env = os.getenv("MINERU_BACKEND", "pipeline").lower()
    if backend_env == "vlm-sglang-engine":
        # Only support server/engine backend as requested; no client here
        server_url = os.getenv("MINERU_SGLANG_SERVER_URL")  # optional, generally not needed for engine
        return convert_to_markdown_vlm(pdf_bytes, backend=backend_env, server_url=server_url)
    # Fallback to existing pipeline behavior
    return convert_to_markdown(pdf_bytes, **kwargs)


def _convert_entry(args_tuple):
    """Top-level helper for subprocess execution."""
    (
        pdf_bytes,
        backend_env,
        server_url,
        lang,
        parse_method,
        formula_enable,
        table_enable,
    ) = args_tuple
    if backend_env:
        os.environ["MINERU_BACKEND"] = backend_env
    if server_url:
        os.environ["MINERU_SGLANG_SERVER_URL"] = server_url
    return convert_to_markdown_dispatch(
        pdf_bytes,
        lang=lang,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
    )


def convert_to_markdown_with_timeout(
    pdf_bytes,
    timeout_seconds=None,
    *,
    backend_env: str | None,
    server_url: str | None,
    lang: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
):
    """Run conversion in a separate process with an optional timeout.
    Keeps conversion in the main thread when no timeout is requested.
    """
    # If no timeout, run inline in the main process/thread to allow libraries that require main-thread signals.
    if not timeout_seconds or timeout_seconds <= 0:
        # Ensure env is applied for inline run as well
        if backend_env:
            os.environ["MINERU_BACKEND"] = backend_env
        if server_url:
            os.environ["MINERU_SGLANG_SERVER_URL"] = server_url
        return convert_to_markdown_dispatch(
            pdf_bytes,
            lang=lang,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
        )

    args_tuple = (
        pdf_bytes,
        backend_env,
        server_url,
        lang,
        parse_method,
        formula_enable,
        table_enable,
    )
    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_convert_entry, args_tuple)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            raise TimeoutError(f"PDF processing timed out after {timeout_seconds} seconds")


def handler(event):
    """Main serverless handler - returns only markdown (synchronous)."""
    try:
        input_data = event.get("input", {})
        base64_content = input_data.get("file_content")
        filename = input_data.get("filename")
        timeout = input_data.get("timeout")
        created_at = input_data.get("created_at")
        
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

        # Process PDF
        pdf_bytes = base64.b64decode(base64_content)

        # Read backend envs once and pass into subprocess when needed
        backend_env = os.getenv("MINERU_BACKEND", "pipeline").lower()
        server_url = os.getenv("MINERU_SGLANG_SERVER_URL")

        md_content = convert_to_markdown_with_timeout(
            pdf_bytes=pdf_bytes,
            timeout_seconds=timeout_seconds,
            backend_env=backend_env,
            server_url=server_url,
            lang=lang,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
        )

        return {"markdown": md_content, "status": "SUCCESS"}
        
    except TimeoutError as e:
        return {"error": str(e), "status": "TIMEOUT"}
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}

if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})