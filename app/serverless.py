import base64
import os
import time
import tempfile
import copy
import io
import asyncio

import runpod


from pypdf import PdfReader, PdfWriter

# # Ensure MinerU custom HF classes are available and optionally pre-initialize sglang engine
# try:
#     # Importing registers custom model types and/or enables custom code for AutoConfig
#     from mineru.backend.vlm.hf_predictor import HuggingfacePredictor  # noqa: F401
# except Exception:
#     pass
# try:
#     from mineru.backend.vlm.predictor import get_predictor  # noqa: F401
# except Exception:
#     pass

def _maybe_init_sglang_engine_in_main() -> None:
    """Initialize sglang engine in the main process if requested via env.

    Per MinerU guidance, sglang-engine must be initialized in the main process.
    This avoids scheduler failures when workers spawn without prior initialization.
    """
    backend_env = os.getenv("MINERU_BACKEND", "pipeline").lower()
    if backend_env == "vlm-sglang-engine":
        try:
            from mineru.backend.vlm.vlm_analyze import ModelSingleton
            # Initialize once; ModelSingleton handles idempotency
            ModelSingleton().get_model("sglang-engine", None, None)
        except Exception:
            # Defer detailed errors to runtime path to avoid import-time crashes
            pass

_maybe_init_sglang_engine_in_main()

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
        # Lazy imports to avoid import-time signal handling in non-main threads
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.enum_class import MakeMode
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

        # Optionally limit to first N pages
        if max_pages is not None:
            try:
                max_pages_int = int(max_pages)
            except Exception:
                raise Exception("Invalid max_pages value; must be an integer")
            pdf_bytes = _trim_pdf_to_max_pages(pdf_bytes, max_pages_int)

        # Analyze the PDF
        infer_results, all_image_lists, all_pdf_docs, lang_list_result, ocr_enabled_list = pipeline_doc_analyze(
            [pdf_bytes], [lang], parse_method=parse_method, formula_enable=formula_enable, table_enable=table_enable
        )

        model_list = infer_results[0]
        images_list = all_image_lists[0]
        pdf_doc = all_pdf_docs[0]
        _lang = lang_list_result[0]
        _ocr_enable = ocr_enabled_list[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            image_writer = FileBasedDataWriter(temp_dir)
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable
            )
            pdf_info = middle_json["pdf_info"]
            return pipeline_union_make(pdf_info, MakeMode.MM_MD, "images")
    except Exception as e:
        raise Exception(f"Error converting PDF to markdown: {str(e)}")


def convert_to_markdown_vlm(pdf_bytes, backend="vlm-sglang-engine", server_url=None):
    """Convert PDF bytes to markdown using VLM backends; returns markdown string."""
    # Lazy imports to avoid import-time signal handling in non-main threads
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.enum_class import MakeMode
    from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make

    normalized_backend = backend[4:] if backend.startswith("vlm-") else backend
    with tempfile.TemporaryDirectory() as temp_dir:
        image_writer = FileBasedDataWriter(temp_dir)
        middle_json, _ = vlm_doc_analyze(
            pdf_bytes, image_writer=image_writer, backend=normalized_backend, server_url=server_url
        )
        pdf_info = middle_json["pdf_info"]
        return vlm_union_make(pdf_info, MakeMode.MM_MD, "images")


def _convert_to_markdown_via_aio(
    pdf_bytes: bytes,
    filename: str,
    *,
    lang: str = "en",
    backend: str = "pipeline",
    parse_method: str = "auto",
    formula_enable: bool = True,
    table_enable: bool = True,
    server_url: str | None = None,
    max_pages: int | None = None,
) -> str:
    """Use MinerU's aio_do_parse to produce markdown and return its content."""
    # Lazy import to keep module import light
    from mineru.cli.common import aio_do_parse

    # Map max_pages to end_page_id semantics (inclusive end index)
    start_page_id = 0
    end_page_id = None
    if max_pages is not None:
        try:
            max_pages_int = int(max_pages)
            if max_pages_int > 0:
                end_page_id = max_pages_int - 1
        except Exception:
            raise Exception("Invalid max_pages value; must be an integer")

    with tempfile.TemporaryDirectory() as output_dir:
        # Run async parse
        async def _run():
            await aio_do_parse(
                output_dir=output_dir,
                pdf_file_names=[filename],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=[lang],
                backend=backend,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                server_url=server_url,
                f_draw_layout_bbox=False,
                f_draw_span_bbox=False,
                f_dump_md=True,
                f_dump_middle_json=False,
                f_dump_model_output=False,
                f_dump_orig_pdf=False,
                f_dump_content_list=False,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )

        asyncio.run(_run())

        # Locate markdown file
        parse_subdir = parse_method if backend.startswith("pipeline") else "vlm"
        parse_dir = os.path.join(output_dir, filename, parse_subdir)
        md_path = os.path.join(parse_dir, f"{filename}.md")
        if not os.path.exists(md_path):
            raise Exception("Markdown output not found after parsing")
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()


def convert_to_markdown_dispatch(pdf_bytes, filename=None, **kwargs):
    """Dispatch to pipeline or VLM engine based on env MINERU_BACKEND.

    Prefer using aio_do_parse to match official MinerU entrypoints.
    """
    backend_env = os.getenv("MINERU_BACKEND", "pipeline").lower()
    server_url = os.getenv("MINERU_SGLANG_SERVER_URL")
    lang = kwargs.get("lang", "en")
    parse_method = kwargs.get("parse_method", "auto")
    formula_enable = kwargs.get("formula_enable", True)
    table_enable = kwargs.get("table_enable", True)
    max_pages = kwargs.get("max_pages")

    if filename is None:
        filename = "document"

    # Use aio_do_parse path for both pipeline and vlm backends
    if backend_env.startswith("vlm"):
        parse_method = "vlm"
    backend_for_aio = backend_env
    return _convert_to_markdown_via_aio(
        pdf_bytes,
        filename,
        lang=lang,
        backend=backend_for_aio,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        server_url=server_url,
        max_pages=max_pages,
    )



def handler(event):
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

        md_content = convert_to_markdown_dispatch(
            pdf_bytes=pdf_bytes,
            filename=os.path.splitext(os.path.basename(filename))[0] if filename else "document",
            lang=lang,
            parse_method=parse_method,
            formula_enable=formula_enable,
            table_enable=table_enable,
            max_pages=max_pages
        )

        return {"markdown": md_content, "status": "SUCCESS"}
    except Exception as e:
        return {"error": str(e), "status": "ERROR"}


if __name__ == "__main__":
    print("Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})