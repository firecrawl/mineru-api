import base64
import os
import time
from pathlib import Path
from uuid import uuid4
import shutil
import io
import asyncio  # Add asyncio for async/await and timeout

import magic_pdf.model as model_config
import runpod
from pypdf import PdfReader

# New magic-pdf imports based on example
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

from .office_converter import OfficeConverter, OfficeExts

# Configure model settings (Keep for now, might be handled differently by new API)
model_config.__use_inside_model__ = True
model_config.__model_mode__ = "full"

_tmp_dir = "/tmp/{uuid}"

# Define custom TimeoutError for timeout handling
class TimeoutError(Exception):
    """Custom timeout exception."""
    pass

def convert_to_markdown(pdf_bytes, tmp_dir, filename, max_pages):
    """Convert file to markdown and handle office document conversion if needed"""
    # Set up temporary directories
    local_image_dir = Path(tmp_dir) / "images"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(local_image_dir, exist_ok=True)

    num_pages = 0 # Default page count
    pdf_bytes_for_processing = None

    try:
        # Handle office documents conversion
        if filename.endswith(OfficeExts.__args__):
            input_file: Path = Path(tmp_dir) / filename
            input_file.write_bytes(pdf_bytes)
            output_file: Path = Path(tmp_dir) / f"{Path(filename).stem}.pdf"
            office_converter = OfficeConverter()
            office_converter.convert(input_file, output_file)
            pdf_bytes_for_processing = output_file.read_bytes()
        elif filename.endswith(".pdf"):
            pdf_bytes_for_processing = pdf_bytes
        else:
            raise ValueError("Unsupported file type")

        # Get page count using pypdf
        try:
            pdf_file_like_object = io.BytesIO(pdf_bytes_for_processing)
            reader = PdfReader(pdf_file_like_object)
            num_pages = len(reader.pages)
        except Exception as e:
            # Log or handle pypdf errors if necessary
            print(f"Could not get page count using pypdf: {e}")
            pass # Continue even if page count fails

        # --- Start New magic-pdf API implementation ---
        if max_pages is not None and num_pages > max_pages:
            raise ValueError(f"File has {num_pages} pages, but max_pages is set to {max_pages}")
        # 1. Setup Writer
        # Use str() for FileBasedDataWriter path
        image_writer = FileBasedDataWriter(str(local_image_dir))
        relative_image_dir = str(local_image_dir.name) # Should be "images"

        # 2. Create Dataset
        ds = PymuDocDataset(pdf_bytes_for_processing)

        # 3. Classify and Apply model
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        # 4. Get Markdown
        md_content = pipe_result.get_markdown(relative_image_dir)

        # --- End New magic-pdf API implementation ---

        return md_content, num_pages
    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

def setup():
    # Create a sample directory for initialization
    # This function will now use the updated convert_to_markdown
    sample_dir = "../pdfs"
    filename = "small_ocr2.pdf"
    sample_pdf_path = os.path.join(sample_dir, filename)

    if os.path.exists(sample_pdf_path):
        with open(sample_pdf_path, "rb") as f:
            sample_pdf_bytes = f.read()
        # Warm up the conversion process
        print("Warming up convert_to_markdown...")
        try:
            # Use a dedicated temp dir for warmup
            warmup_tmp_dir = "/tmp/warmup_" + str(uuid4())
            convert_to_markdown(sample_pdf_bytes, warmup_tmp_dir, filename, None)  # Pass None for max_pages
            print("Warmup finished.")
        except Exception as e:
            print(f"Warmup failed: {e}")
        finally:
            if 'warmup_tmp_dir' in locals() and os.path.exists(warmup_tmp_dir):
                 shutil.rmtree(warmup_tmp_dir, ignore_errors=True)
    else:
        print(f"Warmup file not found: {sample_pdf_path}, skipping warmup.")

def init_model():
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton
    try:
        model_manager = ModelSingleton()
        txt_model = model_manager.get_model(False, False)  # noqa: F841
        print('txt_model init final') # Uses logger, we can use print
        ocr_model = model_manager.get_model(True, False)  # noqa: F841
        print('ocr_model init final') # Uses logger, we can use print
        return 0
    except Exception as e:
        print(e) # Uses logger, we can use print
        return -1

#lets not init the model here, cause it might not be needed and it runs on the first request anyways
#check an env variable to see if we should init the model
if os.getenv("INIT_MODEL") == "true":
    model_init = init_model() # Called globally at startup
    print(f'model_init: {model_init}') # Uses logger, we can use print
else:
    print("Model not initialized")
    
# Wrap convert_to_markdown in an async function for timeout control
async def async_convert_to_markdown(pdf_bytes, tmp_dir, filename, max_pages, timeout_seconds=None):
    """Async wrapper for convert_to_markdown that respects timeouts"""
    # Use a separate thread for CPU-bound work
    loop = asyncio.get_running_loop()
    
    if timeout_seconds is not None and timeout_seconds > 0:
        # Use asyncio.wait_for to enforce timeout
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: convert_to_markdown(pdf_bytes, tmp_dir, filename, max_pages)),
                timeout=timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            # Clean up temp directory if timed out
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise TimeoutError(f"PDF processing timed out after {timeout_seconds} seconds")
    else:
        # No timeout, just run normally
        return await loop.run_in_executor(None, lambda: convert_to_markdown(pdf_bytes, tmp_dir, filename, max_pages))

async def handler(event):
    try:
        print(f"Received event: {event}")
        # Extract base64 encoded file and filename from the event
        input_data = event.get("input", {})
        base64_content = input_data.get("file_content")
        filename = input_data.get("filename")
        max_pages = input_data.get("max_pages", None)

        timeout = event.get("timeout", None)
        created_at = event.get("created_at", None)
        # Set default timeout if needed
        if timeout is not None:
            # Convert JS timeout value (likely milliseconds) to seconds
            timeout_seconds = int(timeout) / 1000
            
            if created_at is not None:
                # Calculate elapsed time in seconds
                # JS timestamps are in milliseconds, convert to seconds
                created_at_seconds = created_at / 1000
                current_time = time.time()
                elapsed_seconds = current_time - created_at_seconds
                
                # Check if already timed out
                if elapsed_seconds >= timeout_seconds:
                    raise Exception("Request timed out before processing started")
                
                # Calculate remaining time for this operation
                remaining_seconds = max(0, timeout_seconds - elapsed_seconds)
                print(f"Request age: {elapsed_seconds:.2f}s, Timeout: {timeout_seconds:.2f}s, Remaining: {remaining_seconds:.2f}s")
                
                # Use the remaining time as our effective timeout
                timeout_seconds = remaining_seconds
            
            # If timeout is very small, return immediately
            if timeout_seconds < 1:
                return {"error": "Insufficient time remaining to process request", "status": "TIMEOUT"}
        else:
            timeout_seconds = None  # No timeout specified

        if not base64_content or not filename:
            return {"error": "Missing file_content or filename", "status": "ERROR"}

        # Decode base64 content
        pdf_bytes = base64.b64decode(base64_content)

        # Create unique temporary directory
        uuid_str = str(uuid4())
        tmp_dir = _tmp_dir.format(uuid=uuid_str)

        try:
            # Use the async wrapper with timeout control
            md_content, num_pages = await async_convert_to_markdown(
                pdf_bytes, tmp_dir, filename, max_pages, timeout_seconds
            )
            return {"markdown": md_content, "num_pages": num_pages, "status": "SUCCESS"}
        except TimeoutError as e:
            return {"error": str(e), "status": "TIMEOUT"}

    except Exception as e:
        # Consider more specific error handling/logging
        print(f"Error in handler: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "status": "ERROR"}




print("Starting RunPod serverless handler...")
# Runpod will now use the async handler
runpod.serverless.start({"handler": handler})
# This line might not be reached in normal serverless operation
print("RunPod serverless handler finished.") 