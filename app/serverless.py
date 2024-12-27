import base64
import os
from pathlib import Path
from uuid import uuid4
import shutil

import magic_pdf.model as model_config
import runpod
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

from .office_converter import OfficeConverter, OfficeExts

# Configure model settings
model_config.__use_inside_model__ = True
model_config.__model_mode__ = "full"

_tmp_dir = "/tmp/{uuid}"

def convert_to_markdown(pdf_bytes, tmp_dir, filename):
    """Convert file to markdown and handle office document conversion if needed"""
    # Set up temporary directories
    local_image_dir = f"{tmp_dir}/images"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(local_image_dir, exist_ok=True)

    try:
        # Handle office documents conversion
        if filename.endswith(OfficeExts.__args__):
            input_file: Path = Path(tmp_dir) / filename
            input_file.write_bytes(pdf_bytes)
            output_file: Path = Path(tmp_dir) / f"{Path(filename).stem}.pdf"
            office_converter = OfficeConverter()
            office_converter.convert(input_file, output_file)
            pdf_bytes = output_file.read_bytes()
        elif not filename.endswith(".pdf"):
            raise ValueError("Unsupported file type")

        # Process PDF
        image_writer = DiskReaderWriter(local_image_dir)
        jso_useful_key = {"_pdf_type": "", "model_list": []}
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer, is_debug=True)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
        return pipe.pipe_mk_markdown(local_image_dir, drop_mode="none")
    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

def setup():
    # Create a sample directory for initialization
    sample_dir = "../pdfs"
    filename = "small_ocr2.pdf"
    sample_pdf_path = os.path.join(sample_dir, filename)
    
    with open(sample_pdf_path, "rb") as f:
        sample_pdf_bytes = f.read()
    
    # Warm up the conversion process
    convert_to_markdown(sample_pdf_bytes, sample_dir, filename)

def handler(event):
    try:
        # Extract base64 encoded file and filename from the event
        input_data = event.get("input", {})
        base64_content = input_data.get("file_content")
        filename = input_data.get("filename")

        if not base64_content or not filename:
            return {"error": "Missing file_content or filename"}

        # Decode base64 content
        pdf_bytes = base64.b64decode(base64_content)
        
        # Create unique temporary directory
        uuid_str = str(uuid4())
        tmp_dir = _tmp_dir.format(uuid=uuid_str)

        # Convert file to markdown
        md_content = convert_to_markdown(pdf_bytes, tmp_dir, filename)
        return {"markdown": md_content}

    except Exception as e:
        return {"error": str(e)}

# Call setup to initiate and warm up resources
# setup()

runpod.serverless.start({"handler": handler}) 