import base64
import os
from pathlib import Path
from uuid import uuid4

import magic_pdf.model as model_config
import runpod
from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter

from app.office_converter import OfficeConverter, OfficeExts

# Configure model settings
model_config.__use_inside_model__ = True
model_config.__model_mode__ = "full"

_tmp_dir = "/tmp/{uuid}"
_local_image_dir = "/tmp/{uuid}/images"

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
        
        # Set up temporary directories
        uuid_str = str(uuid4())
        tmp_dir = _tmp_dir.format(uuid=uuid_str)
        local_image_dir = _local_image_dir.format(uuid=uuid_str)
        os.makedirs(tmp_dir, exist_ok=True)
        os.makedirs(local_image_dir, exist_ok=True)

        # Handle office documents conversion
        if filename.endswith(OfficeExts.__args__):
            input_file: Path = Path(tmp_dir) / filename
            input_file.write_bytes(pdf_bytes)
            output_file: Path = Path(tmp_dir) / f"{Path(filename).stem}.pdf"
            office_converter = OfficeConverter()
            office_converter.convert(input_file, output_file)
            pdf_bytes = output_file.read_bytes()
        elif not filename.endswith(".pdf"):
            return {"error": "Unsupported file type"}

        # Process PDF
        image_writer = DiskReaderWriter(local_image_dir)
        jso_useful_key = {"_pdf_type": "", "model_list": []}
        pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer, is_debug=True)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()
        md_content = pipe.pipe_mk_markdown(local_image_dir, drop_mode="none")

        return {"markdown": md_content}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler}) 