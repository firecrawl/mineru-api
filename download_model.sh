#!/bin/bash
set -e

# Define log file location
LOG_FILE="/var/log/model_download.log"

echo "Starting model downloads at $(date)" | tee -a "$LOG_FILE"

# Function to download and extract models
download_and_extract() {
    local url=$1
    local destination_path=$2
    local output_tar="$destination_path.tar"
    local model_name=$3

    echo "Downloading ${model_name} from ${url}" | tee -a "$LOG_FILE"
    curl -L -o "${output_tar}" "${url}" 2>&1 | tee -a "$LOG_FILE"

    echo "Extracting ${model_name}" | tee -a "$LOG_FILE"
    mkdir -p "$(dirname "${destination_path}")"
    tar -xvf "${output_tar}" -C "$(dirname "${destination_path}")" 2>&1 | tee -a "$LOG_FILE"

    echo "Removing archive for ${model_name}" | tee -a "$LOG_FILE"
    rm "${output_tar}" | tee -a "$LOG_FILE"

    echo "${model_name} download and extraction completed." | tee -a "$LOG_FILE"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
}

# Download PP Detect Model to /root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/
download_and_extract "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar" \
                    "/root/.paddleocr/whl/cls/ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.tar" \
                    "PP Detect Model"

# Download PP Rec Model to /root/.paddleocr/whl/rec/ch_ppocr_mobile_v2.0_rec_infer/
download_and_extract "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar" \
                    "/root/.paddleocr/whl/rec/ch_ppocr_mobile_v2.0_rec_infer/ch_ppocr_mobile_v2.0_rec_infer.tar" \
                    "PP Rec Model"
                    
# **New Addition**: Download PP Rec Model to /root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/
download_and_extract "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar" \
                    "/root/.paddleocr/whl/rec/ch/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer.tar" \
                    "PP Rec Model Additional Path"

echo "All model downloads completed at $(date)" | tee -a "$LOG_FILE"