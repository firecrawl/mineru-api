#!/usr/bin/env bash
# Build a patched libpdfium.so for linux/amd64.
# Output: patch/libpdfium.so
#
# Usage:
#   cd patch && ./build-pdfium.sh
#
# Requires Docker with BuildKit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

docker build \
    --platform linux/amd64 \
    -f "$SCRIPT_DIR/build-pdfium.Dockerfile" \
    -o "$SCRIPT_DIR" \
    "$SCRIPT_DIR"

echo "Built: $SCRIPT_DIR/libpdfium.so"
