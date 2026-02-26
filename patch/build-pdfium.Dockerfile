FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git curl python3 pkg-config lsb-release \
    build-essential libstdc++-13-dev \
    && rm -rf /var/lib/apt/lists/*

# Install depot_tools
RUN git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git /opt/depot_tools
ENV PATH="/opt/depot_tools:$PATH"

WORKDIR /build

# Fetch PDFium source
RUN gclient config --unmanaged https://pdfium.googlesource.com/pdfium.git && \
    gclient sync --no-history

# Apply patch
COPY pdfium-null-kids.patch /tmp/pdfium-null-kids.patch
RUN cd pdfium && patch -p1 < /tmp/pdfium-null-kids.patch

# Configure shared library build
RUN mkdir -p pdfium/out/Release && cat > pdfium/out/Release/args.gn << 'EOF'
is_debug = false
pdf_is_standalone = true
is_component_build = true
pdf_enable_v8 = false
pdf_enable_xfa = false
treat_warnings_as_errors = false
use_custom_libcxx = false
use_sysroot = false
use_cxx_modules = false
clang_use_chrome_plugins = false
use_glib = false
EOF

# Build
RUN cd pdfium && gn gen out/Release && ninja -C out/Release pdfium

# Output stage — extract just the shared library
FROM scratch
COPY --from=builder /build/pdfium/out/Release/libpdfium.so /libpdfium.so
