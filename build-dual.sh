#!/bin/bash
set -euo pipefail

# OpenBLAS.wasm Dual Build System
# Builds both SIDE_MODULE (production) and MAIN_MODULE (testing/NPM)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✅${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}❌${NC} $1"
}

# Configuration
BUILD_DIR_SIDE="build-side"
BUILD_DIR_MAIN="build-main"
INSTALL_DIR="install/wasm"
VARIANT="${1:-all}"

# Core BLAS sources - SIMD-optimized WASM implementation
BLAS_SOURCES="
src/wasm_module_simd.c
"

# Exported functions for both builds (malloc/free needed for MAIN_MODULE)
EXPORTED_FUNCTIONS='["_malloc","_free","_blas_has_simd","_blas_sdot","_blas_ddot","_blas_saxpy","_blas_daxpy","_blas_sscal","_blas_dscal","_blas_snrm2","_blas_dnrm2","_blas_sgemm","_blas_dgemm","_blas_get_version","_blas_get_num_threads","_blas_set_num_threads","_blas_get_performance_info","_blas_benchmark_sgemm"]'

# Common compilation flags
COMMON_FLAGS="
-O3 -flto
-msimd128
-DUSE_OPENMP=0
-DMAX_CPU_NUMBER=1
-DNO_FORTRAN=1
-DNO_LAPACK=1
-DEXPAT_SHARED=0
-I.
-Iinterface
-Ikernel/generic
-Idriver/others
-Icommon
-Wall -Wno-unused-function
"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v emcc &> /dev/null; then
        log_error "Emscripten not found. Please install and activate emsdk"
        exit 1
    fi

    log_info "Emscripten version: $(emcc --version | head -n1)"

    # Check for required source files
    missing_files=()
    for file in $BLAS_SOURCES; do
        if [[ ! -f "$file" ]]; then
            missing_files+=("$file")
        fi
    done

    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "Missing required source files:"
        printf '%s\n' "${missing_files[@]}"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Create directories
setup_directories() {
    log_info "Setting up build directories..."
    mkdir -p "$BUILD_DIR_SIDE" "$BUILD_DIR_MAIN" "$INSTALL_DIR"
    log_success "Build directories created"
}

# Build SIDE_MODULE for production use
build_side_module() {
    log_info "Building SIDE_MODULE (production)..."

    cd "$BUILD_DIR_SIDE"

    # SIDE_MODULE specific flags
    SIDE_FLAGS="
    -sSIDE_MODULE=2
    -fPIC
    -sSTANDALONE_WASM=1
    -sEXPORTED_FUNCTIONS=$EXPORTED_FUNCTIONS
    -sINITIAL_MEMORY=33554432
    -sALLOW_MEMORY_GROWTH=1
    -sDEFAULT_TO_CXX=0
    "

    log_info "Compiling SIDE_MODULE with SIMD optimizations..."

    # Create source list with full paths
    SOURCE_LIST=""
    for src in $BLAS_SOURCES; do
        SOURCE_LIST="$SOURCE_LIST ../$src"
    done

    emcc $COMMON_FLAGS $SIDE_FLAGS \
        $SOURCE_LIST \
        -o openblas-side.wasm

    # Verify the build
    if [[ ! -f "openblas-side.wasm" ]]; then
        log_error "SIDE_MODULE build failed"
        cd ..
        exit 1
    fi

    # Copy to install directory
    cp openblas-side.wasm "../$INSTALL_DIR/"

    cd ..
    log_success "SIDE_MODULE build completed ($(du -h $INSTALL_DIR/openblas-side.wasm | cut -f1))"
}

# Build MAIN_MODULE for testing and NPM
build_main_module() {
    log_info "Building MAIN_MODULE (testing/NPM)..."

    cd "$BUILD_DIR_MAIN"

    # MAIN_MODULE specific flags
    MAIN_FLAGS="
    -sMODULARIZE=1
    -sEXPORT_ES6=1
    -sEXPORT_NAME=OpenBLASModule
    -sSINGLE_FILE=0
    -sEXPORTED_FUNCTIONS=$EXPORTED_FUNCTIONS
    -sEXPORTED_RUNTIME_METHODS=[\"cwrap\",\"ccall\",\"UTF8ToString\",\"lengthBytesUTF8\"]
    -sALLOW_MEMORY_GROWTH=1
    -sINITIAL_MEMORY=33554432
    -sMAXIMUM_MEMORY=536870912
    -sENVIRONMENT=web,webview,worker
    -sNODEJS_CATCH_EXIT=0
    -sNODEJS_CATCH_REJECTION=0
    "

    log_info "Compiling MAIN_MODULE with SIMD optimizations..."

    # Create source list with full paths
    SOURCE_LIST=""
    for src in $BLAS_SOURCES; do
        SOURCE_LIST="$SOURCE_LIST ../$src"
    done

    emcc $COMMON_FLAGS $MAIN_FLAGS \
        $SOURCE_LIST \
        -o openblas-main.js

    # Verify the build
    if [[ ! -f "openblas-main.js" || ! -f "openblas-main.wasm" ]]; then
        log_error "MAIN_MODULE build failed"
        cd ..
        exit 1
    fi

    # Copy to install directory
    cp openblas-main.js openblas-main.wasm "../$INSTALL_DIR/"

    cd ..
    log_success "MAIN_MODULE build completed ($(du -h $INSTALL_DIR/openblas-main.wasm | cut -f1))"
}

# Display build summary
build_summary() {
    log_success "Build Summary:"
    echo "  📦 SIDE_MODULE: $INSTALL_DIR/openblas-side.wasm ($(du -h $INSTALL_DIR/openblas-side.wasm | cut -f1))"
    echo "  📦 MAIN_MODULE: $INSTALL_DIR/openblas-main.js + openblas-main.wasm ($(du -h $INSTALL_DIR/openblas-main.wasm | cut -f1))"
    echo ""
    echo "✨ Ready for:"
    echo "  • Production deployment (SIDE_MODULE)"
    echo "  • Development testing (MAIN_MODULE)"
    echo "  • NPM distribution (MAIN_MODULE)"
}

# Main build logic
main() {
    echo "🔧 OpenBLAS.wasm Dual Build System"
    echo "=================================="

    check_prerequisites
    setup_directories

    case "$VARIANT" in
        side)
            build_side_module
            ;;
        main)
            build_main_module
            ;;
        all)
            build_side_module
            build_main_module
            ;;
        *)
            log_error "Unknown variant: $VARIANT"
            echo "Usage: $0 [side|main|all]"
            exit 1
            ;;
    esac

    build_summary
}

# Error handling
trap 'log_error "Build failed at line $LINENO"' ERR

main "$@"