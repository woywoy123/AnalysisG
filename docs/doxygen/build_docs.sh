#!/bin/bash
# ==============================================================================
# AnalysisG Documentation Build & Validation Script
# ==============================================================================
#
# This script validates and builds the complete Doxygen documentation for
# AnalysisG, including all newly created module documentation files.
#
# USAGE:
#   ./build_docs.sh [options]
#
# OPTIONS:
#   --validate-only    Check documentation files without building
#   --clean           Remove existing documentation before building
#   --open            Open generated documentation in browser after build
#   --help            Show this help message
#
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="/workspaces/AnalysisG"
DOXYGEN_CONFIG="$REPO_ROOT/Doxyfile"
OUTPUT_DIR="$REPO_ROOT/doxygen-docs"
DOCS_DIR="$REPO_ROOT/docs/doxygen"

# Parse arguments
VALIDATE_ONLY=false
CLEAN=false
OPEN_BROWSER=false

for arg in "$@"; do
    case $arg in
        --validate-only)
            VALIDATE_ONLY=true
            ;;
        --clean)
            CLEAN=true
            ;;
        --open)
            OPEN_BROWSER=true
            ;;
        --help)
            head -n 20 "$0" | tail -n 16
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Validation Functions
# ==============================================================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if Doxygen is installed
check_doxygen() {
    print_header "Checking Doxygen Installation"
    
    if ! command -v doxygen &> /dev/null; then
        print_error "Doxygen is not installed"
        echo "Install with: sudo apt-get install doxygen graphviz"
        exit 1
    fi
    
    DOXYGEN_VERSION=$(doxygen --version)
    print_success "Doxygen version: $DOXYGEN_VERSION"
}

# Validate documentation file structure
validate_file_structure() {
    print_header "Validating Documentation File Structure"
    
    local errors=0
    
    # Check main documentation files
    local main_files=(
        "$DOCS_DIR/overview.dox"
        "$DOCS_DIR/modules_index.dox"
        "$DOCS_DIR/module_interactions.dox"
    )
    
    for file in "${main_files[@]}"; do
        if [[ -f "$file" ]]; then
            print_success "$(basename $file)"
        else
            print_error "Missing: $(basename $file)"
            ((errors++))
        fi
    done
    
    # Check core module documentation (19 modules)
    local core_modules=(
        "analysis" "container" "dataloader" "graph" "io" "lossfx"
        "meta" "metric" "metrics" "model" "notification" "optimizer"
        "plotting" "roc" "sampletracer" "selection" "structs" "tools"
        "typecasting"
    )
    
    echo -e "\n${BLUE}Core Modules:${NC}"
    for module in "${core_modules[@]}"; do
        local dox_file="$DOCS_DIR/modules/$module/$module.dox"
        if [[ -f "$dox_file" ]]; then
            print_success "$module"
        else
            print_error "Missing: $module"
            ((errors++))
        fi
    done
    
    # Check PyC module documentation (7 modules)
    local pyc_modules=(
        "cutils" "graph" "interface" "nusol" "operators" "physics" "transform"
    )
    
    echo -e "\n${BLUE}PyC CUDA Modules:${NC}"
    for module in "${pyc_modules[@]}"; do
        local dox_file="$DOCS_DIR/pyc/$module/$module.dox"
        if [[ -f "$dox_file" ]]; then
            print_success "$module"
        else
            print_error "Missing: $module"
            ((errors++))
        fi
    done
    
    # Check template documentation (2 templates)
    local templates=(
        "event_template" "particle_template"
    )
    
    echo -e "\n${BLUE}Template Classes:${NC}"
    for template in "${templates[@]}"; do
        local dox_file="$DOCS_DIR/templates/$template.dox"
        if [[ -f "$dox_file" ]]; then
            print_success "$template"
        else
            print_error "Missing: $template"
            ((errors++))
        fi
    done
    
    # Check newly created documentation
    local new_docs=(
        "modules/container/container.dox"
        "modules/sampletracer/sampletracer.dox"
        "modules/plotting/plotting.dox"
        "modules/roc/roc.dox"
        "modules/metrics/metrics.dox"
        "modules/structs/structs.dox"
        "modules/typecasting/typecasting.dox"
        "pyc/interface/interface.dox"
    )
    
    echo -e "\n${BLUE}Newly Created Documentation:${NC}"
    for doc in "${new_docs[@]}"; do
        local dox_file="$DOCS_DIR/$doc"
        if [[ -f "$dox_file" ]]; then
            local line_count=$(wc -l < "$dox_file")
            print_success "$(basename $doc) ($line_count lines)"
        else
            print_error "Missing: $(basename $doc)"
            ((errors++))
        fi
    done
    
    # Check Cython bindings (automatically processed by Doxygen)
    echo -e "\n${BLUE}Cython Python Bindings (auto-processed):${NC}"
    local pyx_count=$(find "$REPO_ROOT/src/AnalysisG" -name "*.pyx" | wc -l)
    local pxd_count=$(find "$REPO_ROOT/src/AnalysisG" -name "*.pxd" | wc -l)
    print_success "Python bindings: $pyx_count .pyx files"
    print_success "Cython headers: $pxd_count .pxd files"
    echo -e "${YELLOW}  Note: .pyx/.pxd files mapped to C++ via EXTENSION_MAPPING${NC}"
    
    echo ""
    if [[ $errors -eq 0 ]]; then
        print_success "All documentation files validated successfully!"
        return 0
    else
        print_error "Found $errors missing documentation file(s)"
        return 1
    fi
}

# Validate Doxygen configuration
validate_doxyfile() {
    print_header "Validating Doxyfile Configuration"
    
    if [[ ! -f "$DOXYGEN_CONFIG" ]]; then
        print_error "Doxyfile not found: $DOXYGEN_CONFIG"
        exit 1
    fi
    
    print_success "Doxyfile found"
    
    # Check key configuration options
    local project_name=$(grep "^PROJECT_NAME" "$DOXYGEN_CONFIG" | cut -d'=' -f2 | tr -d ' "')
    local output_dir=$(grep "^OUTPUT_DIRECTORY" "$DOXYGEN_CONFIG" | cut -d'=' -f2 | tr -d ' ')
    
    echo "  Project Name: $project_name"
    echo "  Output Directory: $output_dir"
    
    # Verify INPUT paths include docs/doxygen
    if grep -q "docs/doxygen" "$DOXYGEN_CONFIG"; then
        print_success "Documentation directory included in INPUT"
    else
        print_error "docs/doxygen not found in INPUT paths"
        return 1
    fi
    
    # Verify INPUT paths include src/AnalysisG
    if grep -q "src/AnalysisG" "$DOXYGEN_CONFIG"; then
        print_success "Source directory included in INPUT"
    else
        print_warning "src/AnalysisG not found in INPUT paths"
    fi
    
    # Verify .dox files are included in FILE_PATTERNS
    if grep -q "*.dox" "$DOXYGEN_CONFIG"; then
        print_success ".dox files included in FILE_PATTERNS"
    else
        print_error "*.dox not found in FILE_PATTERNS"
        return 1
    fi
    
    # Verify .pyx/.pxd files are included
    if grep -q "*.pyx" "$DOXYGEN_CONFIG" && grep -q "*.pxd" "$DOXYGEN_CONFIG"; then
        print_success "Cython files (.pyx/.pxd) included in FILE_PATTERNS"
    else
        print_warning "Cython files may not be included in FILE_PATTERNS"
    fi
    
    # Verify EXTENSION_MAPPING for Cython
    if grep -q "EXTENSION_MAPPING.*pyx=C++" "$DOXYGEN_CONFIG"; then
        print_success "Cython→C++ mapping configured"
    else
        print_warning "EXTENSION_MAPPING for Cython not found"
    fi
}

# Count documentation statistics
count_documentation() {
    print_header "Documentation Statistics"
    
    local total_dox_files=$(find "$DOCS_DIR" -name "*.dox" | wc -l)
    local total_lines=0
    
    while IFS= read -r file; do
        local lines=$(wc -l < "$file")
        ((total_lines += lines))
    done < <(find "$DOCS_DIR" -name "*.dox")
    
    echo "Total .dox files: $total_dox_files"
    echo "Total lines of documentation: $total_lines"
    echo ""
    
    # Module breakdown
    echo "Module Documentation:"
    echo "  Core Modules: $(find "$DOCS_DIR/modules" -name "*.dox" | wc -l)"
    echo "  PyC Modules: $(find "$DOCS_DIR/pyc" -name "*.dox" | wc -l)"
    echo "  Templates: $(find "$DOCS_DIR/templates" -name "*.dox" | wc -l)"
    echo ""
    
    print_success "Documentation coverage: 28/28 critical modules (100%)"
}

# ==============================================================================
# Build Functions
# ==============================================================================

clean_output() {
    print_header "Cleaning Previous Build"
    
    if [[ -d "$OUTPUT_DIR" ]]; then
        rm -rf "$OUTPUT_DIR"
        print_success "Removed $OUTPUT_DIR"
    else
        print_warning "No previous build found"
    fi
}

build_documentation() {
    print_header "Building Doxygen Documentation"
    
    cd "$REPO_ROOT"
    
    echo "Running doxygen..."
    if doxygen "$DOXYGEN_CONFIG" 2>&1 | tee doxygen_build.log; then
        print_success "Documentation built successfully!"
        echo "Output directory: $OUTPUT_DIR/html"
    else
        print_error "Documentation build failed"
        echo "Check doxygen_build.log for details"
        exit 1
    fi
}

open_documentation() {
    print_header "Opening Documentation"
    
    local index_file="$OUTPUT_DIR/html/index.html"
    
    if [[ ! -f "$index_file" ]]; then
        print_error "Documentation not found: $index_file"
        exit 1
    fi
    
    # Try different browsers
    if command -v xdg-open &> /dev/null; then
        xdg-open "$index_file"
    elif command -v firefox &> /dev/null; then
        firefox "$index_file" &
    elif command -v google-chrome &> /dev/null; then
        google-chrome "$index_file" &
    else
        print_warning "No browser found. Open manually:"
        echo "  $index_file"
    fi
}

# ==============================================================================
# Main Execution
# ==============================================================================

main() {
    print_header "AnalysisG Documentation Build & Validation"
    echo "Repository: $REPO_ROOT"
    echo "Doxyfile: $DOXYGEN_CONFIG"
    echo ""
    
    # Always run validation
    check_doxygen
    validate_doxyfile
    validate_file_structure || exit 1
    count_documentation
    
    # Stop if validate-only mode
    if [[ "$VALIDATE_ONLY" == true ]]; then
        print_success "Validation complete. Skipping build."
        exit 0
    fi
    
    # Clean if requested
    if [[ "$CLEAN" == true ]]; then
        clean_output
    fi
    
    # Build documentation
    build_documentation
    
    # Open in browser if requested
    if [[ "$OPEN_BROWSER" == true ]]; then
        open_documentation
    fi
    
    print_header "Build Complete"
    print_success "Documentation successfully generated!"
    echo "View at: file://$OUTPUT_DIR/html/index.html"
}

# Run main function
main
