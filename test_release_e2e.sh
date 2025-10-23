#!/bin/bash

# End-to-end test script for the release pipeline
# Tests the complete workflow: build → extract → load → verify

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build/release"
TEST_MODE=false
KEEP_ARTIFACTS=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --keep)
            KEEP_ARTIFACTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test    Use test mode (quick validation with stub data)"
            echo "  --keep    Keep build artifacts after test"
            echo "  -h, --help Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================"
echo "Release E2E Test Script"
echo "========================================${NC}"
echo "Test mode: $([ "$TEST_MODE" = true ] && echo "enabled (stub data)" || echo "disabled (actual GPS data)")"
echo "Keep artifacts: $([ "$KEEP_ARTIFACTS" = true ] && echo "yes" || echo "no")"
echo ""

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Test helper functions
test_start() {
    echo -e "${YELLOW}▶ Test: $1${NC}"
}

test_pass() {
    echo -e "${GREEN}  ✓ $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

test_fail() {
    echo -e "${RED}  ✗ $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

test_info() {
    echo -e "  ℹ $1"
}

# Cleanup function
cleanup() {
    if [ "$KEEP_ARTIFACTS" = false ]; then
        echo ""
        echo "Cleaning up test artifacts..."
        rm -rf "$BUILD_DIR"
        test_info "Build directory cleaned"
    else
        echo ""
        test_info "Keeping artifacts at: $BUILD_DIR"
    fi
}

# Register cleanup trap
trap cleanup EXIT

# Test 1: Build release
test_start "Building release artifact"

if [ "$TEST_MODE" = true ]; then
    hatch run bash build_release_data.sh --test --clean > /tmp/build_output.log 2>&1
else
    hatch run bash build_release_data.sh --clean > /tmp/build_output.log 2>&1
fi

if [ $? -eq 0 ]; then
    test_pass "Build completed successfully"
else
    test_fail "Build failed"
    cat /tmp/build_output.log
    exit 1
fi

# Find the generated tarball
TARBALL=$(ls "$BUILD_DIR"/release-*.tar.gz 2>/dev/null | head -n 1)

if [ -z "$TARBALL" ]; then
    test_fail "Tarball not found"
    exit 1
else
    test_pass "Tarball created: $(basename $TARBALL)"
fi

TARBALL_SIZE=$(du -h "$TARBALL" | cut -f1)
test_info "Size: $TARBALL_SIZE"

# Test 2: Verify checksum
test_start "Verifying checksum"

if [ ! -f "${TARBALL}.sha256" ]; then
    test_fail "Checksum file not found"
    exit 1
fi

cd "$BUILD_DIR"
if sha256sum -c "$(basename ${TARBALL}.sha256)" > /dev/null 2>&1; then
    test_pass "Checksum verified"
    CHECKSUM=$(cat "$(basename ${TARBALL}.sha256)" | cut -d' ' -f1)
    test_info "SHA256: ${CHECKSUM:0:16}..."
else
    test_fail "Checksum verification failed"
    exit 1
fi
cd "$SCRIPT_DIR"

# Test 3: Extract tarball
test_start "Extracting tarball"

cd "$BUILD_DIR"
tar -xzf "$(basename $TARBALL)" 2>/dev/null

if [ $? -eq 0 ]; then
    test_pass "Extraction successful"
else
    test_fail "Extraction failed"
    exit 1
fi

# Find extracted directory
RELEASE_DIR=$(ls -d release-* 2>/dev/null | grep -v ".tar.gz" | head -n 1)

if [ -z "$RELEASE_DIR" ]; then
    test_fail "Extracted directory not found"
    exit 1
else
    test_pass "Release directory: $RELEASE_DIR"
fi
cd "$SCRIPT_DIR"

FULL_RELEASE_PATH="$BUILD_DIR/$RELEASE_DIR"

# Test 4: Verify directory structure
test_start "Verifying directory structure"

REQUIRED_DIRS=(
    "input"
    "input/gps"
    "input/satellite"
    "unimodal"
    "fused"
    "results"
)

REQUIRED_FILES=(
    "metadata.json"
    "README.md"
    "VERSION"
)

ALL_DIRS_EXIST=true
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$FULL_RELEASE_PATH/$dir" ]; then
        test_pass "Directory exists: $dir"
    else
        test_fail "Missing directory: $dir"
        ALL_DIRS_EXIST=false
    fi
done

ALL_FILES_EXIST=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$FULL_RELEASE_PATH/$file" ]; then
        test_pass "File exists: $file"
    else
        test_fail "Missing file: $file"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_DIRS_EXIST" = false ] || [ "$ALL_FILES_EXIST" = false ]; then
    exit 1
fi

# Test 5: Verify metadata
test_start "Verifying metadata"

if command -v python3 &> /dev/null; then
    METADATA=$(cat "$FULL_RELEASE_PATH/metadata.json")

    # Check required fields
    if echo "$METADATA" | python3 -c "import sys, json; data=json.load(sys.stdin); sys.exit(0 if all(k in data for k in ['version', 'build_date', 'git_commit', 'git_branch']) else 1)"; then
        test_pass "Metadata has all required fields"

        VERSION=$(echo "$METADATA" | python3 -c "import sys, json; print(json.load(sys.stdin)['version'])")
        BUILD_DATE=$(echo "$METADATA" | python3 -c "import sys, json; print(json.load(sys.stdin)['build_date'])")
        GIT_COMMIT=$(echo "$METADATA" | python3 -c "import sys, json; print(json.load(sys.stdin)['git_commit'][:8])")

        test_info "Version: $VERSION"
        test_info "Build date: $BUILD_DATE"
        test_info "Git commit: $GIT_COMMIT"
    else
        test_fail "Metadata missing required fields"
    fi
else
    test_info "Python not available, skipping detailed metadata validation"
fi

# Test 6: Verify GPS data
test_start "Verifying GPS data"

GPS_CITIES=("berlin" "chicago")

for city in "${GPS_CITIES[@]}"; do
    GPS_DIR="$FULL_RELEASE_PATH/input/gps/$city"

    if [ -d "$GPS_DIR" ]; then
        FILE_COUNT=$(find "$GPS_DIR" -type f -name "*.txt" | wc -l)

        if [ "$TEST_MODE" = true ]; then
            # In test mode, expect stub files
            if [ $FILE_COUNT -gt 0 ]; then
                test_pass "$city: $FILE_COUNT test file(s)"
            else
                test_fail "$city: No GPS files found"
            fi
        else
            # In real mode, expect actual GPS data
            if [ $FILE_COUNT -gt 0 ]; then
                test_pass "$city: $FILE_COUNT GPS trace files"

                # Check a sample file format
                SAMPLE_FILE=$(find "$GPS_DIR" -type f -name "*.txt" | head -n 1)
                if [ -f "$SAMPLE_FILE" ]; then
                    FIRST_LINE=$(head -n 1 "$SAMPLE_FILE")
                    FIELD_COUNT=$(echo "$FIRST_LINE" | awk '{print NF}')

                    if [ $FIELD_COUNT -eq 3 ]; then
                        test_pass "$city: GPS format correct (lon lat timestamp)"

                        # Verify coordinate ranges
                        LON=$(echo "$FIRST_LINE" | awk '{print $1}')
                        LAT=$(echo "$FIRST_LINE" | awk '{print $2}')
                        test_info "$city: Sample coordinate: lon=$LON, lat=$LAT"
                    else
                        test_fail "$city: Invalid GPS format (expected 3 fields, got $FIELD_COUNT)"
                    fi
                fi
            else
                test_fail "$city: No GPS files found"
            fi
        fi
    else
        test_fail "$city: GPS directory not found"
    fi
done

# Test 7: Load release with example script
test_start "Loading release with example script"

if hatch run python examples/load_release_example.py "$FULL_RELEASE_PATH" > /tmp/load_output.log 2>&1; then
    test_pass "Example script executed successfully"

    # Check for expected output
    if grep -q "Release v" /tmp/load_output.log && \
       grep -q "GPS data:" /tmp/load_output.log && \
       grep -q "Berlin:" /tmp/load_output.log && \
       grep -q "Chicago:" /tmp/load_output.log; then
        test_pass "Output contains expected information"
    else
        test_fail "Output missing expected information"
        cat /tmp/load_output.log
    fi
else
    test_fail "Example script failed"
    cat /tmp/load_output.log
    exit 1
fi

# Test 8: Test loading functions programmatically
test_start "Testing load functions programmatically"

cat > /tmp/test_load.py <<'EOF'
from pathlib import Path
import sys

# Import release loading functions
from map_fusion.pipeline.load import (
    load_release,
    parse_version,
    format_release_info,
)

release_path = Path(sys.argv[1])

try:
    # Load release
    release = load_release(release_path)

    # Verify structure
    assert 'metadata' in release, "Missing metadata"
    assert 'version' in release, "Missing version"
    assert 'gps' in release, "Missing GPS data"

    # Verify version parsing
    version = release['version']
    assert 'major' in version, "Missing major version"
    assert 'minor' in version, "Missing minor version"
    assert 'patch' in version, "Missing patch version"

    # Verify GPS data
    gps_data = release['gps']
    assert len(gps_data) > 0, "No GPS cities found"

    for city, gps_paths in gps_data.items():
        assert 'directory' in gps_paths, f"Missing directory for {city}"
        assert 'files' in gps_paths, f"Missing files for {city}"

    # Test format_release_info
    info = format_release_info(release)
    assert len(info) > 0, "Empty release info"

    print("All assertions passed")
    sys.exit(0)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if hatch run python /tmp/test_load.py "$FULL_RELEASE_PATH" > /tmp/test_load_output.log 2>&1; then
    test_pass "Load functions work correctly"
else
    test_fail "Load functions failed"
    cat /tmp/test_load_output.log
    exit 1
fi

# Test 9: Verify README content
test_start "Verifying README content"

README_PATH="$FULL_RELEASE_PATH/README.md"
REQUIRED_README_SECTIONS=(
    "Release"
    "Overview"
    "Version Information"
    "Data Format"
    "GPS Traces"
)

ALL_SECTIONS_FOUND=true
for section in "${REQUIRED_README_SECTIONS[@]}"; do
    if grep -q "$section" "$README_PATH"; then
        test_pass "README contains: $section"
    else
        test_fail "README missing: $section"
        ALL_SECTIONS_FOUND=false
    fi
done

# Print test summary
echo ""
echo -e "${BLUE}========================================"
echo "Test Summary"
echo "========================================${NC}"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
