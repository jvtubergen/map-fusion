#!/bin/bash

# Build script for generating release.tar artifact
# This orchestrates the data pipeline to create distributable research artifacts

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION_FILE="$SCRIPT_DIR/VERSION"

# Read version from VERSION file
if [ -f "$VERSION_FILE" ]; then
    VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')
else
    echo "ERROR: VERSION file not found"
    exit 1
fi

# Parse command line arguments
STAGE="all"
CLEAN=false
TEST_MODE=false
OUTPUT_DIR="$SCRIPT_DIR/build/release"

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --stage <stage>     Build specific stage: input|unimodal|fused|results|all (default: all)"
            echo "  --clean             Clean build directory before building"
            echo "  --test              Test mode: quick validation without full data generation"
            echo "  --version <ver>     Override version from VERSION file"
            echo "  --output <dir>      Output directory (default: build/release)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

RELEASE_DIR="$OUTPUT_DIR/release-v${VERSION}"

echo "========================================"
echo "Building map_fusion release artifacts"
echo "========================================"
echo "Version: $VERSION"
echo "Stage: $STAGE"
echo "Output: $RELEASE_DIR"
echo "========================================"
echo ""

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$OUTPUT_DIR"
fi

# Create release directory structure
mkdir -p "$RELEASE_DIR"/{input/{gps,satellite},unimodal,fused,results}

# Create metadata
echo "Generating metadata..."
cat > "$RELEASE_DIR/metadata.json" <<EOF
{
  "version": "$VERSION",
  "build_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "pipeline_stages": {
    "input": "partial",
    "unimodal": "not_implemented",
    "fused": "not_implemented",
    "results": "not_implemented"
  }
}
EOF

# Copy VERSION file
cp "$VERSION_FILE" "$RELEASE_DIR/VERSION"

# Stage 1: Input
if [ "$STAGE" = "all" ] || [ "$STAGE" = "input" ]; then
    echo ""
    echo "==== Stage 1: Input ===="
    echo ""

    echo "Processing GPS data..."
    if [ "$TEST_MODE" = true ]; then
        echo "[TEST MODE] Skipping actual GPS download, using placeholder..."
        mkdir -p "$RELEASE_DIR/input/gps/berlin" "$RELEASE_DIR/input/gps/chicago"
        echo "Test GPS data for Berlin" > "$RELEASE_DIR/input/gps/berlin/test.txt"
        echo "Test GPS data for Chicago" > "$RELEASE_DIR/input/gps/chicago/test.txt"
    else
        # Run actual GPS data preparation
        cd "$SCRIPT_DIR"
        python -m map_fusion.pipeline.input.gps "$RELEASE_DIR/input/gps"
    fi

    echo "✓ GPS data processed"

    # Satellite imagery (not yet implemented)
    echo ""
    echo "Satellite imagery: Not yet implemented"
    cat > "$RELEASE_DIR/input/satellite/README.md" <<EOF
# Satellite Imagery

This directory will contain processed satellite imagery tiles in future releases.

**Status**: Not yet implemented (planned for v0.2.0-alpha series)

**Expected contents**:
- Tiled satellite images for Berlin and Chicago regions
- Preprocessed for sat2graph inference
- Downloaded from Google Maps API
EOF
fi

# Stage 2: Unimodal (not yet implemented)
if [ "$STAGE" = "all" ] || [ "$STAGE" = "unimodal" ]; then
    echo ""
    echo "==== Stage 2: Unimodal ===="
    echo "Status: Not yet implemented (planned for v0.3.0-alpha series)"

    cat > "$RELEASE_DIR/unimodal/README.md" <<EOF
# Unimodal Map Reconstructions

This directory will contain independently reconstructed road network maps in future releases.

**Status**: Not yet implemented (planned for v0.3.0-alpha series)

**Expected contents**:
- gps.graph / gps.graphml - GPS-based road network
- sat.graph / sat.graphml - Satellite-based road network
EOF
fi

# Stage 3: Fused (not yet implemented)
if [ "$STAGE" = "all" ] || [ "$STAGE" = "fused" ]; then
    echo ""
    echo "==== Stage 3: Fused ===="
    echo "Status: Not yet implemented (planned for v0.4.0-alpha series)"

    cat > "$RELEASE_DIR/fused/README.md" <<EOF
# Fused Map Networks

This directory will contain merged road networks combining GPS and satellite data.

**Status**: Not yet implemented (planned for v0.4.0-alpha series)

**Expected contents**:
- A.graph / A.graphml - Conservative fusion (high precision)
- B.graph / B.graphml - Moderate fusion (balanced)
- C.graph / C.graphml - Aggressive fusion (high recall)
EOF
fi

# Stage 4: Results (not yet implemented)
if [ "$STAGE" = "all" ] || [ "$STAGE" = "results" ]; then
    echo ""
    echo "==== Stage 4: Results ===="
    echo "Status: Not yet implemented (planned for v0.5.0-alpha series)"

    cat > "$RELEASE_DIR/results/README.md" <<EOF
# Analysis Results

This directory will contain similarity metrics and visualizations in future releases.

**Status**: Not yet implemented (planned for v0.5.0-alpha series)

**Expected contents**:
- APLS metrics comparing reconstructions to OSM ground truth
- TOPO metrics for topological similarity analysis
- Visualizations and plots for publication
- Statistical analysis reports
EOF
fi

# Create main README
echo ""
echo "Generating release README..."
cat > "$RELEASE_DIR/README.md" <<EOF
# map_fusion Release $VERSION

This release contains data artifacts generated by the map_fusion data pipeline.

## Overview

**map_fusion** is a research project focused on multi-source road network reconstruction and map similarity analysis. This release includes:

- Processed GPS trace data for Berlin and Chicago
- Pipeline infrastructure for future data stages

## What's Included

### Input Data
- \`input/gps/berlin/\` - Processed GPS traces for Berlin
- \`input/gps/chicago/\` - Processed GPS traces for Chicago

### Status of Other Pipeline Stages
- **Satellite imagery**: Not yet implemented
- **Unimodal reconstruction**: Not yet implemented
- **Map fusion**: Not yet implemented
- **Results/metrics**: Not yet implemented

See individual README files in each directory for details.

## Version Information

- **Version**: $VERSION
- **Build date**: $(date -u +"%Y-%m-%d")
- **Git commit**: $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')

See \`metadata.json\` for complete build information.

## Data Format

### GPS Traces

Format: Plain text, one point per line
\`\`\`
<longitude> <latitude> <timestamp>
\`\`\`

Coordinate system: WGS84 (EPSG:4326)

## Usage

This data is intended for research purposes. To reproduce the pipeline:

1. Clone repository: \`git clone https://github.com/yourusername/map-fusion.git\`
2. Install dependencies: \`hatch shell\`
3. Run pipeline: \`bash build_release_data.sh\`

See repository documentation for detailed usage instructions.

## Documentation

- Pipeline architecture: \`documentation/pipeline.md\`
- Release process: \`documentation/releases.md\`
- Repository: https://github.com/yourusername/map-fusion

## Citation

If you use this data in your research, please cite:

\`\`\`
[Your citation information]
\`\`\`

## License

[Your license information]

## Contact

[Your contact information]
EOF

# Create tarball
echo ""
echo "Creating release archive..."
cd "$OUTPUT_DIR"
TARBALL="release-v${VERSION}.tar.gz"
tar -czf "$TARBALL" "release-v${VERSION}/"

# Calculate checksum
echo ""
echo "Calculating checksums..."
CHECKSUM=$(sha256sum "$TARBALL" | cut -d' ' -f1)
echo "$CHECKSUM  $TARBALL" > "${TARBALL}.sha256"

echo ""
echo "========================================"
echo "✓ Build complete!"
echo "========================================"
echo "Archive: $OUTPUT_DIR/$TARBALL"
echo "SHA256: $CHECKSUM"
echo "Size: $(du -h "$OUTPUT_DIR/$TARBALL" | cut -f1)"
echo ""
echo "To create a GitHub release:"
echo "  1. git tag -a v${VERSION} -m 'Release v${VERSION}'"
echo "  2. git push origin v${VERSION}"
echo "  3. Upload $TARBALL to GitHub release"
echo ""
