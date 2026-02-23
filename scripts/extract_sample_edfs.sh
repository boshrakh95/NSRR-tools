#!/bin/bash
#
# Extract a sample of EDF files from compressed tar.zst archives for testing
# 
# Usage:
#   bash extract_sample_edfs.sh <dataset> <num_files>
#
# Example:
#   bash extract_sample_edfs.sh stages 10
#   bash extract_sample_edfs.sh shhs 20
#

set -e

DATASET=$1
NUM_FILES=${2:-10}

if [ -z "$DATASET" ]; then
    echo "Usage: $0 <dataset> [num_files]"
    echo "  dataset: stages, shhs, apples, or mros"
    echo "  num_files: number of sample files to extract (default: 10)"
    exit 1
fi

NSRR_ROOT="/scratch/boshra95/psg/nsrr"
TAR_DIR="$NSRR_ROOT/$DATASET/raw_tar"
OUTPUT_DIR="$NSRR_ROOT/$DATASET/sample_extraction"

echo "=========================================="
echo "Extracting sample EDFs from $DATASET"
echo "=========================================="
echo "Tar directory: $TAR_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of files: $NUM_FILES"
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Dataset-specific extraction
case "$DATASET" in
    stages)
        TAR_FILE="$TAR_DIR/stages_raw.tar.zst"
        echo "Extracting from: $TAR_FILE"
        
        # List contents and extract first N EDF files
        echo "Listing available EDFs..."
        tar -I zstd -tf "$TAR_FILE" | grep '\.edf$' | head -n "$NUM_FILES" > /tmp/stages_sample_list.txt
        
        echo "Extracting $(wc -l < /tmp/stages_sample_list.txt) files..."
        tar -I zstd -xf "$TAR_FILE" -C "$OUTPUT_DIR" -T /tmp/stages_sample_list.txt
        
        echo "✓ Done! Files extracted to: $OUTPUT_DIR"
        ;;
        
    shhs)
        TAR_FILE="$TAR_DIR/shhs1_raw.tar.zst"
        echo "Extracting from: $TAR_FILE (visit 1)"
        
        echo "Listing available EDFs..."
        tar -I zstd -tf "$TAR_FILE" | grep '\.edf$' | head -n "$NUM_FILES" > /tmp/shhs_sample_list.txt
        
        echo "Extracting $(wc -l < /tmp/shhs_sample_list.txt) files..."
        tar -I zstd -xf "$TAR_FILE" -C "$OUTPUT_DIR" -T /tmp/shhs_sample_list.txt
        
        echo "✓ Done! Files extracted to: $OUTPUT_DIR"
        ;;
        
    apples)
        TAR_FILE="$TAR_DIR/apples_raw.tar.zst"
        echo "Extracting from: $TAR_FILE"
        
        echo "Listing available EDFs..."
        tar -I zstd -tf "$TAR_FILE" | grep '\.edf$' | head -n "$NUM_FILES" > /tmp/apples_sample_list.txt
        
        echo "Extracting $(wc -l < /tmp/apples_sample_list.txt) files..."
        tar -I zstd -xf "$TAR_FILE" -C "$OUTPUT_DIR" -T /tmp/apples_sample_list.txt
        
        echo "✓ Done! Files extracted to: $OUTPUT_DIR"
        ;;
        
    mros)
        TAR_FILE="$TAR_DIR/mrosv1_raw.tar.zst"
        echo "Extracting from: $TAR_FILE (visit 1)"
        
        echo "Listing available EDFs..."
        tar -I zstd -tf "$TAR_FILE" | grep '\.edf$' | head -n "$NUM_FILES" > /tmp/mros_sample_list.txt
        
        echo "Extracting $(wc -l < /tmp/mros_sample_list.txt) files..."
        tar -I zstd -xf "$TAR_FILE" -C "$OUTPUT_DIR" -T /tmp/mros_sample_list.txt
        
        echo "✓ Done! Files extracted to: $OUTPUT_DIR"
        ;;
        
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo "Supported: stages, shhs, apples, mros"
        exit 1
        ;;
esac

# Show what was extracted
echo
echo "Extracted files:"
find "$OUTPUT_DIR" -name "*.edf" -type f | head -10
echo
echo "Total EDFs extracted: $(find "$OUTPUT_DIR" -name "*.edf" -type f | wc -l)"
