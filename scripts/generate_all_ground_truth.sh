#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_DIR="$PROJECT_ROOT/test"
BUILD_DIR="$TEST_DIR/build"
RESOURCES_DIR="$TEST_DIR/resources"

TOKENIZERS_CPP_DIR="$PROJECT_ROOT/test/third-party/tokenizers-cpp"

GENERATOR_SRC="$TEST_DIR/generate_ground_truth.cpp"
GENERATOR_BIN="$BUILD_DIR/generate_ground_truth"

echo "Using generator: $GENERATOR_BIN"
echo "Resources directory: $RESOURCES_DIR"
echo ""

echo "=================================================="
echo "Building ground truth generator..."
echo "=================================================="

mkdir -p "$BUILD_DIR"

cd "$TOKENIZERS_CPP_DIR"

# Build Rust library if not already built
if [ ! -f "$TOKENIZERS_CPP_DIR/rust/target/release/libtokenizers_c.a" ] && \
   [ ! -f "$TOKENIZERS_CPP_DIR/rust/target/release/libtokenizers_c.dylib" ]; then
    echo "Building Rust tokenizers library..."
    cd "$TOKENIZERS_CPP_DIR/rust"
    cargo build --release
    cd "$TOKENIZERS_CPP_DIR"
fi

if [ -f "$TOKENIZERS_CPP_DIR/rust/target/release/libtokenizers_c.a" ]; then
    TOKENIZERS_LIB="$TOKENIZERS_CPP_DIR/rust/target/release/libtokenizers_c.a"
elif [ -f "$TOKENIZERS_CPP_DIR/rust/target/release/libtokenizers_c.dylib" ]; then
    TOKENIZERS_LIB="$TOKENIZERS_CPP_DIR/rust/target/release/libtokenizers_c.dylib"
else
    echo -e "${RED}Error: Could not find tokenizers library${NC}"
    exit 1
fi

echo "Compiling ground truth generator..."
c++ -std=c++17 \
    -O3 \
    -I"$TOKENIZERS_CPP_DIR/include" \
    -I"$PROJECT_ROOT/third-party/json/include" \
    -I"$PROJECT_ROOT/include" \
    "$GENERATOR_SRC" \
    "$TOKENIZERS_CPP_DIR/src/huggingface_tokenizer.cc" \
    -o "$GENERATOR_BIN" \
    "$TOKENIZERS_LIB" \
    -framework CoreFoundation \
    -framework Security

echo -e "${GREEN}Build successful!${NC}"
echo ""

MODELS=(
    "phi"
    "minilm"
    "minilm_l12"
    "bielik"
    "hammer"
    "qwen2_5"
    "llama"
    "qwen3"
    "smollm"
    "lfm2"
    "lfm2vl"
    "gemma3"
    "gemma4"
    "glmOCR"
    "gpt2"
    "camemBERT"
    "t5small"
    "roberta"
)

export RESOURCES_PATH="$RESOURCES_DIR"

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_MODELS=()

for tokenizer in "${MODELS[@]}"; do
    echo "=================================================="
    echo "Generating ground truth for: $tokenizer"
    echo "=================================================="

    if "$GENERATOR_BIN" "$tokenizer"; then
        echo -e "${GREEN}✓ Successfully generated ground truth for $tokenizer${NC}"
        ((SUCCESS_COUNT++))
    else
        echo -e "${RED}✗ Failed to generate ground truth for $tokenizer${NC}"
        ((FAIL_COUNT++))
        FAILED_MODELS+=("$tokenizer")
    fi
    echo ""
done

echo "=================================================="
echo "Ground truth generation complete!"
echo "=================================================="
echo ""
echo -e "${GREEN}Successful: $SUCCESS_COUNT${NC}"
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Failed: $FAIL_COUNT${NC}"
    echo "Failed models: ${FAILED_MODELS[*]}"
fi
echo ""

echo "Generated files:"
ls -lh "$RESOURCES_DIR"/*_ground_truth.json 2>/dev/null || echo "No ground truth files found"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}All ground truth files generated successfully!${NC}"
    exit 0
else
    echo -e "${YELLOW}Some ground truth files failed to generate.${NC}"
    exit 1
fi
