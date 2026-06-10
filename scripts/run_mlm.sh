#!/bin/bash

# Shell script to train and run inference with Masked Language Model for antibody design

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
CONFIG="configs/training.yaml"
CHECKPOINT=""
MODE="train"  # train, inference, or both
CHAIN="heavy"
OUTPUT_DIR="mlm_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --chain)
            CHAIN="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_mlm.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode {train|inference|both}    Mode to run (default: train)"
            echo "  --config CONFIG_PATH             Path to config file (default: configs/training.yaml)"
            echo "  --checkpoint CHECKPOINT_PATH     Path to checkpoint for inference"
            echo "  --chain {heavy|light}           Which chain to design in inference (default: heavy)"
            echo "  --output OUTPUT_DIR             Output directory (default: mlm_results)"
            echo "  --help                          Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to print section headers
print_header() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

# TRAINING
if [[ "$MODE" == "train" ]] || [[ "$MODE" == "both" ]]; then
    print_header "TRAINING MASKED LANGUAGE MODEL"
    
    if [[ ! -f "$CONFIG" ]]; then
        echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
        exit 1
    fi
    
    echo "Configuration: $CONFIG"
    echo "Output Directory: $OUTPUT_DIR"
    echo ""
    
    python scripts/train_mlm.py \
        --config-path "../configs" \
        --config-name "training" \
        trainer.model_output_dir="$OUTPUT_DIR"
    
    # Find the latest checkpoint
    LATEST_CHECKPOINT=$(find "$OUTPUT_DIR/checkpoints_mlm" -name "last.ckpt" 2>/dev/null | head -1)
    if [[ -n "$LATEST_CHECKPOINT" ]]; then
        echo -e "\n${GREEN}✓ Training completed!${NC}"
        echo "Latest checkpoint: $LATEST_CHECKPOINT"
        CHECKPOINT="$LATEST_CHECKPOINT"
    else
        echo -e "${YELLOW}Warning: Could not find checkpoint automatically${NC}"
    fi
fi

# INFERENCE
if [[ "$MODE" == "inference" ]] || [[ "$MODE" == "both" ]]; then
    print_header "RUNNING INFERENCE"
    
    if [[ -z "$CHECKPOINT" ]]; then
        echo -e "${RED}Error: No checkpoint specified. Use --checkpoint to specify a checkpoint path.${NC}"
        exit 1
    fi
    
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo -e "${RED}Error: Checkpoint not found: $CHECKPOINT${NC}"
        exit 1
    fi
    
    echo "Checkpoint: $CHECKPOINT"
    echo "Chain to design: $CHAIN"
    echo "Output Directory: $OUTPUT_DIR"
    echo ""
    
    mkdir -p "$OUTPUT_DIR/inference"
    
    python scripts/inference_mlm.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --chain "$CHAIN" \
        --output "$OUTPUT_DIR/inference/mlm_design_${CHAIN}.csv"
    
    echo -e "\n${GREEN}✓ Inference completed!${NC}"
    echo "Results saved to: $OUTPUT_DIR/inference/mlm_design_${CHAIN}.csv"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All done!${NC}"
echo -e "${GREEN}========================================${NC}\n"
