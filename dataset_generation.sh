#!/bin/bash

# Create a shell script to generate datasets with different values of n
# Parameters: n = 6, 12, 16, 20 with num_samples = 20 for each

# Display script banner
echo "====================================================="
echo "  Dataset Generation Script"
echo "  Generating datasets with n = 6, 12, 16, 20"
echo "====================================================="

# Create output directory if it doesn't exist
OUTPUT_DIR="dataset"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Function to run the generation with progress indication
run_generation() {
    local n=$1
    echo ""
    echo "---------------------------------"
    echo "Generating dataset with n = $n..."
    echo "---------------------------------"
    python dataset_generation.py --n $n --num_samples 20 --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully generated dataset with n = $n"
    else
        echo "✗ Failed to generate dataset with n = $n"
    fi
}

# Run the script for each value of n
run_generation 6
run_generation 12
run_generation 16
run_generation 20

echo ""
echo "====================================================="
echo "Dataset generation completed!"
echo "Check the '$OUTPUT_DIR' directory for generated files:"
echo " - $OUTPUT_DIR/6.csv"
echo " - $OUTPUT_DIR/12.csv"
echo " - $OUTPUT_DIR/16.csv"
echo " - $OUTPUT_DIR/20.csv"
echo "====================================================="