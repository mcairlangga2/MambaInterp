#!/bin/bash

# Process the datasets that were previously generated
# This script processes each CSV file to create new files with the same relation

# Display script banner
echo "====================================================="
echo "  Dataset Processing Script"
echo "  Processing datasets with n = 6, 12, 16, 20"
echo "====================================================="

# Directory containing generated datasets
DATASET_DIR="dataset"

# Function to process a single dataset file
process_dataset() {
    local n=$1
    local input_file="$DATASET_DIR/$n.csv"
    local output_file="$DATASET_DIR/${n}_same_relation.csv"
    
    echo ""
    echo "---------------------------------"
    echo "Processing dataset with n = $n..."
    echo "---------------------------------"
    
    if [ ! -f "$input_file" ]; then
        echo "✗ Input file not found: $input_file"
        return 1
    fi
    
    # Run the processing script for this file
    python convert_dataset.py --input_dir "$input_file" --output_dir "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed dataset: $output_file"
    else
        echo "✗ Failed to process dataset with n = $n"
    fi
}

# Check if the dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' not found!"
    echo "Please run generate_datasets.sh first to create the original datasets."
    exit 1
fi

# Process each dataset
process_dataset 6
process_dataset 12
process_dataset 16
process_dataset 20

echo ""
echo "====================================================="
echo "Dataset processing completed!"
echo "Processed files:"
echo " - $DATASET_DIR/6_same_relation.csv"
echo " - $DATASET_DIR/12_same_relation.csv"
echo " - $DATASET_DIR/16_same_relation.csv"
echo " - $DATASET_DIR/20_same_relation.csv"
echo "====================================================="