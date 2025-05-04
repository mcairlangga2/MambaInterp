#!/usr/bin/env python3
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import torch
import os
import argparse
import glob
import pickle

def load_model_and_tokenizer(model_name, device, quantize_bits=4):
    """
    Load model and tokenizer for given model name, with optional quantization.
    
    Args:
        model_name (str): Model name or path.
        device (int or str): GPU index or 'cpu'.
        quantize_bits (int, optional): If set to 8 or 4, quantize the model accordingly.
        
    Returns:
        model, tokenizer, device
    """
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set quantization config if required
    quantization_config = None
    if quantize_bits == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Applying 8-bit quantization")
    elif quantize_bits == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                                 bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        print("Applying 4-bit quantization")

    # Load the model with or without quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config if quantization_config else None,
        device_map="auto" if quantization_config else None
    )

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    if not quantization_config:
        model = model.to(device)

    print(f"Using device: {device}")
    return model, tokenizer, device

def evaluate_model(df, model, tokenizer, device):
    """Evaluate predictions for entire dataset using specified model"""
    results = []
    print(f"\nEvaluating on dataset with {len(df)} rows...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            result = predict_and_evaluate(row, model, tokenizer, device)
            results.append(result)
        except Exception as e:
            print(f"Error processing row: {row['input']}")
            print(f"Error message: {str(e)}")
            results.append({'generated_text': '', 'is_correct': False})
    
    df_results = df.copy()
    df_results['generated_text'] = [r['generated_text'] for r in results]
    df_results['is_correct'] = [r['is_correct'] for r in results]
    
    accuracy = df_results['is_correct'].mean()
    accuracy_by_position = df_results.groupby('target_position')['is_correct'].mean()
    
    return df_results, accuracy, accuracy_by_position

def predict_and_evaluate(row, model, tokenizer, device):
    """Make prediction for a single row using specified model"""
    input_text = row['input']
    target_completion = row['completion']
    
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    input_length = input_ids.shape[1]
    
    outputs = model.generate(input_ids, max_new_tokens=2, do_sample=False)
    generated_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    is_correct = target_completion.lower() in generated_text.lower()
    
    return {
        'generated_text': generated_text,
        'is_correct': is_correct
    }

def extract_sequence_length(filename):
    """Extract sequence length from filename"""
    # Try to extract a number from the filename
    basename = os.path.basename(filename)
    nums = ''.join(c for c in basename if c.isdigit())
    if nums:
        try:
            return int(nums.split('.')[0])
        except ValueError:
            pass
    return None

def parse_dataset_list(dataset_arg):
    """Parse the dataset argument which can be a list or a pattern"""
    if dataset_arg.startswith('[') and dataset_arg.endswith(']'):
        # It's a list format, parse it
        items = dataset_arg.strip('[]').split(',')
        return [item.strip().strip("'\"") for item in items if item.strip()]
    else:
        # It's a pattern, return it as is
        return dataset_arg

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate language model on datasets with different sequence lengths')
    parser.add_argument('--model', type=str, default='tiiuae/falcon-mamba-7b', 
                        help='Model name or path (default: tiiuae/falcon-mamba-7b)')
    parser.add_argument('--datasets', type=str, required=True,
                        help='List of dataset files in format ["file1.csv", "file2.csv"] or a glob pattern like "dataset/*.csv"')
    parser.add_argument('--labels', type=str,
                        help='Optional list of labels for datasets in format ["Label1", "Label2"]. If not provided, filenames will be used.')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results and pickle file (default: results)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index to use (default: 0)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set CUDA device if specified
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    
    # Parse the datasets argument
    datasets_input = parse_dataset_list(args.datasets)
    
    # Find all dataset files
    if isinstance(datasets_input, list):
        # It's already a list of files
        file_paths = datasets_input
        # Check if files exist
        for file in file_paths:
            if not os.path.exists(file):
                print(f"Warning: File does not exist: {file}")
    else:
        # It's a pattern, use glob to find files
        file_paths = glob.glob(datasets_input)
        if not file_paths:
            print(f"Error: No files found matching pattern: {datasets_input}")
            return
    
    # Sort files by name
    file_paths.sort()
    
    # Parse the labels if provided
    if args.labels:
        labels_input = parse_dataset_list(args.labels)
        if isinstance(labels_input, list):
            dataset_labels = labels_input
            if len(dataset_labels) != len(file_paths):
                print(f"Warning: Number of labels ({len(dataset_labels)}) doesn't match number of datasets ({len(file_paths)})")
                # Use as many labels as possible, fill the rest with filenames
                if len(dataset_labels) < len(file_paths):
                    dataset_labels.extend([os.path.basename(f) for f in file_paths[len(dataset_labels):]])
                else:
                    dataset_labels = dataset_labels[:len(file_paths)]
        else:
            # If it's not a valid list, use filenames as labels
            dataset_labels = [os.path.basename(f) for f in file_paths]
    else:
        # No labels provided, use filenames
        dataset_labels = [os.path.basename(f) for f in file_paths]
    
    # Extract sequence lengths for each file
    sequence_lengths = []
    for file in file_paths:
        length = extract_sequence_length(file)
        if length is not None:
            sequence_lengths.append(length)
        else:
            # If can't extract, assume length 0 (unknown)
            sequence_lengths.append(0)
            print(f"Warning: Could not extract sequence length from {file}, assuming unknown length")
    
    # Create dataset info
    dataset_info = []
    for i, (file, label, length) in enumerate(zip(file_paths, dataset_labels, sequence_lengths)):
        dataset_info.append({
            "file": file,
            "label": label,
            "length": length
        })
    
    print(f"Processing {len(dataset_info)} datasets:")
    for info in dataset_info:
        print(f"  {info['label']} (Length: {info['length']}): {info['file']}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dictionary to store accuracy by position for each dataset
    all_accuracy_data = {
        'dataset_info': dataset_info,
        'accuracy_by_position': []
    }
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model, args.gpu)
    
    # Process each dataset
    for info in dataset_info:
        file_path = info["file"]
        print(f"\nProcessing dataset: {info['label']} ({file_path})...")
        try:
            df = pd.read_csv(file_path)
            df_results, overall_accuracy, accuracy_by_position = evaluate_model(df, model, tokenizer, device)
            
            print(f"Overall accuracy for {info['label']}: {overall_accuracy:.4f}")
            print("Accuracy by position:")
            print(accuracy_by_position)
            
            # Store accuracy by position for this dataset
            all_accuracy_data['accuracy_by_position'].append(accuracy_by_position)
            
            # Save results to CSV for reference
            basename = os.path.basename(file_path)
            filename_without_ext = os.path.splitext(basename)[0]
            results_file = f"{args.output_dir}/results_{filename_without_ext}.csv"
            df_results.to_csv(results_file, index=False)
            print(f"Saved detailed results to {results_file}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            # Add empty series for failed datasets to maintain order
            all_accuracy_data['accuracy_by_position'].append(pd.Series())
    
    # Save the accuracy data to a pickle file
    pickle_file = f"{args.output_dir}/accuracy_by_position.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_accuracy_data, f)
    
    print(f"\nSaved accuracy by position data to {pickle_file}")
    print(f"You can use this pickle file later for plotting.")

if __name__ == "__main__":
    main()