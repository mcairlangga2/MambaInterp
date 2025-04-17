#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import seaborn as sns
import torch
import os
import argparse
import glob
import matplotlib.cm as cm

def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer for given model name"""
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Get the device
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    return model.to(device), tokenizer, device

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

def plot_results(all_accuracy_by_position, all_overall_accuracy, dataset_info, output_dir, plot_name="accuracy_results.png"):
    """Plot position-based accuracy and overall accuracy for different datasets"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get the dataset labels and lengths
    dataset_labels = [info["label"] for info in dataset_info]
    sequence_lengths = [info["length"] for info in dataset_info]
    
    # Create color map based on number of datasets
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i/len(dataset_info)) for i in range(len(dataset_info))]
    
    # Plot 1: Line chart of accuracy by position for each dataset
    for i, (info, color) in enumerate(zip(dataset_info, colors)):
        if i >= len(all_accuracy_by_position):
            continue
            
        positions = list(all_accuracy_by_position[i].index)
        accuracies = list(all_accuracy_by_position[i].values)
        
        # Only plot positions that exist in the dataset
        valid_positions = [pos for pos in positions if pos <= info["length"]]
        valid_accuracies = [acc for pos, acc in zip(positions, accuracies) if pos <= info["length"]]
        
        ax1.plot(valid_positions, valid_accuracies, marker='o', label=info["label"], color=color)
    
    ax1.set_xlabel('Position in Sequence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Position for Different Datasets')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Bar chart of overall accuracy for each dataset
    x_pos = np.arange(len(dataset_labels))
    ax2.bar(x_pos, all_overall_accuracy, color=colors)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Overall Accuracy')
    ax2.set_title('Overall Accuracy by Dataset')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(dataset_labels, rotation=45, ha='right')
    
    # Add text labels on bars
    for i, acc in enumerate(all_overall_accuracy):
        ax2.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Save the plot
    output_path = os.path.join(output_dir, plot_name)
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to '{output_path}'")
    
    return fig

def plot_only_position_accuracy(all_accuracy_by_position, dataset_info, output_dir, plot_name="position_accuracy.png"):
    """Plot only the position-based accuracy chart with larger size"""
    plt.figure(figsize=(12, 8))
    
    # Get dataset labels
    dataset_labels = [info["label"] for info in dataset_info]
    
    # Create color map based on number of datasets
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i/len(dataset_info)) for i in range(len(dataset_info))]
    
    for i, (info, color) in enumerate(zip(dataset_info, colors)):
        if i >= len(all_accuracy_by_position) or all_accuracy_by_position[i].empty:
            continue
            
        positions = list(all_accuracy_by_position[i].index)
        accuracies = list(all_accuracy_by_position[i].values)
        
        # Only plot positions that exist in the dataset
        valid_positions = [pos for pos in positions if pos <= info["length"]]
        valid_accuracies = [acc for pos, acc in zip(positions, accuracies) if pos <= info["length"]]
        
        plt.plot(valid_positions, valid_accuracies, marker='o', 
                 label=info["label"], color=color, linewidth=2)
    
    plt.xlabel('Position in Sequence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy by Position for Different Datasets', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.ylim(0, 1.05)
    
    # Set x-ticks to be integers
    max_position = max([max(acc.index) for acc in all_accuracy_by_position if not acc.empty], default=0)
    if max_position > 0:
        plt.xticks(np.arange(1, max_position + 1))
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, plot_name)
    plt.savefig(output_path, dpi=300)
    print(f"Saved position accuracy plot to '{output_path}'")

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
                        help='Directory to save results and plots (default: results)')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only plot results from existing result files without running model evaluation')
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
    
    all_overall_accuracy = []
    all_accuracy_by_position = []
    
    if args.plot_only:
        # Just load results from existing files
        for info in dataset_info:
            # Try to find results file
            basename = os.path.basename(info["file"])
            filename_without_ext = os.path.splitext(basename)[0]
            results_file = f"{args.output_dir}/results_{filename_without_ext}.csv"
            
            if not os.path.exists(results_file):
                # Try with just the length
                results_file = f"{args.output_dir}/results_{info['length']}.csv"
            
            if os.path.exists(results_file):
                try:
                    df_results = pd.read_csv(results_file)
                    overall_accuracy = df_results['is_correct'].mean()
                    accuracy_by_position = df_results.groupby('target_position')['is_correct'].mean()
                    
                    all_overall_accuracy.append(overall_accuracy)
                    all_accuracy_by_position.append(accuracy_by_position)
                    
                    print(f"Loaded results for {info['label']}: accuracy = {overall_accuracy:.4f}")
                except Exception as e:
                    print(f"Error loading results for {info['label']}: {str(e)}")
                    all_overall_accuracy.append(0)
                    all_accuracy_by_position.append(pd.Series())
            else:
                print(f"Warning: Results file not found for {info['label']}: {results_file}")
                all_overall_accuracy.append(0)
                all_accuracy_by_position.append(pd.Series())
    else:
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
                
                all_overall_accuracy.append(overall_accuracy)
                all_accuracy_by_position.append(accuracy_by_position)
                
                # Save results to CSV
                basename = os.path.basename(file_path)
                filename_without_ext = os.path.splitext(basename)[0]
                results_file = f"{args.output_dir}/results_{filename_without_ext}.csv"
                df_results.to_csv(results_file, index=False)
                print(f"Saved detailed results to {results_file}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                all_overall_accuracy.append(0)
                all_accuracy_by_position.append(pd.Series())
    
    # Plot the results
    if all_overall_accuracy:
        plot_results(all_accuracy_by_position, all_overall_accuracy, dataset_info, args.output_dir)
        plot_only_position_accuracy(all_accuracy_by_position, dataset_info, args.output_dir)
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()