#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from pathlib import Path
import re


def process_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    processed_rows = []

    for _, row in df.iterrows():
        input_text = row['input']
        target_position = row['target_position']
        completion = row['completion']

        # Match all sentences ending with a period
        sentences = re.findall(r'[^.]+?\.', input_text)

        # Handle any trailing incomplete sentence (doesn't end in a period)
        incomplete_part = input_text[len("".join(sentences)):].strip()

        # Extract verb from first complete sentence
        first_sentence = sentences[0].strip()
        first_parts = first_sentence.split()
        if len(first_parts) >= 2:
            target_verb = first_parts[1]
        else:
            processed_rows.append(row)
            continue

        # Replace verbs in complete sentences
        new_sentences = []
        for sentence in sentences:
            parts = sentence.strip().split()
            if len(parts) == 3:  # assume "Name verb Name."
                parts[1] = target_verb
                new_sentence = " ".join(parts) # add period back
                new_sentences.append(new_sentence)
            else:
                new_sentences.append(sentence.strip())

        # Handle the last incomplete sentence
        if incomplete_part:
            incomplete_parts = incomplete_part.strip().split()
            if len(incomplete_parts) == 2:  # "Name verb"
                incomplete_parts[1] = target_verb
                incomplete_part = " ".join(incomplete_parts)
            elif len(incomplete_parts) == 3:  # "Name verb Name" form
                incomplete_parts[1] = target_verb
                incomplete_part = " ".join(incomplete_parts)

        # Reconstruct full text
        new_input_text = " ".join(new_sentences)
        if incomplete_part:
            new_input_text = new_input_text + " " + incomplete_part

        # Append processed row
        processed_rows.append({
            'input': new_input_text,
            'target_position': target_position,
            'completion': completion
        })

    # Create and preview new DataFrame
    processed_df = pd.DataFrame(processed_rows)
    processed_df.to_csv(output_file, index=False)
    print(f"Processed {len(processed_df)} rows. Saved to {output_file}")


def main():
    """Main entry point for the script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process CSV files from input directory to output directory.')
    parser.add_argument('--input_dir', help='Input CSV file path')
    parser.add_argument('--output_dir', help='Output CSV file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the file
    if not args.input_dir or not args.output_dir:
        print("Error: Both input and output file paths are required")
        return
    
    # Process the CSV file
    process_csv(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()