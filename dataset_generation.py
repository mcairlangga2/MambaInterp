# Usage python dataset_generation.py --dataset_type <dataset_type> --n <number of discourse> --num_samples <number of samples for each discourse> --output_dir <output directory>
import os
import argparse
import random
import csv
from transformers import AutoTokenizer

random.seed(42)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('tiiuae/falcon-mamba-7b')

# Extended lists to check
first_subjects = ['Mark', 'John', 'Mary', 'Sarah', 'David', 'James', 'Emma', 'Alex', 'Lisa', 'Mike', 
                'Tom', 'Kate', 'Peter', 'Anna', 'Paul', 'Susan', 'Chris', 'Laura', 'Kevin', 'Amy',
                'Brian', 'Emily', 'Ryan', 'Sophia', 'Jake', 'Olivia', 'Eric', 'Grace', 'Adam', 'Zoe',
                'Daniel', 'Lucy', 'Robert', 'Jane', 'Andrew', 'Linda', 'Thomas', 'Helen', 'Jason', 'Karen',
                'Scott', 'Lily', 'Steven', 'Ella', 'Jeff', 'Diane', 'Tony', 'Maria', 'Frank', 'Ruth',
                'Josh', 'Anne', 'Jack', 'Carol', 'Tim', 'Rose', 'Will', 'Julia', 'Joe', 'Alice',
                'Ben', 'Chloe', 'Sam', 'Megan', 'Max', 'Abby', 'Greg', 'Molly', 'Bob', 'Nina',
                'Bill', 'Eva', 'Luke', 'Tina', 'Nick', 'Claire', 'Matt', 'Gina', 'Pat', 'Ellen',
                'Steve', 'Fiona', 'Alan', 'Wendy', 'Dave', 'Donna', 'George', 'Holly', 'Rick', 'Katie']

relations = [' loves', ' hates', ' knows', ' meets', ' helps', ' calls', ' sees', ' likes', ' trusts', ' follows',
           ' needs', ' finds', ' wants', ' asks', ' tells', ' texts', ' visits', ' pays', ' thanks', ' greets',
           ' fears', ' joins', ' hears', ' avoids', ' fights', ' hugs', ' misses', ' stops', ' serves', ' blocks',
           ' leaves', ' seeks', ' takes', ' brings', ' shows', ' teaches', ' leads', ' drives', ' sends', ' writes',
           ' meets', ' saves', ' treats', ' serves', ' joins', ' warns', ' moves', ' guards', ' seeks', ' cooks',
           ' hires', ' trains', ' holds', ' reads', ' feeds', ' grows', ' draws', ' wakes', ' lifts', ' keeps',
           ' risks', ' parks', ' scores', ' grabs', ' heals', ' judges', ' faces', ' opens', ' stops', ' builds',
           ' cleans', ' stays', ' gives', ' feels', ' makes', ' fixes', ' checks', ' burns', ' hides', ' shares']

names = [' Mark', ' John', ' Mary', ' Sarah', ' David', ' James', ' Emma', ' Alex', ' Lisa', ' Mike',
       ' Tom', ' Kate', ' Peter', ' Anna', ' Paul', ' Susan', ' Chris', ' Laura', ' Kevin', ' Amy',
       ' Brian', ' Emily', ' Ryan', ' Sophia', ' Jake', ' Olivia', ' Eric', ' Grace', ' Adam', ' Zoe',
       ' Daniel', ' Lucy', ' Robert', ' Jane', ' Andrew', ' Linda', ' Thomas', ' Helen', ' Jason', ' Karen',
       ' Scott', ' Lily', ' Steven', ' Ella', ' Jeff', ' Diane', ' Tony', ' Maria', ' Frank', ' Ruth',
       ' Josh', ' Anne', ' Jack', ' Carol', ' Tim', ' Rose', ' Will', ' Julia', ' Joe', ' Alice',
       ' Ben', ' Chloe', ' Sam', ' Megan', ' Max', ' Abby', ' Greg', ' Molly', ' Bob', ' Nina',
       ' Bill', ' Eva', ' Luke', ' Tina', ' Nick', ' Claire', ' Matt', ' Gina', ' Pat', ' Ellen',
       ' Steve', ' Fiona', ' Alan', ' Wendy', ' Dave', ' Donna', ' George', ' Holly', ' Rick', ' Katie']

def filter_single_token_items(items_list):
    """Filter a list to only include items that tokenize as a single token."""
    filtered_items = []
    for item in items_list:
        tokens = tokenizer.tokenize(item)
        if len(tokens) == 1:
            filtered_items.append(item)
    return filtered_items

# Filter lists to keep only single-token items
filtered_first_subjects = filter_single_token_items(first_subjects)[:20]
filtered_relations = filter_single_token_items(relations)[:20]
filtered_names = filter_single_token_items(names)[20:40]

def create_synthetic_dataset(n, num_samples):
    """Create a synthetic dataset with n triples and num_samples examples."""
    dataset = []
    
    for position in range(n):
        position_queries = []
        
        for _ in range(num_samples):
            # Shuffle the lists to get random combinations
            random.shuffle(filtered_first_subjects)
            random.shuffle(filtered_relations)
            random.shuffle(filtered_names)
            
            # Create n unique triples for the context
            context_triples = []
            used_combinations = set()
            
            for i in range(n):
                # Find unique combinations
                while True:
                    subj = filtered_first_subjects[i % len(filtered_first_subjects)]
                    rel = filtered_relations[i % len(filtered_relations)]
                    name = filtered_names[i % len(filtered_names)]
                    
                    triple_key = (subj, rel, name)
                    if triple_key not in used_combinations:
                        used_combinations.add(triple_key)
                        context_triples.append(f"{subj}{rel}{name}")
                        break
                    
                    # Shift to find a new combination
                    i += 1
            
            # Create the query with the appropriate completion
            context = ". ".join(context_triples)
            
            # For position 0, the next subject and relation should be the same as the first triple
            if position == 0:
                query = f"{context}. {filtered_first_subjects[0]}{filtered_relations[0]}"
            else:
                # For positions 1-7, the next subject and relation match the respective position
                query = f"{context}. {filtered_first_subjects[position]}{filtered_relations[position]}"
            
            position_queries.append(query)
        
        dataset.append(position_queries)
    
    return dataset

def create_csv_from_dataset(dataset, filename="synthetic_data.csv"):
    """Generate a CSV file from the dataset."""
    try:
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['input', 'target_position', 'completion'])
            
            for position, queries in enumerate(dataset):
                for query in queries:
                    # Extract all the context before the final subject+relation
                    last_dot_index = query.rindex('.')
                    full_context = query[:last_dot_index+1].strip()
                    partial_completion = query[last_dot_index+1:].strip()
                    
                    # Parse triples from the context
                    triples = full_context.split('.')
                    triples = [t.strip() for t in triples if t.strip()]
                    
                    # Get the selected triple based on position
                    if position < len(triples):
                        selected_triple = triples[position]
                        
                        # Split the triple into subject, relation, and object
                        words = selected_triple.split(' ')
                        
                        # First word is the subject
                        subject = words[0]
                        
                        # Second word is the relation (with space)
                        relation = ' ' + words[1] if len(words) > 1 else ''
                        
                        # The rest is the object
                        obj = ' '.join(words[2:]) if len(words) > 2 else ''
                        
                        # Input should include subject and relation of the position's triple
                        input_text = full_context + ' ' + subject + relation
                        completion = obj
                    else:
                        input_text = full_context
                        completion = ""
                    
                    csvwriter.writerow([input_text, position+1, completion])
        
        print(f"CSV file '{filename}' created successfully.")
        return True
    except Exception as e:
        print(f"Error creating CSV file: {e}")
        return False

def ensure_directory_exists(directory_path):
    """Ensure that the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
        except Exception as e:
            print(f"Error creating directory {directory_path}: {e}")
            return False
    return True

def process_dataset(n, num_samples, output_dir):
    """Process the dataset and save to the specified output directory."""
    # Ensure output directory exists
    if not ensure_directory_exists(output_dir):
        print(f"Failed to create or access output directory: {output_dir}")
        return False
    
    # Generate dataset
    try:
        synthetic_dataset = create_synthetic_dataset(n, num_samples)
        output_file = os.path.join(output_dir, f'{n}.csv')
        if create_csv_from_dataset(synthetic_dataset, filename=output_file):
            print(f"Dataset processed successfully and saved to {output_file}")
            return True
        else:
            print(f"Failed to create CSV file at {output_file}")
            return False
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return False

def main():
    """Main entry point for the script."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate synthetic datasets for discourse modeling.')
    parser.add_argument('--n', type=int, required=True, help='Number of discourse triples')
    parser.add_argument('--num_samples', type=int, required=True, help='Number of samples for each discourse position')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for CSV files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.n <= 0:
        print("Error: Number of discourse triples (--n) must be positive")
        return
    
    if args.num_samples <= 0:
        print("Error: Number of samples (--num_samples) must be positive")
        return
    
    if not args.output_dir:
        print("Error: Output directory (--output_dir) must be specified")
        return
    
    # Process dataset
    process_dataset(args.n, args.num_samples, args.output_dir)

if __name__ == "__main__":
    main()