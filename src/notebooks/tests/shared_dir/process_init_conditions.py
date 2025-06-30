import pandas as pd
import argparse
import os


def process_initial_conditions(file_path, output_path, sample_size=-1):
    # Load the CSV file into a DataFrame
    all_initial_conditions = pd.read_csv(file_path)
    all_initial_conditions.rename(columns={all_initial_conditions.columns[0]: 'ID'}, inplace=True)

    if sample_size > 0:
        # Sample a subset of the DataFrame
        all_initial_conditions = all_initial_conditions.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} initial conditions from the dataset.")
    # Save as CSV
    all_initial_conditions.to_csv(output_path, index=False)
    print(f"Processed initial conditions saved to '{output_path}'")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="src/notebooks/tests/median-ccle_protein_expression-fgfr4_model_ccle_match_rules-375x51-initial_conditions.csv.csv",
        help="Path to initial conditions CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="src/notebooks/tests/shared_dir/src/all_initial_conditions.csv",
        help="Output CSV file path"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1,
        help="Number of initial conditions to sample from the dataset. Use -1 for all."
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    process_initial_conditions(args.input, args.output)
