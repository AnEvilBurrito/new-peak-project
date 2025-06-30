import pandas as pd
import argparse
import os
import pickle

def load_feature_data(file_path, sample_size=-1):
    """
    Load feature matrix with cell names as index and optionally sample rows.
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(file_path, index_col=0)  # Treat first column as index (cell names)
    
    if sample_size > 0:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows from feature data.")

    print(f"Loaded feature data with shape: {df.shape}")
    return df

def save_pickle(obj, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved feature data to pickle file: '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="src/notebooks/tests/median-ccle_protein_expression-fgfr4_model_ccle_match_rules-375x51-initial_conditions.csv.csv",
        help="Path to initial conditions CSV file"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=-1,
        help="Number of rows to sample from the dataset. Use -1 for all."
    )
    parser.add_argument(
        "--save_pickle",
        type=str,
        default="src/notebooks/tests/shared_dir/src/feature_data.pkl",
        help="Optional path to save the feature_data object as a .pkl file"
    )
    args = parser.parse_args()

    feature_data = load_feature_data(args.input, sample_size=args.sample_size)

    if args.save_pickle:
        os.makedirs(os.path.dirname(args.save_pickle), exist_ok=True)
        save_pickle(feature_data, args.save_pickle)
