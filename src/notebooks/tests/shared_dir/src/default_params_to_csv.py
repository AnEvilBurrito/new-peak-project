# processes the default parameters (single row csv-dict format) to the standard .csv format for batch processing 
import pandas as pd
import argparse
import os

def load_csv_as_dict(file_path):
    df = pd.read_csv(file_path)  # Expects headers: 'Parameter', 'Value'
    return dict(zip(df['Parameter'], df['Value']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="parameters.csv", help="Base parameters input CSV")
    parser.add_argument("--output", type=str, default="true_parameters.csv", help="Output CSV file")
    args = parser.parse_args()

    data_dir = "src/notebooks/tests/shared_dir/src"
    os.makedirs(data_dir, exist_ok=True)

    base_params = load_csv_as_dict(f"{data_dir}/{args.input}")

    # Create a DataFrame with a single row
    df = pd.DataFrame([base_params])
    df.insert(0, "param_set_id", 0)

    output_path = f"{data_dir}/{args.output}"
    df.to_csv(output_path, index=False)
    print(f"Saved base parameters to '{output_path}'")
