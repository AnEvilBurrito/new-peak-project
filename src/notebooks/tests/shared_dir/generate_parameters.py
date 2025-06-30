import pandas as pd
import numpy as np
import argparse

def load_csv_as_dict(file_path):
    df = pd.read_csv(file_path)  # has headers: 'Parameter', 'Value'
    return dict(zip(df['Parameter'], df['Value']))

def distort_parameters(base_params, num_samples=1000, distortion_range=(0.25, 4.0), random_seed=42):
    rng = np.random.default_rng(random_seed)
    keys = list(base_params.keys())

    modified_parameter_sets = []
    for i in range(num_samples):
        new_params = {
            "param_set_id": i,
            **{k: base_params[k] * rng.uniform(*distortion_range) for k in keys}
        }
        modified_parameter_sets.append(new_params)

    return pd.DataFrame(modified_parameter_sets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--samples", type=int, default=1000, help="Number of parameter sets to generate")
    parser.add_argument("--min_distort", type=float, default=0.25, help="Minimum distortion factor")
    parser.add_argument("--max_distort", type=float, default=4.0, help="Maximum distortion factor")
    args = parser.parse_args()

    data_dir = "src/notebooks/tests/shared_dir/src"
    base_params = load_csv_as_dict(f"{data_dir}/parameters.csv")
    df = distort_parameters(
        base_params,
        num_samples=args.samples,
        distortion_range=(args.min_distort, args.max_distort),
        random_seed=args.seed
    )
    df.to_hdf(f"{data_dir}/modified_parameters.h5", key="params", mode="w", format="table", index=False)
    print(f"Saved {args.samples} distorted parameter sets to 'modified_parameters.h5'")
