import pandas as pd
import numpy as np
import argparse
import os

def load_csv_as_dict(file_path):
    df = pd.read_csv(file_path)  # Expects headers: 'Parameter', 'Value'
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
    parser.add_argument("--samples", type=int, default=375, help="Number of parameter sets to generate")
    parser.add_argument("--distort_scale", type=float, default=4, help="distortion factor")    
    parser.add_argument("--output", type=str, default=f"modified_parameters_.csv",)
    args = parser.parse_args()
    scale = args.distort_scale
    min_scale = 1 / scale
    max_scale = scale

    data_dir = "src/notebooks/tests/shared_dir/src"
    os.makedirs(data_dir, exist_ok=True)
    base_params = load_csv_as_dict(f"{data_dir}/parameters.csv")
    df = distort_parameters(
        base_params,
        num_samples=args.samples,
        distortion_range=(min_scale, max_scale),
        random_seed=args.seed
    )
    # append distortion scale to args.output
    args.output = args.output.replace(".csv", f"distorted_{scale}.csv")
    output_path = f"{data_dir}/{args.output}"
    df.to_csv(output_path, index=False)
    print(f"Saved {args.samples} distorted parameter sets to '{output_path}'")
