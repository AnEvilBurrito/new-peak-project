# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse
import os

def load_csv_as_dict(file_path):
    df = pd.read_csv(file_path)  # Expects headers: 'Parameter', 'Value'
    return dict(zip(df['Parameter'], df['Value']))

def apply_gaussian_distortion(original_params, distortion_factor, seed=42):
    """
    Apply multiplicative Gaussian noise to parameters: p' = p * (1 + epsilon), epsilon ~ N(0, distortion_factor)
    
    Args:
        original_params: Dictionary of original parameters
        distortion_factor: Standard deviation of relative error (e.g., 0.1 = 10%)
        seed: Random seed for reproducible distortion
    
    Returns:
        Dictionary of distorted parameters (clipped at 1e-8 to ensure positivity)
    """
    if distortion_factor == 0:
        return original_params  # No distortion for baseline
    
    rng = np.random.default_rng(seed)
    distorted_params = {}
    
    for key, value in original_params.items():
        # Generate relative noise: epsilon ~ N(0, distortion_factor)
        relative_noise = rng.normal(loc=0, scale=distortion_factor)
        
        # Apply multiplicative noise: p' = p * (1 + epsilon)
        distorted_value = value * (1 + relative_noise)
        
        # Ensure non-negative (parameters must be positive)
        distorted_params[key] = max(distorted_value, 1e-8)
    
    return distorted_params

def distort_parameters(base_params, num_samples=1000, distortion_factor=0.1, random_seed=42):
    """
    Generate distorted parameter sets using Gaussian multiplicative noise.
    
    Args:
        base_params: Dictionary of base parameters
        num_samples: Number of parameter sets to generate
        distortion_factor: Standard deviation of relative error (e.g., 0.1 = 10%)
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame of distorted parameter sets with 'param_set_id' column
    """
    rng = np.random.default_rng(random_seed)

    modified_parameter_sets = []
    for i in range(num_samples):
        # Use different seed for each sample for variation
        sample_seed = rng.integers(0, 2**32)
        new_params = apply_gaussian_distortion(base_params, distortion_factor, sample_seed)
        new_params["param_set_id"] = i
        modified_parameter_sets.append(new_params)

    return pd.DataFrame(modified_parameter_sets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--samples", type=int, default=375, help="Number of parameter sets to generate")
    parser.add_argument("--distortion_factors", type=float, nargs='+', help="List of distortion factors (standard deviations) for batch processing, e.g., 0.1 0.2 0.3")
    parser.add_argument("--distortion_factor", type=float, default=None, help="Single distortion factor (standard deviation) for Gaussian multiplicative noise (e.g., 0.1 = 10% noise)")
    parser.add_argument("--distort_scale", type=float, default=None, help="Alias for --distortion_factor (backward compatibility)")
    parser.add_argument("--output", type=str, default="modified_parameters_.csv")
    args = parser.parse_args()
    
    # Determine which argument to use
    # Priority: --distortion_factors > --distortion_factor > --distort_scale > default (0.1)
    if args.distortion_factors is not None:
        distortion_factors = args.distortion_factors
        mode = "batch"
        print(f"Batch processing {len(distortion_factors)} distortion factors: {distortion_factors}")
    elif args.distortion_factor is not None:
        distortion_factors = [args.distortion_factor]
        mode = "single"
        using_arg = "--distortion_factor"
    elif args.distort_scale is not None:
        distortion_factors = [args.distort_scale]
        mode = "single"
        using_arg = "--distort_scale"
        print(f"Note: Using --distort_scale {args.distort_scale} as distortion_factor (std dev). Consider using --distortion_factor for clarity.")
    else:
        distortion_factors = [0.1]  # Default
        mode = "default"
        using_arg = "default"
    
    data_dir = "src/notebooks/tests/shared_dir/src"
    os.makedirs(data_dir, exist_ok=True)
    base_params = load_csv_as_dict(f"{data_dir}/parameters.csv")
    
    for distortion_factor in distortion_factors:
        # Warning for large distortion factors
        if distortion_factor > 1.0:
            print(f"Warning: distortion_factor={distortion_factor} is large (>1.0 = 100% std dev). Most parameters will change by more than 100%.")
        
        df = distort_parameters(
            base_params,
            num_samples=args.samples,
            distortion_factor=distortion_factor,
            random_seed=args.seed
        )
        
        # Generate output filename with distortion factor
        # Use the actual distortion_factor value for the filename
        output_base = args.output.replace(".csv", "")
        # Replace dots with underscores to avoid issues with decimal points in filenames
        distortion_str = f"{distortion_factor:.3f}".replace(".", "_")
        output_filename = f"{output_base}distorted_{distortion_str}.csv"
        output_path = f"{data_dir}/{output_filename}"
        df.to_csv(output_path, index=False)
        print(f"Saved {args.samples} distorted parameter sets to '{output_path}'")
        print(f"Using Gaussian multiplicative noise with distortion factor (std dev) = {distortion_factor}")
        print("-" * 50)
    
    if mode == "batch":
        print(f" Batch processing complete. Generated {len(distortion_factors)} parameter sets.")