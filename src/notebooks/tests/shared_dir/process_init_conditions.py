# process a csv file with initial conditions into hdf5 format using pandas

import pandas as pd
import numpy as np


def process_initial_conditions(file_path):
    # Load the CSV file into a DataFrame
    all_initial_conditions = pd.read_csv('notebooks/tests/median-ccle_protein_expression-fgfr4_model_ccle_match_rules-375x51-initial_conditions.csv.csv')
    
    
    
    # # Save the DataFrame to an HDF5 file
    # hdf5_file_path = f"{data_dir}/initial_conditions.h5"
    # df.to_hdf(hdf5_file_path, key='data', mode='w')
    
    # print(f"Processed initial conditions saved to '{hdf5_file_path}'")