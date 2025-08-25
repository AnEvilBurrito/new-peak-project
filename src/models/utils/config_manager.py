# src/utils/config_manager.py

import os
import yaml
import dotenv

dotenv.load_dotenv()
new_path = os.getenv("NEW_DATA_PATH")

def load_configs(folder_name: str, config_suffix: str = "v1") -> dict:
    config_path = os.path.join(new_path, folder_name, f"config_{config_suffix}.yml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)  # Use safe_load to prevent code execution
    return config

def save_configs(folder_name: str, config: dict, config_suffix: str = "v1") -> None:
    config_path = os.path.join(new_path, folder_name, f"config_{config_suffix}.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
def initialise_config(folder_name: str, verbose: int = 0) -> None:
    '''
    Create a folder and set-up the initial experimental structure
    '''
    folder_path = os.path.join(new_path, folder_name)
    # check if folder exists, if not create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        # make 'data' and 'figures' subfolders
        os.makedirs(os.path.join(folder_path, 'data'), exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'figures'), exist_ok=True)
    if verbose > 0:
        print(f"Created folder structure at {folder_path}")
    else: 
        if verbose > 0:
            print(f"Folder {folder_path} already exists. No changes made.")
        
    