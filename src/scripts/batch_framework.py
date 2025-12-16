"""
Batch Execution Framework for Sequential Re-assembly Pattern

This module provides the core infrastructure for batch execution of dataset generation
with sequential re-assembly capabilities for multi-day processes.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import pickle

# Import project-specific utilities
from models.utils.s3_config_manager import S3ConfigManager
from models.utils.config_manager import save_data, load_data


class BatchExecutionFramework:
    """Main framework for batch execution with sequential re-assembly"""
    
    def __init__(self, notebook_config: Dict[str, Any], storage_backend: str = 's3'):
        """
        Initialize batch execution framework
        
        Args:
            notebook_config: Configuration dictionary following ch5-paper standards  
            storage_backend: 's3' or 'local' for storage backend
        """
        self.notebook_config = notebook_config
        self.storage_backend = storage_backend
        
        # Initialize appropriate storage manager
        if storage_backend == 's3':
            self.storage_manager = S3ConfigManager()
        else:
            self.storage_manager = None  # Use local config_manager
        
        # Assembly tracking - keep in memory for better consistency
        self.assembly_list_file = self._get_assembly_list_filename()
        self.current_assembly_id = None
        self._loaded_assembly_list = None  # Cache for assembly list
        
        # Load assembly list immediately to populate cache
        self._load_initial_assembly_list()
    
    def _load_initial_assembly_list(self):
        """Load assembly list once and cache it for better performance"""
        try:
            assembly_list = self.load_assembly_list_from_storage()
            if assembly_list is not None and not assembly_list.empty:
                self._loaded_assembly_list = assembly_list
            else:
                self._loaded_assembly_list = pd.DataFrame(columns=['assembly_id', 'timestamp', 'status'])
        except:
            self._loaded_assembly_list = pd.DataFrame(columns=['assembly_id', 'timestamp', 'status'])
    
    def load_assembly_list_from_storage(self):
        """Load assembly list directly from storage without caching"""
        try:
            if self.storage_backend == 's3':
                data = self.storage_manager.load_data(
                    self.notebook_config, 'assembly_list', 'csv'
                )
                if data is not None and not data.empty:
                    return data
            else:
                data = load_data(self.notebook_config, 'assembly_list', 'csv')
                if data is not None and not data.empty:
                    return data
        except:
            pass
        return None
    
    def _get_cached_assembly_list(self):
        """Get assembly list from cache, reload if needed"""
        if self._loaded_assembly_list is None:
            self._load_initial_assembly_list()
        return self._loaded_assembly_list
    
    def _update_cached_assembly_list(self, new_list):
        """Update the cached assembly list"""
        self._loaded_assembly_list = new_list
        return self._loaded_assembly_list
        
    def _get_assembly_list_filename(self) -> str:
        """Generate assembly list filename following naming conventions"""
        exp_number = self.notebook_config.get('exp_number', '01')
        version_number = self.notebook_config.get('version_number', 'v1')
        notebook_name = self.notebook_config.get('notebook_name', 'batch-execution')
        
        if self.storage_backend == 's3':
            return f"{exp_number}_{version_number}_{notebook_name}/assembly_list.csv"
        else:
            return f"assembly_list.csv"
    
    def generate_assembly_id(self, config_value: Union[str, float]) -> str:
        """
        Generate assembly ID from config value
        
        Args:
            config_value: The configuration value (e.g., distortion factor, noise level)
            
        Returns:
            Assembly ID string
        """
        assembly_id = str(config_value)
        self.current_assembly_id = assembly_id
        self._register_assembly_id(assembly_id)
        
        return assembly_id
    
    def get_pending_assemblies(self, config_value_list: List[Union[str, float]]) -> List[Union[str, float]]:
        """
        Get config values that haven't been processed
        
        Args:
            config_value_list: List of config values to check
            
        Returns:
            List of config values that are still pending
        """
        try:
            assembly_list = self.load_assembly_list()
            completed_assemblies = assembly_list[assembly_list['status'] == 'completed']['assembly_id'].tolist()
            
            # Convert completed assemblies back to original types for comparison
            completed_values = []
            for assembly_id in completed_assemblies:
                try:
                    # Try to convert back to float if it was a numeric config value
                    completed_values.append(float(assembly_id))
                except ValueError:
                    completed_values.append(assembly_id)
            
            # Find values not in completed list
            pending = []
            for value in config_value_list:
                value_str = str(value)
                if value_str not in completed_assemblies:
                    pending.append(value)
            
            return pending
            
        except Exception as e:
            print(f"Error checking pending assemblies: {e}")
            # If we can't load assembly list, assume all are pending
            return config_value_list
    
    def _register_assembly_id(self, assembly_id: str):
        """Register assembly ID in the assembly list using caching - with duplicate prevention"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Use cached version
            assembly_list = self._get_cached_assembly_list()
            
            # Clean up any existing duplicates first
            assembly_list = self._clean_assembly_list(assembly_list)
            
            # Check if this assembly_id already exists
            mask = assembly_list['assembly_id'] == assembly_id
            
            if mask.any():
                # Update existing record to 'started'
                assembly_list.loc[mask, 'status'] = 'started'
                assembly_list.loc[mask, 'timestamp'] = timestamp
                
                # If there are multiple entries, keep only the most recent one
                if mask.sum() > 1:
                    assembly_list = assembly_list.sort_values('timestamp', ascending=False).drop_duplicates('assembly_id', keep='last')
            else:
                # Append new record correctly using pd.concat()
                assembly_record = {
                    'assembly_id': assembly_id,
                    'timestamp': timestamp,
                    'status': 'started'
                }
                new_record_df = pd.DataFrame([assembly_record])
                assembly_list = pd.concat([assembly_list, new_record_df], ignore_index=True)
            
            # Update cache immediately
            self._update_cached_assembly_list(assembly_list)
            
            # Then save to storage
            self.save_assembly_list(assembly_list)
            
        except Exception as e:
            print(f"Error registering assembly {assembly_id}: {e}")
            # Create new assembly list and update cache
            assembly_record = {
                'assembly_id': assembly_id,
                'timestamp': timestamp,
                'status': 'started'
            }
            assembly_list = pd.DataFrame([assembly_record])
            self._update_cached_assembly_list(assembly_list)
            self.save_assembly_list(assembly_list)
    
    def _clean_assembly_list(self, assembly_list: pd.DataFrame) -> pd.DataFrame:
        """Clean up assembly list by removing duplicates and keeping only latest status"""
        if assembly_list.empty:
            return assembly_list
            
        # Sort by timestamp (most recent first) and keep only the latest entry per assembly_id
        cleaned_list = assembly_list.sort_values('timestamp', ascending=False)
        cleaned_list = cleaned_list.drop_duplicates('assembly_id', keep='first')
        
        return cleaned_list
    
    def cleanup_assembly_list(self) -> bool:
        """Clean up the current assembly list and save it"""
        try:
            assembly_list = self.load_assembly_list()
            cleaned_list = self._clean_assembly_list(assembly_list)
            
            if len(cleaned_list) < len(assembly_list):
                print(f"Cleaned assembly list: removed {len(assembly_list) - len(cleaned_list)} duplicate entries")
                self.save_assembly_list(cleaned_list)
                self._update_cached_assembly_list(cleaned_list)
                return True
            else:
                print("Assembly list is already clean")
                return True
                
        except Exception as e:
            print(f"Error cleaning assembly list: {e}")
            return False

    def mark_assembly_completed(self):
        """Mark current assembly as completed in the assembly list"""
        if self.current_assembly_id is None:
            raise ValueError("No active assembly ID")
        
        try:
            # Use cached version for immediate update
            assembly_list = self._get_cached_assembly_list()
            mask = assembly_list['assembly_id'] == self.current_assembly_id
            if mask.any():
                assembly_list.loc[mask, 'status'] = 'completed'
                # Update cache immediately
                self._update_cached_assembly_list(assembly_list)
                # Then save to storage
                self.save_assembly_list(assembly_list)
                print(f"✅ Assembly {self.current_assembly_id} marked as completed")
            else:
                print(f"Warning: Assembly ID {self.current_assembly_id} not found in assembly list")
                # Try to register it and mark completed
                self._register_assembly_id(self.current_assembly_id)
                assembly_list = self._get_cached_assembly_list()
                mask = assembly_list['assembly_id'] == self.current_assembly_id
                if mask.any():
                    assembly_list.loc[mask, 'status'] = 'completed'
                    self._update_cached_assembly_list(assembly_list)
                    self.save_assembly_list(assembly_list)
                    print(f"✅ Assembly {self.current_assembly_id} registered and completed")
                
        except Exception as e:
            print(f"Error marking assembly as completed: {e}")
            
    def load_assembly_list(self) -> pd.DataFrame:
        """Load assembly list using caching to avoid S3 synchronization issues"""
        # Use cached version if available, otherwise load from storage
        return self._get_cached_assembly_list()
    
    def save_assembly_list(self, assembly_list: pd.DataFrame):
        """Save assembly list to storage"""
        if self.storage_backend == 's3':
            self.storage_manager.save_data(
                self.notebook_config, assembly_list, 'assembly_list', 'csv'
            )
        else:
            save_data(self.notebook_config, assembly_list, 'assembly_list', 'csv')
    
    def sequential_reassembly(self, data_file_common_name: str, 
                             data_path: Optional[str] = None, 
                             auto_clean: bool = True) -> pd.DataFrame:
        """
        Re-assemble multiple data files into a single DataFrame
        
        Args:
            data_file_common_name: Common name pattern for data files
            data_path: Optional path prefix for data files
            auto_clean: Whether to automatically clean assembly list before re-assembly
            
        Returns:
            Combined DataFrame
        """
        # Clean up assembly list if requested (recommended for production)
        if auto_clean:
            self.cleanup_assembly_list()
        
        assembly_list = self.load_assembly_list()
        valid_assemblies = assembly_list[assembly_list['status'] == 'completed']
        
        if valid_assemblies.empty:
            raise ValueError("No completed assemblies found for re-assembly")
        
        # Ensure we have unique assembly IDs (safety check) - sort by most recent timestamp first
        unique_assemblies = valid_assemblies.sort_values('timestamp', ascending=False).drop_duplicates('assembly_id', keep='first')
        if len(unique_assemblies) < len(valid_assemblies):
            print(f"⚠️ Duplicate assembly IDs detected, using {len(unique_assemblies)} unique assemblies (most recent timestamps prioritized)")
            valid_assemblies = unique_assemblies
        
        data_frames = []
        
        for _, assembly in valid_assemblies.iterrows():
            assembly_id = assembly['assembly_id']
            data_filename = f"{data_file_common_name}_{assembly_id}"
            
            try:
                if self.storage_backend == 's3':
                    data = self.storage_manager.load_data(
                        self.notebook_config, data_filename, 'pkl'
                    )
                else:
                    data = load_data(self.notebook_config, data_filename, 'pkl')
                
                if isinstance(data, pd.DataFrame):
                    data_frames.append(data)
                else:
                    print(f"Warning: Data for {assembly_id} is not a DataFrame")
                    
            except Exception as e:
                print(f"Warning: Could not load data for {assembly_id}: {e}")
        
        if not data_frames:
            raise ValueError("No valid data files found for re-assembly")
        
        # Concatenate all data frames
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        print(f"Re-assembled {len(data_frames)} data files into DataFrame with shape {combined_df.shape}")
        return combined_df
    
    def save_batch_data(self, data: Any, data_name: str):
        """Save batch data with assembly ID in filename"""
        if self.current_assembly_id is None:
            raise ValueError("No active assembly ID. Call generate_assembly_id() first.")
        
        full_data_name = f"{data_name}_{self.current_assembly_id}"
        
        if self.storage_backend == 's3':
            self.storage_manager.save_data(
                self.notebook_config, data, full_data_name, 'pkl'
            )
        else:
            save_data(self.notebook_config, data, full_data_name, 'pkl')
    
    def get_assembly_status(self, assembly_id: str) -> str:
        """Get status of a specific assembly"""
        assembly_list = self.load_assembly_list()
        match = assembly_list[assembly_list['assembly_id'] == assembly_id]
        
        if match.empty:
            return 'not_found'
        return match.iloc[0]['status']


# Convenience function for quick batch execution
def create_batch_executor(notebook_name: str, exp_number: str, 
                         version_number: str = 'v1', section_number: str = '4',
                         storage_backend: str = 's3') -> BatchExecutionFramework:
    """
    Convenience function to create a BatchExecutionFramework
    
    Args:
        notebook_name: Name of the notebook/experiment
        exp_number: Experiment number (e.g., '01', '02')
        version_number: Version number (default 'v1')
        section_number: Section number (default '4')
        storage_backend: Storage backend ('s3' or 'local')
    
    Returns:
        BatchExecutionFramework instance
    """
    notebook_config = {
        'notebook_name': notebook_name,
        'exp_number': exp_number,
        'version_number': version_number,
        'section_number': section_number
    }
    
    return BatchExecutionFramework(notebook_config, storage_backend)
