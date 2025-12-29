import os
import yaml
import pickle
import pandas as pd
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import dotenv_values
import io
import tqdm


class S3ConfigManager:
    """
    Configuration manager for S3 storage operations.
    
    This class provides methods to save, load, and manage experiment configurations,
    data, and figures in S3-compatible storage.
    """
    
    def __init__(self, env_config=None):
        """
        Initialize the S3 client with environment variables.
        
        Args:
            env_config: Optional dictionary with S3 configuration. 
                       If None, uses environment variables from os.getenv().
        """
        # Get environment variables from system environment
        self.endpoint_url = self._get_env_var("S3_ENDPOINT", env_config)
        self.bucket_name = self._get_env_var("S3_BUCKET_NAME", env_config)
        self.access_key = self._get_env_var("S3_ACCESS_KEY", env_config)
        self.secret_key = self._get_env_var("S3_SECRET_KEY", env_config)
        self.region_name = self._get_env_var("S3_REGION_NAME", env_config, "us-east-1")
        self.save_result_path = self._get_env_var("SAVE_RESULT_PATH", env_config, "new-peak-project/experiments/ch5-paper")
        
        # Validate required environment variables
        self._validate_environment_variables()
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region_name
        )
        
        # Test connection
        self._test_connection()
    
    def _resolve_project_path(self):
        """
        Resolve the project path using the same pattern as working notebooks.
        This ensures the .env file is found in the correct location.
        """
        path = os.getcwd()
        index_project = path.find('project')
        if index_project == -1:
            # Try to find the project from current working directory
            # Look for paths that might contain the project
            if 'new-peak-project' in path:
                index_project = path.find('new-peak-project')
                if index_project != -1:
                    project_path = path[:index_project + len('new-peak-project')]
                    return os.path.join(project_path, 'src')
            raise RuntimeError("Could not resolve project path. Ensure you're running from within the project directory.")
        
        project_path = path[:index_project + 7]
        src_path = os.path.join(project_path, 'src')
        return src_path
    
    def _get_env_var(self, key, env_config=None, default=None):
        """
        Get environment variable with proper fallback logic.
        
        Args:
            key: Environment variable name
            env_config: Optional dictionary with configuration
            default: Default value if variable not found
            
        Returns:
            Environment variable value or default
        """
        # First try provided configuration dictionary
        if env_config and key in env_config:
            value = env_config[key]
            if value is not None:
                return value
        
        # Try dotenv file from resolved project path
        try:
            project_src_path = self._resolve_project_path()
            env_file_path = os.path.join(project_src_path, '.env')
            if os.path.exists(env_file_path):
                env_config = dotenv_values(env_file_path)
                if env_config and key in env_config:
                    value = env_config[key]
                    if value is not None:
                        return value
        except Exception:
            # If path resolution fails, continue to system environment
            pass
        
        # Fall back to system environment variables
        value = os.getenv(key, default)
        return value
    
    def _validate_environment_variables(self):
        """
        Validate that required environment variables are present.
        """
        required_vars = {
            'S3_ENDPOINT': 'endpoint_url',
            'S3_BUCKET_NAME': 'bucket_name', 
            'S3_ACCESS_KEY': 'access_key',
            'S3_SECRET_KEY': 'secret_key'
        }
        missing_vars = []
        
        for env_var, attr_name in required_vars.items():
            value = getattr(self, attr_name)
            if value is None:
                missing_vars.append(env_var)
        
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                f"Please ensure these are set in your environment or .env file"
            )
    
    def _test_connection(self):
        """Test S3 connection and bucket access."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✅ S3 connection successful. Bucket: {self.bucket_name}")
        except (ClientError, NoCredentialsError) as e:
            raise ConnectionError(f"S3 connection failed: {e}")
    
    def _get_s3_key(self, notebook_config, subfolder=None, filename=None):
        """
        Generate S3 key path following the naming convention.
        
        Args:
            notebook_config: Dictionary with notebook configuration
            subfolder: Optional subfolder (e.g., 'data', 'figures')
            filename: Optional filename to append
            
        Returns:
            S3 key path string
        """
        section_number = notebook_config.get('section_number', '00')
        exp_number = notebook_config.get('exp_number')
        version_number = notebook_config.get('version_number', 'v1')
        notebook_name = notebook_config.get('notebook_name')
        
        base_path = f"{self.save_result_path}/{section_number}_{exp_number}_{version_number}_{notebook_name}"
        
        if subfolder:
            base_path = f"{base_path}/{subfolder}"
        
        if filename:
            base_path = f"{base_path}/{filename}"
            
        return base_path
    
    def _upload_with_progress(self, body, key, content_type=None):
        """
        Upload file to S3 with progress indicator.
        
        Args:
            body: File content (bytes, string, or file-like object)
            key: S3 key path
            content_type: Optional content type for metadata
        """
        # Convert to bytes if needed
        if isinstance(body, str):
            body = body.encode('utf-8')
        elif hasattr(body, 'getvalue'):
            body = body.getvalue()
            
        file_size = len(body)
        
        # Create progress bar
        with tqdm.tqdm(
            total=file_size, 
            unit='B', 
            unit_scale=True, 
            desc=f"Uploading {key.split('/')[-1]}"
        ) as pbar:
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=body,
                ContentType=content_type
            )
            pbar.update(file_size)
    
    def _download_with_progress(self, key):
        """
        Download file from S3 with progress indicator.
        
        Args:
            key: S3 key path
            
        Returns:
            File content as bytes
        """
        # Get file size for progress tracking
        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            file_size = response['ContentLength']
        except ClientError:
            raise FileNotFoundError(f"Key not found: {key}")
        
        # Download with progress
        with tqdm.tqdm(
            total=file_size, 
            unit='B', 
            unit_scale=True, 
            desc=f"Downloading {key.split('/')[-1]}"
        ) as pbar:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            
            # Read in chunks and update progress
            body = b''
            for chunk in response['Body'].iter_chunks(chunk_size=8192):
                body += chunk
                pbar.update(len(chunk))
                
        return body
    
    def save_config(self, notebook_config, config_data, config_suffix='v1'):
        """
        Save configuration dictionary as YAML to S3.
        
        Args:
            notebook_config: Dictionary with notebook configuration
            config_data: Configuration data to save
            config_suffix: Configuration version suffix (default: 'v1')
        """
        config_filename = f"{config_suffix}_config.yml"
        key = self._get_s3_key(notebook_config, filename=config_filename)
        
        # Convert config to YAML string
        config_yaml = yaml.dump(config_data, sort_keys=False)
        
        # Upload to S3
        self._upload_with_progress(config_yaml, key, content_type='application/x-yaml')
    
    def load_config(self, notebook_config, config_suffix='v1'):
        """
        Load configuration from YAML file in S3.
        
        Args:
            notebook_config: Dictionary with notebook configuration
            config_suffix: Configuration version suffix (default: 'v1')
            
        Returns:
            Configuration dictionary
        """
        config_filename = f"{config_suffix}_config.yml"
        key = self._get_s3_key(notebook_config, filename=config_filename)
        
        # Download from S3
        config_content = self._download_with_progress(key)
        
        # Parse YAML
        return yaml.safe_load(config_content.decode('utf-8'))
    
    def save_data(self, notebook_config, data, data_name, data_format='pkl', **kwargs):
        """
        Save data to S3 in specified format.
        
        Args:
            notebook_config: Dictionary with notebook configuration
            data: Data to save (any serializable object)
            data_name: Name for the data file
            data_format: Format ('pkl', 'csv', 'txt')
            **kwargs: Additional arguments for serialization
        """
        version_number = notebook_config.get('version_number', 'v1')
        filename = f"{version_number}_{data_name}.{data_format}"
        key = self._get_s3_key(notebook_config, subfolder='data', filename=filename)
        
        # Serialize data based on format
        if data_format == 'pkl':
            body = io.BytesIO()
            pickle.dump(data, body, **kwargs)
            content_type = 'application/octet-stream'
            
        elif data_format == 'csv':
            if hasattr(data, 'to_csv'):
                body = io.StringIO()
                # Ensure index is not saved to avoid column mismatch
                kwargs.setdefault('index', False)
                data.to_csv(body, **kwargs)
                content_type = 'text/csv'
            else:
                raise ValueError("Data does not have a 'to_csv' method")
                
        elif data_format == 'parquet':
            if hasattr(data, 'to_parquet'):
                body = io.BytesIO()
                # Ensure index is not saved for parquet as well
                kwargs.setdefault('index', False)
                data.to_parquet(body, **kwargs)
                content_type = 'application/octet-stream'
            else:
                raise ValueError("Data does not have a 'to_parquet' method")
                
        elif data_format == 'txt':
            body = str(data)
            content_type = 'text/plain'
            
        else:
            raise ValueError("Unsupported data format. Use 'pkl', 'csv', 'parquet', or 'txt'")
        
        # Upload to S3
        self._upload_with_progress(body, key, content_type=content_type)
    
    def load_data(self, notebook_config, data_name, data_format='pkl', **kwargs):
        """
        Load data from S3 in specified format.

        Args:
            notebook_config: Dictionary with notebook configuration
            data_name: Name of the data file
            data_format: Format ('pkl', 'csv', 'parquet')
            **kwargs: Additional arguments for deserialization

        Returns:
            Loaded data object
        """
        version_number = notebook_config.get('version_number', 'v1')
        filename = f"{version_number}_{data_name}.{data_format}"
        key = self._get_s3_key(notebook_config, subfolder='data', filename=filename)

        # Download from S3
        data_content = self._download_with_progress(key)

        # Deserialize based on format
        if data_format == 'pkl':
            return pickle.loads(data_content, **kwargs)

        elif data_format == 'csv':
            # Ensure consistent dtype inference for integers
            kwargs.setdefault('dtype', None)  # Let pandas infer properly
            return pd.read_csv(io.BytesIO(data_content), **kwargs)

        elif data_format == 'parquet':
            return pd.read_parquet(io.BytesIO(data_content), **kwargs)

        else:
            raise ValueError("Unsupported data format. Use 'pkl', 'csv', or 'parquet'")
    
    def load_data_from_path(self, s3_key, data_format='pkl', **kwargs):
        """
        Load data from S3 using a direct key path.

        Args:
            s3_key: Full S3 key path (e.g., 'new-peak-project/experiments/ch5-paper/00_99_v1_test-experiment/data/v1_test_data.pkl')
            data_format: Format ('pkl', 'csv', 'parquet', 'txt')
            **kwargs: Additional arguments for deserialization

        Returns:
            Loaded data object (pickle object, pandas DataFrame, or string for txt)

        Raises:
            FileNotFoundError: If the S3 key does not exist
            ValueError: If data_format is unsupported
        """
        # Download from S3
        data_content = self._download_with_progress(s3_key)

        # Deserialize based on format
        if data_format == 'pkl':
            return pickle.loads(data_content, **kwargs)

        elif data_format == 'csv':
            # Ensure consistent dtype inference for integers
            kwargs.setdefault('dtype', None)  # Let pandas infer properly
            return pd.read_csv(io.BytesIO(data_content), **kwargs)

        elif data_format == 'parquet':
            return pd.read_parquet(io.BytesIO(data_content), **kwargs)

        elif data_format == 'txt':
            # Return decoded string
            return data_content.decode('utf-8')

        else:
            raise ValueError("Unsupported data format. Use 'pkl', 'csv', 'parquet', or 'txt'")

    def save_data_from_path(self, s3_key, data, data_format='pkl', **kwargs):
        """
        Save data to S3 using a direct key path.
        
        Args:
            s3_key: Full S3 key path (e.g., 'new-peak-project/experiments/ch5-paper/00_99_v1_test-experiment/data/v1_test_data.pkl')
            data: Data to save (any serializable object)
            data_format: Format ('pkl', 'csv', 'parquet', 'txt')
            **kwargs: Additional arguments for serialization (e.g., pickle protocol, CSV options)
        
        Raises:
            ValueError: If data_format is unsupported or data lacks required methods (to_csv/to_parquet)
        """
        # Serialize data based on format (reuse logic from save_data)
        if data_format == 'pkl':
            body = io.BytesIO()
            pickle.dump(data, body, **kwargs)
            content_type = 'application/octet-stream'
            
        elif data_format == 'csv':
            if hasattr(data, 'to_csv'):
                body = io.StringIO()
                # Ensure index is not saved to avoid column mismatch
                kwargs.setdefault('index', False)
                data.to_csv(body, **kwargs)
                content_type = 'text/csv'
            else:
                raise ValueError("Data does not have a 'to_csv' method")
                
        elif data_format == 'parquet':
            if hasattr(data, 'to_parquet'):
                body = io.BytesIO()
                # Ensure index is not saved for parquet as well
                kwargs.setdefault('index', False)
                data.to_parquet(body, **kwargs)
                content_type = 'application/octet-stream'
            else:
                raise ValueError("Data does not have a 'to_parquet' method")
                
        elif data_format == 'txt':
            body = str(data)
            content_type = 'text/plain'
            
        else:
            raise ValueError("Unsupported data format. Use 'pkl', 'csv', 'parquet', or 'txt'")
        
        # Upload to S3
        self._upload_with_progress(body, s3_key, content_type=content_type)
    
    def save_figure(self, notebook_config, fig, fig_name, fig_format='png', **kwargs):
        """
        Save matplotlib figure to S3.
        
        Args:
            notebook_config: Dictionary with notebook configuration
            fig: Matplotlib figure object
            fig_name: Name for the figure file
            fig_format: Image format ('png', 'pdf', 'svg')
            **kwargs: Additional arguments for fig.savefig
        """
        version_number = notebook_config.get('version_number', 'v1')
        filename = f"{version_number}_{fig_name}.{fig_format}"
        key = self._get_s3_key(notebook_config, subfolder='figures', filename=filename)
        
        # Save figure to bytes buffer
        buffer = io.BytesIO()
        fig.savefig(buffer, format=fig_format, **kwargs)
        
        # Determine content type
        content_types = {
            'png': 'image/png',
            'pdf': 'application/pdf',
            'svg': 'image/svg+xml',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg'
        }
        content_type = content_types.get(fig_format, 'application/octet-stream')
        
        # Upload to S3
        self._upload_with_progress(buffer, key, content_type=content_type)
    
    def list_experiment_files(self, notebook_config):
        """
        List all files for an experiment in S3.
        
        Args:
            notebook_config: Dictionary with notebook configuration
            
        Returns:
            Dictionary with file lists for config, data, and figures
        """
        base_key = self._get_s3_key(notebook_config)
        
        result = {'config': [], 'data': [], 'figures': []}
        
        try:
            # List objects with prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=base_key):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('_config.yml'):
                            result['config'].append(key)
                        elif '/data/' in key:
                            result['data'].append(key)
                        elif '/figures/' in key:
                            result['figures'].append(key)
        except ClientError as e:
            print(f"Error listing files: {e}")
            
        return result
    
    def list_files_from_path(self, s3_prefix):
        """
        List all files under a given S3 prefix (directory).
        
        Args:
            s3_prefix: S3 prefix (e.g., 'new-peak-project/experiments/ch5-paper/00_99_v1_test-experiment/')
            
        Returns:
            List of S3 keys under the prefix
        """
        keys = []
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        keys.append(obj['Key'])
        except ClientError as e:
            print(f"Error listing files: {e}")
            
        return keys

    def delete_files_from_path(self, s3_prefix):
        """
        Delete all files under a given S3 prefix.
        
        Args:
            s3_prefix: S3 prefix (e.g., 'new-peak-project/experiments/ch5-paper/00_99_v1_test-experiment/')
        """
        keys = self.list_files_from_path(s3_prefix)
        
        if keys:
            # Delete objects in batches of 1000 (S3 limit)
            for i in range(0, len(keys), 1000):
                batch = keys[i:i+1000]
                delete_objects = [{'Key': key} for key in batch]
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': delete_objects}
                )
                
            print(f"Deleted {len(keys)} files from prefix: {s3_prefix}")

    def delete_experiment_files(self, notebook_config, delete_config=True, delete_data=True, delete_figures=True):
        """
        Delete all files for an experiment in S3.
        
        Args:
            notebook_config: Dictionary with notebook configuration
            delete_config: Delete config files
            delete_data: Delete data files
            delete_figures: Delete figure files
        """
        files = self.list_experiment_files(notebook_config)
        
        keys_to_delete = []
        if delete_config:
            keys_to_delete.extend(files['config'])
        if delete_data:
            keys_to_delete.extend(files['data'])
        if delete_figures:
            keys_to_delete.extend(files['figures'])
        
        if keys_to_delete:
            # Delete objects in batches of 1000 (S3 limit)
            for i in range(0, len(keys_to_delete), 1000):
                batch = keys_to_delete[i:i+1000]
                delete_objects = [{'Key': key} for key in batch]
                self.s3_client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': delete_objects}
                )
                
            print(f"Deleted {len(keys_to_delete)} files for experiment")


# Convenience function for quick S3 connection test
def test_s3_connection():
    """Test S3 connection with current environment configuration."""
    try:
        manager = S3ConfigManager()
        print("✅ S3ConfigManager initialized successfully")
        return True
    except Exception as e:
        print(f"❌ S3ConfigManager initialization failed: {e}")
        return False
