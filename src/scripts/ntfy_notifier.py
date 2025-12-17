"""
ntfy Notification Helper for Batch Scripts

Provides functions to send notifications via ntfy.sh service.
Environment variables required:
- NTFY_URL: Base URL (e.g., 'https://lidonglan.myqnapcloud.com:8098')
- NTFY_TOPIC: Topic name (e.g., 'scripts')
- NTFY_TOKEN: Authentication token (optional)
"""

import os
import json
import time
import socket
import logging
from typing import Optional
from dotenv import dotenv_values

try:
    import requests
    from requests.exceptions import RequestException, Timeout
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configure module-level logger
logger = logging.getLogger(__name__)


def _load_dotenv():
    """
    Load environment variables from .env file using same pattern as s3_config_manager.
    Returns dict of loaded variables.
    """
    # Try to resolve project path to find .env file
    try:
        import sys
        import os.path
        
        # Get current file directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to src directory
        src_dir = os.path.join(current_dir, '..')
        env_file_path = os.path.join(src_dir, '.env')
        
        if os.path.exists(env_file_path):
            env_config = dotenv_values(env_file_path)
            # Merge into os.environ (but don't overwrite existing)
            for key, value in env_config.items():
                if value and key not in os.environ:
                    os.environ[key] = value
            return env_config
        else:
            logger.debug(f".env file not found at {env_file_path}")
    except Exception as e:
        logger.debug(f"Failed to load .env file: {e}")
    
    return {}


def _get_ntfy_config():
    """Get ntfy configuration from environment variables."""
    # Load .env file if variables not set
    if not os.environ.get('NTFY_URL') or not os.environ.get('NTFY_TOPIC'):
        _load_dotenv()
    
    url = os.environ.get('NTFY_URL')
    topic = os.environ.get('NTFY_TOPIC')
    token = os.environ.get('NTFY_TOKEN')
    
    if not url or not topic:
        logger.warning("NTFY_URL or NTFY_TOPIC environment variables not set. Notifications disabled.")
        return None, None, None
    
    # Ensure URL doesn't end with slash
    if url.endswith('/'):
        url = url.rstrip('/')
    
    # Construct full topic URL
    topic_url = f"{url}/{topic}"
    
    return topic_url, token, True


def send_ntfy_notification(
    message: str,
    title: Optional[str] = None,
    priority: Optional[int] = 3,
    tags: Optional[list] = None,
    timeout: float = 5.0
) -> bool:
    """
    Send a notification to ntfy server.
    
    Args:
        message: Notification message content
        title: Optional title (defaults to script name)
        priority: Priority level (1=min, 5=max, default=3)
        tags: List of tags (emoji or text)
        timeout: Request timeout in seconds
    
    Returns:
        bool: True if notification sent successfully, False otherwise
    """
    # Get configuration
    config = _get_ntfy_config()
    if config[0] is None:
        return False  # Not configured
    
    topic_url, token, _ = config
    
    if not REQUESTS_AVAILABLE:
        logger.warning("Requests library not available. Cannot send ntfy notifications.")
        return False
    
    # Prepare headers
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
    # Prepare data
    data = message.encode('utf-8')
    
    # Prepare additional ntfy fields
    if title:
        headers['Title'] = title
    if priority:
        headers['Priority'] = str(priority)
    if tags:
        headers['Tags'] = ','.join(tags)
    
    # Add timestamp
    headers['Timestamp'] = str(int(time.time()))
    
    try:
        response = requests.post(
            topic_url,
            data=data,
            headers=headers,
            timeout=timeout
        )
        
        if response.status_code == 200:
            logger.debug(f"ntfy notification sent successfully: {message[:50]}...")
            return True
        else:
            logger.warning(f"ntfy notification failed with status {response.status_code}: {response.text}")
            return False
            
    except Timeout:
        logger.warning("ntfy notification timeout - server not responding")
        return False
    except RequestException as e:
        logger.warning(f"ntfy notification failed: {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error sending ntfy notification: {e}")
        return False


def notify_start(script_name: str, assembly_id: Optional[str] = None, **extra):
    """
    Send start notification for a batch script.
    
    Args:
        script_name: Name of the script/batch
        assembly_id: Optional assembly ID
        **extra: Additional key-value pairs to include in message
    """
    title = f"{script_name} started"
    message_lines = [f"Batch script {script_name} started execution."]
    
    if assembly_id:
        message_lines.append(f"Assembly ID: {assembly_id}")
    
    if extra:
        for key, value in extra.items():
            message_lines.append(f"{key}: {value}")
    
    message = '\n'.join(message_lines)
    
    send_ntfy_notification(
        message=message,
        title=title,
        priority=3,
        tags=['rocket']
    )


def notify_success(script_name: str, duration_seconds: float, 
                   processed_count: Optional[int] = None, **extra):
    """
    Send success notification for a batch script.
    
    Args:
        script_name: Name of the script/batch
        duration_seconds: Total execution time in seconds
        processed_count: Number of items processed (e.g., distortion factors)
        **extra: Additional key-value pairs to include in message
    """
    title = f"{script_name} completed"
    
    # Format duration
    if duration_seconds < 60:
        duration_str = f"{duration_seconds:.1f}s"
    else:
        minutes = duration_seconds / 60
        duration_str = f"{minutes:.1f}m"
    
    message_lines = [f"Batch script {script_name} completed successfully."]
    message_lines.append(f"Duration: {duration_str}")
    
    if processed_count is not None:
        message_lines.append(f"Processed: {processed_count} items")
    
    if extra:
        for key, value in extra.items():
            message_lines.append(f"{key}: {value}")
    
    message = '\n'.join(message_lines)
    
    send_ntfy_notification(
        message=message,
        title=title,
        priority=2,
        tags=['white_check_mark']
    )


def notify_failure(script_name: str, error: Exception, duration_seconds: float = None):
    """
    Send failure notification for a batch script.
    
    Args:
        script_name: Name of the script/batch
        error: The exception that caused failure
        duration_seconds: Optional execution time before failure
    """
    title = f"{script_name} failed"
    
    error_msg = str(error)
    # Truncate long error messages
    if len(error_msg) > 200:
        error_msg = error_msg[:197] + "..."
    
    message_lines = [f"Batch script {script_name} failed with error:"]
    message_lines.append(f"Error: {error_msg}")
    message_lines.append(f"Error type: {type(error).__name__}")
    
    if duration_seconds is not None:
        message_lines.append(f"Failed after: {duration_seconds:.1f}s")
    
    message = '\n'.join(message_lines)
    
    send_ntfy_notification(
        message=message,
        title=title,
        priority=5,  # High priority for failures
        tags=['x', 'warning']
    )


def notify_custom(script_name: str, status: str, details: str, **extra):
    """
    Send custom notification.
    
    Args:
        script_name: Name of the script/batch
        status: Status text (e.g., "Processing", "Warning")
        details: Detailed message
        **extra: Additional key-value pairs
    """
    title = f"{status} - {script_name}"
    message_lines = [details]
    
    if extra:
        for key, value in extra.items():
            message_lines.append(f"{key}: {value}")
    
    message = '\n'.join(message_lines)
    
    send_ntfy_notification(
        message=message,
        title=title,
        priority=3
    )


# Example usage
if __name__ == "__main__":
    # Test the module (only runs when executed directly)
    import sys
    print("Testing ntfy_notifier module...")
    
    # Set test environment variables
    os.environ['NTFY_URL'] = os.environ.get('NTFY_URL', 'https://ntfy.sh')
    os.environ['NTFY_TOPIC'] = os.environ.get('NTFY_TOPIC', 'test-topic')
    
    # Test basic notification
    success = send_ntfy_notification("Test notification from ntfy_notifier.py")
    if success:
        print("✓ Test notification sent successfully")
    else:
        print("✗ Test notification failed (may be expected if ntfy not configured)")
    
    sys.exit(0 if success else 1)
