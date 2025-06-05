import logging
import logging.config
import logging.handlers
import queue
import sys
import json
import os
from typing import Optional
from datetime import datetime

class LoggingManager:
    def __init__(self):
        # === Define the central log queue ===
        self.log_queue = queue.Queue(-1)
        self.listener = None
        self.loggers = {}
        # Get project root directory (where the LOGGING_SERVICE folder is)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
    def _get_log_path(self, log_file):
        """Convert relative log path to absolute path from project root."""
        return os.path.join(self.project_root, log_file)
        
    def _write_separator_to_logs(self):
        """Write a separator line to all log files."""
        separator = "#" * 50
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        session_start = f"\n{separator}\nNew Session Started at {timestamp}\n{separator}\n"
        
        log_files = [
            "Logs/api_log.log",
            "Logs/server_log.log",
            "Logs/app_combined.log",
            "Logs/system_log.log"
        ]
        
        for log_file in log_files:
            try:
                abs_path = self._get_log_path(log_file)
                if os.path.exists(abs_path):
                    with open(abs_path, 'a') as f:
                        f.write(session_start)
            except Exception as e:
                print(f"Warning: Could not write separator to {abs_path}: {e}")

    def load_logging_config(self) -> dict:
        """Load and validate logging configuration from JSON file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Ensure log directories exist
            log_files = [
                "Logs/api_log.log",
                "Logs/server_log.log",
                "Logs/app_combined.log",
                "Logs/system_log.log"
            ]
            
            # Create Logs directory in project root
            log_folder = os.path.join(self.project_root, "Logs")
            os.makedirs(log_folder, exist_ok=True)
            
            # Update config with absolute paths
            if 'handlers' in config:
                for handler in config['handlers'].values():
                    if 'filename' in handler:
                        # Convert relative path to absolute path from project root
                        handler['filename'] = self._get_log_path(handler['filename'])
            
            # Create log files if they don't exist
            for log_file in log_files:
                abs_path = self._get_log_path(log_file)
                log_dir = os.path.dirname(abs_path)
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                # Touch the file to create it if it doesn't exist
                if not os.path.exists(abs_path):
                    with open(abs_path, 'a') as f:
                        pass
            
            # Dynamically set the queue for the QueueHandler
            if 'handlers' in config and 'queue_handler' in config['handlers']:
                config['handlers']['queue_handler']['queue'] = self.log_queue
            
            return config
        except FileNotFoundError:
            raise RuntimeError(f"Logging configuration file not found at {config_path}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON in logging configuration file at {config_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading logging configuration: {str(e)}")

    def setup_logging(self):
        """Initialize the logging system with the configuration."""
        try:
            # Apply config
            config = self.load_logging_config()
            logging.config.dictConfig(config)

            # Add QueueListener and output handlers for combined log
            central_handler = logging.FileHandler(self._get_log_path("Logs/app_combined.log"))
            central_handler.setFormatter(
                logging.Formatter("[ROOT] %(asctime)s | %(name)s | %(levelname)s | %(message)s")
            )
            central_handler.setLevel(logging.DEBUG)

            self.listener = logging.handlers.QueueListener(
                self.log_queue,
                central_handler,
                respect_handler_level=True
            )
            self.listener.start()

            # Initialize logger references
            self.loggers = {
                'stdout': logging.getLogger("log.stdout"),
                'stderr': logging.getLogger("log.stderr"),
                'syslog': logging.getLogger("log.syslog"),
                'app': logging.getLogger("log.appfile")
            }
        except Exception as e:
            raise RuntimeError(f"Failed to setup logging: {str(e)}")

    def get_logger(self, name: str) -> Optional[logging.Logger]:
        """Get a logger by name (stdout, stderr, syslog, app)."""
        return self.loggers.get(name)

    def shutdown(self):
        """Properly shutdown logging system."""
        if self.listener:
            self.listener.stop()
            self.listener = None
        # Ensure all handlers are closed properly
        for logger in self.loggers.values():
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        logging.shutdown()

# Global logging manager instance
_logging_manager = None

def init_logging():
    """Initialize the logging system."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
        _logging_manager.setup_logging()
        _logging_manager._write_separator_to_logs()
    return _logging_manager

def get_logger(name: str) -> Optional[logging.Logger]:
    """Get a logger by name. Initializes logging system if needed."""
    if _logging_manager is None:
        init_logging()
    return _logging_manager.get_logger(name)

def shutdown_logging():
    """Shutdown the logging system."""
    global _logging_manager
    if _logging_manager:
        _logging_manager.shutdown()
        _logging_manager = None

# Example usage
if __name__ == "__main__":
    try:
        init_logging()
        
        # Get loggers
        stdout_logger = get_logger('stdout')
        stderr_logger = get_logger('stderr')
        syslog_logger = get_logger('syslog')
        app_logger = get_logger('app')

        # Example logging
        stdout_logger.info("Something to stdout")
        stderr_logger.error("An error occurred")
        syslog_logger.warning("System warning")
        app_logger.debug("Debugging internal app logic")
    
    finally:
        # Ensure proper cleanup
        shutdown_logging()
