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
    def __init__(self, service_name: str = "default"):
        self.service_name = service_name
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
        session_start = f"\n{separator}\nNew {self.service_name} Session Started at {timestamp}\n{separator}\n"
        
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
                logging.Formatter(f"[{self.service_name.upper()}] %(asctime)s | %(name)s | %(levelname)s | %(message)s")
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
                'stdout': logging.getLogger(f"log.{self.service_name}.stdout"),
                'stderr': logging.getLogger(f"log.{self.service_name}.stderr"),
                'syslog': logging.getLogger(f"log.{self.service_name}.syslog"),
                'app': logging.getLogger(f"log.{self.service_name}.appfile")
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

# Global logging manager instances for different services
api_logging_manager = LoggingManager(service_name="api")
training_logging_manager = LoggingManager(service_name="training")

# Initialize both managers
api_logging_manager.setup_logging()
training_logging_manager.setup_logging()

def get_logger(name: str, service: str = "default") -> Optional[logging.Logger]:
    if service == "api":
        return api_logging_manager.get_logger(name)
    elif service == "training":
        return training_logging_manager.get_logger(name)
    else:
        # For backward compatibility
        default_manager = LoggingManager(service_name=service)
        default_manager.setup_logging()
        return default_manager.get_logger(name)

def shutdown_logging(service: str = None):
    """Shutdown the logging system for a specific service or all services."""
    if service == "api":
        api_logging_manager.shutdown()
    elif service == "training":
        training_logging_manager.shutdown()
    else:
        # Shutdown all
        api_logging_manager.shutdown()
        training_logging_manager.shutdown()

