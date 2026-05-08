"""Dual-output logging utility for evaluation scripts.

Captures all print() statements to both console and timestamped log files.
"""

import os
import sys
from datetime import datetime
from typing import TextIO


class DualOutput:
    """Write to both console and log file simultaneously."""

    def __init__(self, log_file: TextIO, console: TextIO):
        self.log_file = log_file
        self.console = console

    def write(self, message: str):
        """Write message to both outputs."""
        self.console.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to disk

    def flush(self):
        """Flush both outputs."""
        self.console.flush()
        self.log_file.flush()


def setup_dual_output(script_path: str) -> None:
    """Set up dual output (console + log file) for the script.

    Creates logs directory and redirects stdout/stderr to both console
    and a timestamped log file.

    Args:
        script_path: Path to the calling script (use __file__)
    """
    # Get script name without extension
    script_name = os.path.splitext(os.path.basename(script_path))[0]

    # Create logs directory
    logs_dir = os.path.join(os.path.dirname(script_path), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, log_filename)

    # Open log file (overwrite mode)
    log_file = open(log_path, "w", encoding="utf-8")

    # Redirect stdout and stderr
    sys.stdout = DualOutput(log_file, sys.__stdout__)
    sys.stderr = DualOutput(log_file, sys.__stderr__)

    print(f"Logging to: {log_path}")
    print("=" * 70)
