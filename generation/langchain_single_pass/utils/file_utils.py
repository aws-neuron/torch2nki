"""
File utility functions for the NKI kernel generation pipeline.
"""
import datetime
from typing import Optional


def read_file(path: str) -> str:
    """Read content from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> None:
    """Write content to a file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def log_to_file(log_file_path: str, message: str, append: bool = True) -> None:
    """Log a message to a file, with option to append or overwrite."""
    mode = "a" if append else "w"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, mode, encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def ensure_directory_exists(path: str) -> None:
    """Ensure that a directory exists, creating it if necessary."""
    import os
    os.makedirs(path, exist_ok=True) 