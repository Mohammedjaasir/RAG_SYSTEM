import sys
import os
from pathlib import Path

class TerminalTee:
    """
    Context manager to redirect stdout to both the terminal and a file.
    """
    def __init__(self, filename):
        self.filename = filename
        self.terminal = sys.stdout
        self.log = None

    def __enter__(self):
        self.log = open(self.filename, 'w', encoding='utf-8')
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal
        if self.log:
            self.log.close()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        """Standard file-like method used by libraries to check for TTY."""
        return self.terminal.isatty() if hasattr(self.terminal, 'isatty') else False

    @property
    def encoding(self):
        """Preserve the terminal's encoding."""
        return getattr(self.terminal, 'encoding', 'utf-8')

    @property
    def errors(self):
        """Preserve the terminal's error handling."""
        return getattr(self.terminal, 'errors', 'strict')

def setup_output_capture(script_path):
    """
    Setup output capture for a script.
    Returns a TerminalTee instance that can be used as a context manager.
    """
    p = Path(script_path)
    output_file = p.parent / f"{p.stem}_output.txt"
    return TerminalTee(output_file)
