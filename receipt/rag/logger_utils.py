import sys
import os
from pathlib import Path

class TerminalTee:
    """
    Context manager to redirect both stdout and stderr to both the terminal and a file.
    """
    def __init__(self, filename):
        self.filename = filename
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.log = None

    def __enter__(self):
        self.log = open(self.filename, 'w', encoding='utf-8')
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.terminal_stdout
        sys.stderr = self.terminal_stderr
        
        # If there's an exception, write it to the log too
        if exc_type:
            import traceback
            self.log.write("\n" + "="*70 + "\n")
            self.log.write("CRITICAL ERROR CAPTURED:\n")
            self.log.write("".join(traceback.format_exception(exc_type, exc_val, exc_tb)))
            self.log.write("="*70 + "\n")
            
        if self.log:
            self.log.close()

    def write(self, message):
        """Write message to the original terminal and the log file."""
        self.terminal_stdout.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        """Flush both original terminal and log file."""
        if self.terminal_stdout:
            self.terminal_stdout.flush()
        if self.log:
            self.log.flush()

    def close(self):
        """Close the log file."""
        if self.log:
            self.log.close()
            self.log = None

    def isatty(self):
        """Standard file-like method used by libraries to check for TTY."""
        return self.terminal_stdout.isatty() if hasattr(self.terminal_stdout, 'isatty') else False

    @property
    def encoding(self):
        """Preserve the terminal's encoding."""
        return getattr(self.terminal_stdout, 'encoding', 'utf-8')

    @property
    def errors(self):
        """Preserve the terminal's error handling."""
        return getattr(self.terminal_stdout, 'errors', 'strict')

def setup_output_capture(script_path):
    """
    Setup output capture for a script.
    Returns a TerminalTee instance that can be used as a context manager.
    """
    p = Path(script_path)
    output_file = p.parent / f"{p.stem}_output.txt"
    return TerminalTee(output_file)
