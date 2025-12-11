from pathlib import Path
from .base_tool import BaseTool
from ..models import ToolResult
from pydantic import BaseModel, Field


class ReadFileArgs(BaseModel):
    """Arguments for reading a file."""
    file_path: str = Field(..., description="The name of the file to read from the workspace")


class ReadFileTool(BaseTool):
    """Tool for reading file contents from the workspace."""

    def __init__(self, root_dir: str = "./workspace"):
        """Initialize the read file tool.

        Args:
            root_dir: The root directory for file operations. Defaults to "./workspace".
        """
        super().__init__(
            name="read_file",
            description="Reads the content of a file from the workspace. Only accepts filenames, not paths.",
            args_schema=ReadFileArgs
        )
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def run(self, file_path: str) -> ToolResult:
        """Read and return the contents of a file.

        Args:
            file_path: The name of the file to read (filename only, no directory paths).

        Returns:
            A ToolResult with the file contents or error message.
        """
        try:
            # Only allow simple filenames, no directory separators
            if '/' in file_path or '\\' in file_path or '..' in file_path:
                return ToolResult(success=False, error="Only filenames are allowed. Directory paths are not permitted.")

            # Prevent absolute paths
            path_obj = Path(file_path)
            if path_obj.is_absolute():
                return ToolResult(success=False, error="Only filenames are allowed, not absolute paths.")

            target_path = self.root_dir / file_path

            if not target_path.exists():
                return ToolResult(success=False, error=f"File '{file_path}' not found.")

            if not target_path.is_file():
                return ToolResult(success=False, error=f"'{file_path}' is not a file.")

            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return ToolResult(success=True, result=content)

        except UnicodeDecodeError:
            return ToolResult(success=False, error=f"File '{file_path}' is not a valid UTF-8 text file.")
        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied to read '{file_path}'.")
        except Exception as e:
            return ToolResult(success=False, error=f"Error reading file: {str(e)}")
