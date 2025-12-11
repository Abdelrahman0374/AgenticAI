from pathlib import Path
from pydantic import BaseModel, Field
from .base_tool import BaseTool
from ..models import ToolResult


class WriteFileArgs(BaseModel):
    """Arguments for writing a file."""
    file_path: str = Field(..., description="The name of the file to write in the workspace")
    content: str = Field(..., description="The content to write to the file")
    mode : str = Field("w", description="The file write mode, e.g., 'w' for write, 'a' for append")

class WriteFileTool(BaseTool):
    """Tool for writing content to files in the workspace."""

    def __init__(self, root_dir: str = "./workspace"):
        """Initialize the write file tool.

        Args:
            root_dir: The root directory for file operations. Defaults to "./workspace".
        """
        super().__init__(
            name="write_file",
            description="Writes content to a file in the workspace. Only accepts filenames, not paths.",
            args_schema=WriteFileArgs
        )
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def run(self, file_path: str, content: str, mode: str = "w") -> ToolResult:
        """Write content to a file.

        Args:
            file_path: The name of the file to write (filename only, no directory paths).
            content: The content to write to the file.

        Returns:
            A ToolResult with success message or error.
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

            with open(target_path, mode, encoding='utf-8') as f:
                f.write(content)

            return ToolResult(
                success=True,
                result=f"File written to '{file_path}'.",
            )

        except PermissionError:
            return ToolResult(success=False, error=f"Permission denied to write to '{file_path}'.")
        except OSError as e:
            return ToolResult(success=False, error=f"Cannot write to '{file_path}': {str(e)}")
        except Exception as e:
            return ToolResult(success=False, error=f"Error writing file: {str(e)}")
