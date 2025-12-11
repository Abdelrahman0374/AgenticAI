from pathlib import Path
from pydantic import BaseModel, Field
from .base_tool import BaseTool
from ..models import ToolResult


class WriteFileArgs(BaseModel):
    """Argument schema for the WriteFileTool.

    Attributes:
        file_path: Name of the file to write (filename only, no directory paths)
        content: Text content to write to the file
        mode: File write mode - 'w' for write/overwrite, 'a' for append
    """
    file_path: str = Field(..., description="The name of the file to write in the workspace")
    content: str = Field(..., description="The content to write to the file")
    mode : str = Field("w", description="The file write mode, e.g., 'w' for write, 'a' for append")

class WriteFileTool(BaseTool):
    """Tool for writing content to files in a workspace directory.

    Provides secure file writing with path validation to prevent directory
    traversal attacks. Only accepts simple filenames (no paths) and writes
    files to the configured workspace directory. Supports both write and
    append modes.

    Security features:
    - Rejects absolute paths
    - Rejects relative paths with directory separators
    - Prevents directory traversal with '..' sequences
    - Enforces UTF-8 encoding

    Attributes:
        name: "write_file"
        description: Tool description for LLM
        root_dir: Path to workspace directory where files are stored
        args_schema: WriteFileArgs for argument validation

    Example:
        tool = WriteFileTool(root_dir="./workspace")
        result = tool.run(file_path="output.txt", content="Hello World")
        if result.success:
            print(result.result)  # "File written to 'output.txt'."
    """

    def __init__(self, root_dir: str = "./workspace"):
        """Initialize the write file tool.

        Args:
            root_dir: The root directory for file operations. All file writes
                     are confined to this directory. Defaults to "./workspace".
                     The directory is created if it doesn't exist.
        """
        super().__init__(
            name="write_file",
            description="Writes content to a file in the workspace. Only accepts filenames, not paths.",
            args_schema=WriteFileArgs
        )
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def run(self, file_path: str, content: str, mode: str = "w") -> ToolResult:
        """Write content to a file in the workspace.

        Validates the file path and writes the content to the file using the
        specified mode. Creates the file if it doesn't exist (write mode) or
        appends to it (append mode).

        Args:
            file_path: The name of the file to write (filename only, no directory
                      paths). Must not contain '/', '\\', or '..' sequences.
            content: The text content to write to the file (UTF-8 encoded)
            mode: File write mode - 'w' for write/overwrite, 'a' for append.
                 Defaults to 'w'.

        Returns:
            ToolResult with:
                - success=True and result=confirmation message if successful
                - success=False and error=message if failed

        Error conditions:
            - Directory paths or absolute paths provided
            - Permission denied
            - Invalid file path or OS error
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
