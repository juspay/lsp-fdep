import json
import asyncio
import subprocess
import os
import threading
import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from enum import Enum
from pathlib import Path
import urllib.parse

from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSPMessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


class Position(BaseModel):
    """LSP Position structure"""

    line: int  # 0-based
    character: int  # 0-based


class Range(BaseModel):
    """LSP Range structure"""

    start: Position
    end: Position


class Location(BaseModel):
    """LSP Location structure"""

    uri: str
    range: Range


class TextEdit(BaseModel):
    """LSP TextEdit structure"""

    range: Range
    newText: str


class Diagnostic(BaseModel):
    """LSP Diagnostic structure"""

    range: Range
    message: str
    severity: Optional[int] = None
    code: Optional[Union[int, str]] = None
    source: Optional[str] = None


class LSPMessage(BaseModel):
    """LSP Message structure"""

    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class LSPError(Exception):
    """LSP specific error"""

    def __init__(self, code: int, message: str, data: Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"LSP Error {code}: {message}")


class LSPClient:
    """
    Language Server Protocol Client implementation
    """

    def __init__(self, command: str, args: List[str] = None, cwd: str = None):
        self.command = command
        self.args = args or []
        self.cwd = cwd or os.getcwd()

        # Process management
        self.process: Optional[subprocess.Popen] = None
        self.running = False

        # Message handling
        self.request_id = 0
        self.pending_requests: Dict[Union[str, int], asyncio.Future] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        self.request_handlers: Dict[str, Callable] = {}

        # State management
        self.initialized = False
        self.server_capabilities: Dict[str, Any] = {}
        self.open_files: Dict[str, Dict[str, Any]] = {}
        self.diagnostics: Dict[str, List[Diagnostic]] = {}

        # Async management
        self.read_task: Optional[asyncio.Task] = None
        self.lock = asyncio.Lock()

    def path_to_uri(self, path: str) -> str:
        """Convert file path to URI"""
        abs_path = os.path.abspath(path)
        return f"file://{urllib.parse.quote(abs_path)}"

    def uri_to_path(self, uri: str) -> str:
        """Convert URI to file path"""
        if uri.startswith("file://"):
            return urllib.parse.unquote(uri[7:])
        return uri

    def detect_language_id(self, file_path: str) -> str:
        """Detect language ID from file extension"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".clj": "clojure",
            ".dart": "dart",
            ".lua": "lua",
            ".sh": "shellscript",
            ".bash": "shellscript",
            ".zsh": "shellscript",
            ".fish": "shellscript",
            ".ps1": "powershell",
            ".sql": "sql",
            ".html": "html",
            ".htm": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".less": "less",
            ".xml": "xml",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".md": "markdown",
            ".tex": "latex",
            ".r": "r",
            ".R": "r",
        }
        return language_map.get(ext, "plaintext")

    async def start(self) -> None:
        """Start the LSP server process"""
        if self.running:
            return

        try:
            # Start the language server process
            cmd = [self.command] + self.args
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                text=False,
                bufsize=0,
            )

            self.running = True
            logger.info(f"Started LSP server: {' '.join(cmd)}")

            # Start reading messages
            self.read_task = asyncio.create_task(self._read_messages())

            # Start stderr monitoring
            asyncio.create_task(self._monitor_stderr())

        except Exception as e:
            logger.error(f"Failed to start LSP server: {e}")
            raise

    async def _monitor_stderr(self) -> None:
        """Monitor stderr output from the language server"""
        if not self.process or not self.process.stderr:
            return

        loop = asyncio.get_event_loop()
        while self.running:
            try:
                line = await loop.run_in_executor(None, self.process.stderr.readline)
                if line:
                    decoded_line = line.decode("utf-8", errors="ignore").strip()
                    if decoded_line:
                        logger.info(f"LSP stderr: {decoded_line}")
                else:
                    break
            except Exception as e:
                logger.error(f"Error reading stderr: {e}")
                break

    async def _read_messages(self) -> None:
        """Read messages from the LSP server"""
        buffer = b""

        while self.running and self.process and self.process.stdout:
            try:
                # Read data
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.read, 4096
                )

                if not data:
                    break

                buffer += data

                # Process complete messages
                while b"\r\n\r\n" in buffer:
                    header_end = buffer.find(b"\r\n\r\n")
                    header = buffer[:header_end].decode("utf-8")

                    # Parse content length
                    content_length = None
                    for line in header.split("\r\n"):
                        if line.startswith("Content-Length:"):
                            content_length = int(line.split(":")[1].strip())
                            break

                    if content_length is None:
                        logger.error("No Content-Length header found")
                        buffer = buffer[header_end + 4 :]
                        continue

                    # Check if we have the complete message
                    message_start = header_end + 4
                    if len(buffer) < message_start + content_length:
                        break

                    # Extract and process message
                    message_data = buffer[
                        message_start : message_start + content_length
                    ]
                    buffer = buffer[message_start + content_length :]

                    try:
                        message_str = message_data.decode("utf-8")
                        message = json.loads(message_str)
                        await self._handle_message(message)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

            except Exception as e:
                logger.error(f"Error reading from LSP server: {e}")
                break

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle incoming LSP message"""
        try:
            if "id" in message and "method" not in message:
                # Response
                msg_id = message["id"]
                if msg_id in self.pending_requests:
                    future = self.pending_requests.pop(msg_id)
                    if "error" in message:
                        error = message["error"]
                        future.set_exception(
                            LSPError(
                                error.get("code", -1),
                                error.get("message", "Unknown error"),
                                error.get("data"),
                            )
                        )
                    else:
                        future.set_result(message.get("result"))

            elif "method" in message:
                method = message["method"]
                params = message.get("params", {})

                if "id" in message:
                    # Server request
                    if method in self.request_handlers:
                        try:
                            result = await self.request_handlers[method](params)
                            await self._send_response(message["id"], result)
                        except Exception as e:
                            await self._send_error(message["id"], -32603, str(e))
                    else:
                        await self._send_error(
                            message["id"], -32601, f"Method not found: {method}"
                        )
                else:
                    # Notification
                    if method in self.notification_handlers:
                        try:
                            await self.notification_handlers[method](params)
                        except Exception as e:
                            logger.error(f"Error handling notification {method}: {e}")

                    # Handle built-in notifications
                    await self._handle_builtin_notification(method, params)

        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_builtin_notification(
        self, method: str, params: Dict[str, Any]
    ) -> None:
        """Handle built-in LSP notifications"""
        if method == "textDocument/publishDiagnostics":
            uri = params.get("uri", "")
            diagnostics_data = params.get("diagnostics", [])

            # Convert to Diagnostic objects
            diagnostics = []
            for diag_data in diagnostics_data:
                range_data = diag_data["range"]
                diag = Diagnostic(
                    range=Range(
                        start=Position(
                            line=range_data["start"]["line"],
                            character=range_data["start"]["character"],
                        ),
                        end=Position(
                            line=range_data["end"]["line"],
                            character=range_data["end"]["character"],
                        ),
                    ),
                    message=diag_data["message"],
                    severity=diag_data.get("severity"),
                    code=diag_data.get("code"),
                    source=diag_data.get("source"),
                )
                diagnostics.append(diag)

            self.diagnostics[uri] = diagnostics
            logger.debug(
                f"Updated diagnostics for {uri}: {len(diagnostics)} diagnostics"
            )

    async def _send_message(self, message: Dict[str, Any]) -> None:
        """Send message to LSP server"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("LSP server not running")

        message_str = json.dumps(message, separators=(",", ":"))
        message_bytes = message_str.encode("utf-8")

        header = f"Content-Length: {len(message_bytes)}\r\n\r\n"
        header_bytes = header.encode("utf-8")

        try:
            self.process.stdin.write(header_bytes + message_bytes)
            self.process.stdin.flush()
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise

    async def _send_request(self, method: str, params: Any = None) -> Any:
        """Send request and wait for response"""
        async with self.lock:
            self.request_id += 1
            request_id = self.request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }

        future = asyncio.Future()
        self.pending_requests[request_id] = future

        await self._send_message(message)

        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self.pending_requests.pop(request_id, None)
            raise LSPError(-32000, f"Request timeout: {method}")

    async def _send_notification(self, method: str, params: Any = None) -> None:
        """Send notification"""
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        await self._send_message(message)

    async def _send_response(self, request_id: Union[str, int], result: Any) -> None:
        """Send response to server request"""
        message = {"jsonrpc": "2.0", "id": request_id, "result": result}
        await self._send_message(message)

    async def _send_error(
        self, request_id: Union[str, int], code: int, message: str, data: Any = None
    ) -> None:
        """Send error response to server request"""
        error_msg = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
        if data is not None:
            error_msg["error"]["data"] = data
        await self._send_message(error_msg)

    # LSP Lifecycle Methods

    async def initialize(self, workspace_path: str) -> Dict[str, Any]:
        """Initialize the LSP server"""
        params = {
            "processId": os.getpid(),
            "clientInfo": {"name": "python-lsp-client", "version": "1.0.0"},
            "rootPath": workspace_path,
            "rootUri": self.path_to_uri(workspace_path),
            "capabilities": {
                "workspace": {
                    "configuration": True,
                    "didChangeConfiguration": {"dynamicRegistration": True},
                    "didChangeWatchedFiles": {
                        "dynamicRegistration": True,
                        "relativePatternSupport": True,
                    },
                },
                "textDocument": {
                    "synchronization": {"dynamicRegistration": True, "didSave": True},
                    "completion": {"completionItem": {}},
                    "hover": {
                        "dynamicRegistration": True,
                        "contentFormat": ["markdown", "plaintext"],
                    },
                    "signatureHelp": {"dynamicRegistration": True},
                    "definition": {"dynamicRegistration": True, "linkSupport": True},
                    "references": {"dynamicRegistration": True},
                    "documentHighlight": {"dynamicRegistration": True},
                    "documentSymbol": {
                        "dynamicRegistration": True,
                        "hierarchicalDocumentSymbolSupport": True,
                    },
                    "formatting": {"dynamicRegistration": True},
                    "rangeFormatting": {"dynamicRegistration": True},
                    "onTypeFormatting": {"dynamicRegistration": True},
                    "rename": {"dynamicRegistration": True, "prepareSupport": True},
                    "codeAction": {
                        "dynamicRegistration": True,
                        "codeActionLiteralSupport": {
                            "codeActionKind": {"valueSet": []}
                        },
                    },
                    "codeLens": {"dynamicRegistration": True},
                    "documentLink": {"dynamicRegistration": True},
                    "publishDiagnostics": {"versionSupport": True},
                },
                "window": {},
            },
            "initializationOptions": {},
            "workspaceFolders": [
                {
                    "uri": self.path_to_uri(workspace_path),
                    "name": os.path.basename(workspace_path),
                }
            ],
        }

        result = await self._send_request("initialize", params)
        self.server_capabilities = result.get("capabilities", {})
        self.initialized = True

        # Send initialized notification
        await self._send_notification("initialized", {})

        logger.info("LSP server initialized successfully")
        return result

    async def shutdown(self) -> None:
        """Shutdown the LSP server"""
        if self.initialized:
            await self._send_request("shutdown")
            await self._send_notification("exit")
            self.initialized = False

    async def stop(self) -> None:
        """Stop the LSP client and server"""
        if not self.running:
            return

        self.running = False

        try:
            if self.initialized:
                await self.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        if self.read_task:
            self.read_task.cancel()
            try:
                await self.read_task
            except asyncio.CancelledError:
                pass

        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.error(f"Error stopping process: {e}")

        logger.info("LSP client stopped")

    # File Operations

    async def did_open(self, file_path: str) -> None:
        """Notify server that a file is opened"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        uri = self.path_to_uri(file_path)

        params = {
            "textDocument": {
                "uri": uri,
                "languageId": self.detect_language_id(file_path),
                "version": 1,
                "text": content,
            }
        }

        self.open_files[uri] = {
            "version": 1,
            "languageId": self.detect_language_id(file_path),
            "path": file_path,
        }

        await self._send_notification("textDocument/didOpen", params)
        logger.debug(f"Opened file: {file_path}")

    async def did_change(self, file_path: str, content: str = None) -> None:
        """Notify server that a file has changed"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)
            return

        if content is None:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        version = self.open_files[uri]["version"] + 1
        self.open_files[uri]["version"] = version

        params = {
            "textDocument": {"uri": uri, "version": version},
            "contentChanges": [{"text": content}],
        }

        await self._send_notification("textDocument/didChange", params)

    async def did_save(self, file_path: str) -> None:
        """Notify server that a file has been saved"""
        uri = self.path_to_uri(file_path)

        params = {"textDocument": {"uri": uri}}

        await self._send_notification("textDocument/didSave", params)

    async def did_close(self, file_path: str) -> None:
        """Notify server that a file is closed"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            return

        params = {"textDocument": {"uri": uri}}

        del self.open_files[uri]
        await self._send_notification("textDocument/didClose", params)
        logger.debug(f"Closed file: {file_path}")

    # Language Server Features

    async def get_hover(
        self, file_path: str, line: int, character: int
    ) -> Optional[Dict[str, Any]]:
        """Get hover information at position"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        }

        result = await self._send_request("textDocument/hover", params)
        return result

    async def get_definition(
        self, file_path: str, line: int, character: int
    ) -> List[Location]:
        """Get definition at position"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        }

        result = await self._send_request("textDocument/definition", params)

        if not result:
            return []

        # Handle both single location and array of locations
        locations = result if isinstance(result, list) else [result]

        return [
            Location(
                uri=loc["uri"],
                range=Range(
                    start=Position(
                        line=loc["range"]["start"]["line"],
                        character=loc["range"]["start"]["character"],
                    ),
                    end=Position(
                        line=loc["range"]["end"]["line"],
                        character=loc["range"]["end"]["character"],
                    ),
                ),
            )
            for loc in locations
        ]

    async def get_references(
        self,
        file_path: str,
        line: int,
        character: int,
        include_declaration: bool = True,
    ) -> List[Location]:
        """Get references at position"""
        uri = self.path_to_uri(file_path)
        print(uri)
        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
            "context": {"includeDeclaration": include_declaration},
        }

        result = await self._send_request("textDocument/references", params)

        if not result:
            return []
        return [
            Location(
                uri=loc["uri"],
                range=Range(
                    start=Position(
                        line=loc["range"]["start"]["line"],
                        character=loc["range"]["start"]["character"],
                    ),
                    end=Position(
                        line=loc["range"]["end"]["line"],
                        character=loc["range"]["end"]["character"],
                    ),
                ),
            )
            for loc in result
        ]

    async def get_completion(
        self, file_path: str, line: int, character: int
    ) -> Optional[Dict[str, Any]]:
        """Get completion suggestions at position"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        }

        result = await self._send_request("textDocument/completion", params)
        return result

    async def get_signature_help(
        self, file_path: str, line: int, character: int
    ) -> Optional[Dict[str, Any]]:
        """Get signature help at position"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        }

        result = await self._send_request("textDocument/signatureHelp", params)
        return result

    async def get_document_symbols(self, file_path: str) -> List[Dict[str, Any]]:
        """Get document symbols"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {"textDocument": {"uri": uri}}

        result = await self._send_request("textDocument/documentSymbol", params)
        return result or []

    async def get_workspace_symbols(self, query: str = "") -> List[Dict[str, Any]]:
        """Get workspace symbols"""
        params = {"query": query}

        result = await self._send_request("workspace/symbol", params)
        return result or []

    async def format_document(self, file_path: str) -> List[TextEdit]:
        """Format entire document"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {
            "textDocument": {"uri": uri},
            "options": {"tabSize": 4, "insertSpaces": True},
        }

        result = await self._send_request("textDocument/formatting", params)

        if not result:
            return []

        return [
            TextEdit(
                range=Range(
                    start=Position(
                        line=edit["range"]["start"]["line"],
                        character=edit["range"]["start"]["character"],
                    ),
                    end=Position(
                        line=edit["range"]["end"]["line"],
                        character=edit["range"]["end"]["character"],
                    ),
                ),
                newText=edit["newText"],
            )
            for edit in result
        ]

    async def format_range(
        self,
        file_path: str,
        start_line: int,
        start_char: int,
        end_line: int,
        end_char: int,
    ) -> List[TextEdit]:
        """Format document range"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {
            "textDocument": {"uri": uri},
            "range": {
                "start": {"line": start_line, "character": start_char},
                "end": {"line": end_line, "character": end_char},
            },
            "options": {"tabSize": 4, "insertSpaces": True},
        }

        result = await self._send_request("textDocument/rangeFormatting", params)

        if not result:
            return []

        return [
            TextEdit(
                range=Range(
                    start=Position(
                        line=edit["range"]["start"]["line"],
                        character=edit["range"]["start"]["character"],
                    ),
                    end=Position(
                        line=edit["range"]["end"]["line"],
                        character=edit["range"]["end"]["character"],
                    ),
                ),
                newText=edit["newText"],
            )
            for edit in result
        ]

    async def rename_symbol(
        self, file_path: str, line: int, character: int, new_name: str
    ) -> Optional[Dict[str, Any]]:
        """Rename symbol at position"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
            "newName": new_name,
        }

        result = await self._send_request("textDocument/rename", params)
        return result

    async def get_code_actions(
        self,
        file_path: str,
        start_line: int,
        start_char: int,
        end_line: int,
        end_char: int,
        diagnostics: List[Diagnostic] = None,
    ) -> List[Dict[str, Any]]:
        """Get code actions for range"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        diag_data = []
        if diagnostics:
            for diag in diagnostics:
                diag_data.append(
                    {
                        "range": {
                            "start": {
                                "line": diag.range.start.line,
                                "character": diag.range.start.character,
                            },
                            "end": {
                                "line": diag.range.end.line,
                                "character": diag.range.end.character,
                            },
                        },
                        "message": diag.message,
                        "severity": diag.severity,
                        "code": diag.code,
                        "source": diag.source,
                    }
                )

        params = {
            "textDocument": {"uri": uri},
            "range": {
                "start": {"line": start_line, "character": start_char},
                "end": {"line": end_line, "character": end_char},
            },
            "context": {"diagnostics": diag_data},
        }

        result = await self._send_request("textDocument/codeAction", params)
        return result or []

    async def get_code_lens(self, file_path: str) -> List[Dict[str, Any]]:
        """Get code lens for document"""
        uri = self.path_to_uri(file_path)

        if uri not in self.open_files:
            await self.did_open(file_path)

        params = {"textDocument": {"uri": uri}}

        result = await self._send_request("textDocument/codeLens", params)
        return result or []

    async def execute_command(self, command: str, arguments: List[Any] = None) -> Any:
        """Execute workspace command"""
        params = {"command": command, "arguments": arguments or []}

        result = await self._send_request("workspace/executeCommand", params)
        return result

    # Utility methods for high-level operations

    async def get_diagnostics(self, file_path: str) -> List[Diagnostic]:
        """Get cached diagnostics for a file"""
        uri = self.path_to_uri(file_path)
        return self.diagnostics.get(uri, [])

    async def find_symbol_definition(
        self, symbol_name: str, file_path: str = None
    ) -> Optional[Location]:
        """Find the definition of a symbol by name"""
        # First try workspace symbols
        symbols = await self.get_workspace_symbols(symbol_name)

        for symbol in symbols:
            if symbol.get("name") == symbol_name:
                location_data = symbol.get("location")
                if location_data:
                    return Location(
                        uri=location_data["uri"],
                        range=Range(
                            start=Position(
                                line=location_data["range"]["start"]["line"],
                                character=location_data["range"]["start"]["character"],
                            ),
                            end=Position(
                                line=location_data["range"]["end"]["line"],
                                character=location_data["range"]["end"]["character"],
                            ),
                        ),
                    )

        return None

    async def apply_text_edits(self, edits: List[Tuple[str, List[TextEdit]]]) -> None:
        """Apply text edits to multiple files"""
        for file_path, file_edits in edits:
            # Read current file content
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Sort edits by position (reverse order to avoid offset issues)
            sorted_edits = sorted(
                file_edits,
                key=lambda e: (e.range.start.line, e.range.start.character),
                reverse=True,
            )

            # Apply edits
            for edit in sorted_edits:
                start_line = edit.range.start.line
                start_char = edit.range.start.character
                end_line = edit.range.end.line
                end_char = edit.range.end.character

                if start_line == end_line:
                    # Single line edit
                    line = lines[start_line]
                    lines[start_line] = (
                        line[:start_char] + edit.newText + line[end_char:]
                    )
                else:
                    # Multi-line edit
                    start_line_content = lines[start_line][:start_char]
                    end_line_content = lines[end_line][end_char:]

                    # Replace the range with new text
                    new_content = start_line_content + edit.newText + end_line_content

                    # Remove old lines and insert new content
                    del lines[start_line : end_line + 1]
                    if new_content:
                        lines.insert(start_line, new_content)

            # Write back to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            # Notify server of change
            await self.did_change(file_path)

    def register_notification_handler(
        self, method: str, handler: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Register custom notification handler"""
        self.notification_handlers[method] = handler

    def register_request_handler(
        self, method: str, handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Register custom request handler"""
        self.request_handlers[method] = handler

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()


# High-level functions for common operations


async def create_lsp_client(
    command: str, args: List[str] = None, workspace_path: str = None
) -> LSPClient:
    """Create and initialize an LSP client"""
    client = LSPClient(command, args, workspace_path)
    await client.start()

    if workspace_path:
        await client.initialize(workspace_path)

    return client


class LSPTools:
    """High-level tools for LSP operations"""

    def __init__(self, client: LSPClient):
        self.client = client

    async def get_symbol_definition_content(self, symbol_name: str) -> str:
        """Get the source code content of a symbol definition"""
        location = await self.client.find_symbol_definition(symbol_name)

        if not location:
            return f"Symbol '{symbol_name}' not found"

        file_path = self.client.uri_to_path(location.uri)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            start_line = location.range.start.line
            end_line = location.range.end.line

            # Get the definition content
            definition_lines = lines[start_line : end_line + 1]

            # Add line numbers
            numbered_lines = []
            for i, line in enumerate(definition_lines):
                line_num = start_line + i + 1
                numbered_lines.append(f"{line_num:4d}: {line.rstrip()}")

            result = f"Symbol: {symbol_name}\n"
            result += f"File: {file_path}\n"
            result += f"Range: L{start_line + 1}:C{location.range.start.character + 1} - L{end_line + 1}:C{location.range.end.character + 1}\n\n"
            result += "\n".join(numbered_lines)

            return result

        except Exception as e:
            return f"Error reading definition: {e}"

    async def get_all_references_content(
        self, file_path: str, line: int, character: int
    ) -> str:
        """Get all references to a symbol with surrounding context"""
        references = await self.client.get_references(file_path, line, character)

        if not references:
            return "No references found"

        result = f"Found {len(references)} reference(s):\n\n"

        for i, ref in enumerate(references):
            ref_file = self.client.uri_to_path(ref.uri)
            ref_line = ref.range.start.line
            ref_char = ref.range.start.character

            try:
                with open(ref_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Get context lines
                context_start = max(0, ref_line - 2)
                context_end = min(len(lines), ref_line + 3)
                context_lines = lines[context_start:context_end]

                result += f"Reference {i + 1}:\n"
                result += f"File: {ref_file}\n"
                result += f"Position: L{ref_line + 1}:C{ref_char + 1}\n\n"

                for j, line in enumerate(context_lines):
                    line_num = context_start + j + 1
                    marker = ">>>" if context_start + j == ref_line else "   "
                    result += f"{marker} {line_num:4d}: {line.rstrip()}\n"

                result += "\n"

            except Exception as e:
                result += f"Error reading reference: {e}\n\n"

        return result

    async def get_diagnostics_report(self, file_path: str) -> str:
        """Get a formatted diagnostics report for a file"""
        diagnostics = await self.client.get_diagnostics(file_path)

        if not diagnostics:
            return f"No diagnostics found for {file_path}"

        severity_map = {1: "ERROR", 2: "WARNING", 3: "INFO", 4: "HINT"}

        result = f"Diagnostics for {file_path} ({len(diagnostics)} issue(s)):\n\n"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for i, diag in enumerate(diagnostics):
                severity = severity_map.get(diag.severity, "UNKNOWN")
                line_num = diag.range.start.line + 1
                char_num = diag.range.start.character + 1

                result += f"Issue {i + 1}: {severity}\n"
                result += f"Position: L{line_num}:C{char_num}\n"
                result += f"Message: {diag.message}\n"

                if diag.code:
                    result += f"Code: {diag.code}\n"
                if diag.source:
                    result += f"Source: {diag.source}\n"

                # Show the problematic line
                if diag.range.start.line < len(lines):
                    problem_line = lines[diag.range.start.line].rstrip()
                    result += f"Line: {problem_line}\n"

                    # Show pointer to the problem area
                    pointer = " " * (char_num - 1) + "^"
                    if diag.range.end.character > diag.range.start.character:
                        pointer += "~" * (
                            diag.range.end.character - diag.range.start.character - 1
                        )
                    result += f"      {pointer}\n"

                result += "\n"

        except Exception as e:
            result += f"Error reading file: {e}\n"
        return result
