# LSP Client for Python

A comprehensive Python implementation of the Language Server Protocol (LSP) client based on the [MCP Language Server](https://github.com/isaacphi/mcp-language-server) project structure. This client provides all LSP interactions with easy-to-use functions to trigger various language server operations.

## Features

- **Complete LSP Implementation**: Support for all major LSP features including hover, definition, references, diagnostics, formatting, and more
- **Async/Await Support**: Fully asynchronous implementation using `asyncio`
- **Multiple Language Server Support**: Works with any LSP-compliant language server
- **High-level Tools**: Additional utilities for common operations like getting symbol definitions and formatted diagnostics
- **Error Handling**: Robust error handling and timeout management
- **File Management**: Automatic file opening/closing and change notifications
- **Context Management**: Easy-to-use async context managers

## Installation

1. Clone or download the `lsp_client.py` file
2. Install required dependencies:
```bash
pip install asyncio
```

3. Install a language server for your target language:

### Python
```bash
pip install python-lsp-server
```

### TypeScript/JavaScript
```bash
npm install -g typescript typescript-language-server
```

### Go
```bash
go install golang.org/x/tools/gopls@latest
```

### Rust
```bash
rustup component add rust-analyzer
```

## Quick Start

```python
import asyncio
from lsp_client import LSPClient, LSPTools

async def main():
    # Create and start an LSP client
    async with LSPClient("pylsp") as client:
        # Initialize with workspace
        await client.initialize("/path/to/your/project")
        
        # Open a file
        await client.did_open("example.py")
        
        # Get hover information
        hover = await client.get_hover("example.py", 10, 5)
        print("Hover:", hover)
        
        # Get definition
        definition = await client.get_definition("example.py", 10, 5)
        print("Definition:", definition)
        
        # Get references
        references = await client.get_references("example.py", 10, 5)
        print("References:", references)

asyncio.run(main())
```

## LSP Client API Reference

### Core LSP Operations

#### File Operations
- `did_open(file_path)` - Notify server that a file is opened
- `did_change(file_path, content=None)` - Notify server of file changes
- `did_save(file_path)` - Notify server that a file was saved
- `did_close(file_path)` - Notify server that a file is closed

#### Language Features
- `get_hover(file_path, line, character)` - Get hover information at position
- `get_definition(file_path, line, character)` - Get definition at position
- `get_references(file_path, line, character, include_declaration=True)` - Get references
- `get_completion(file_path, line, character)` - Get completion suggestions
- `get_signature_help(file_path, line, character)` - Get signature help
- `get_document_symbols(file_path)` - Get document symbols
- `get_workspace_symbols(query="")` - Get workspace symbols
- `format_document(file_path)` - Format entire document
- `format_range(file_path, start_line, start_char, end_line, end_char)` - Format range
- `rename_symbol(file_path, line, character, new_name)` - Rename symbol
- `get_code_actions(file_path, start_line, start_char, end_line, end_char, diagnostics=None)` - Get code actions
- `get_code_lens(file_path)` - Get code lens information
- `execute_command(command, arguments=None)` - Execute workspace command

#### Diagnostics
- `get_diagnostics(file_path)` - Get cached diagnostics for a file

#### Utility Methods
- `find_symbol_definition(symbol_name, file_path=None)` - Find definition by symbol name
- `apply_text_edits(edits)` - Apply multiple text edits to files

### High-Level Tools (LSPTools)

The `LSPTools` class provides high-level operations built on top of the core LSP client:

```python
tools = LSPTools(client)

# Get formatted symbol definition with source code
definition_content = await tools.get_symbol_definition_content("MyFunction")

# Get all references with surrounding context
references_content = await tools.get_all_references_content("file.py", 10, 5)

# Get formatted diagnostics report
diagnostics_report = await tools.get_diagnostics_report("file.py")
```

## Language Server Configurations

### Python (pylsp)
```python
async with LSPClient("pylsp") as client:
    await client.initialize("/path/to/python/project")
    # Use Python LSP features
```

### TypeScript (typescript-language-server)
```python
async with LSPClient("typescript-language-server", ["--stdio"]) as client:
    await client.initialize("/path/to/typescript/project")
    # Use TypeScript LSP features
```

### Go (gopls)
```python
async with LSPClient("gopls") as client:
    await client.initialize("/path/to/go/project")
    # Use Go LSP features
```

### Rust (rust-analyzer)
```python
async with LSPClient("rust-analyzer") as client:
    await client.initialize("/path/to/rust/project")
    # Use Rust LSP features
```

## Data Structures

### Position
```python
@dataclass
class Position:
    line: int      # 0-based line number
    character: int # 0-based character offset
```

### Range
```python
@dataclass
class Range:
    start: Position
    end: Position
```

### Location
```python
@dataclass
class Location:
    uri: str    # File URI
    range: Range
```

### TextEdit
```python
@dataclass
class TextEdit:
    range: Range
    newText: str
```

### Diagnostic
```python
@dataclass
class Diagnostic:
    range: Range
    message: str
    severity: Optional[int] = None      # 1=Error, 2=Warning, 3=Info, 4=Hint
    code: Optional[Union[int, str]] = None
    source: Optional[str] = None
```

## Examples

Run the comprehensive examples:

```bash
python lsp_examples.py
```

### Basic Usage Example

```python
import asyncio
from lsp_client import LSPClient

async def basic_example():
    async with LSPClient("pylsp") as client:
        await client.initialize("/path/to/project")
        
        # Open a Python file
        await client.did_open("script.py")
        
        # Get symbols in the document
        symbols = await client.get_document_symbols("script.py")
        for symbol in symbols:
            print(f"Symbol: {symbol.get('name')} (kind: {symbol.get('kind')})")
        
        # Get hover info for a function
        hover = await client.get_hover("script.py", 5, 10)
        if hover and 'contents' in hover:
            print(f"Hover: {hover['contents']}")
        
        # Find all references to a symbol
        refs = await client.get_references("script.py", 8, 15)
        print(f"Found {len(refs)} references")

asyncio.run(basic_example())
```

### Advanced Example with Error Handling

```python
import asyncio
from lsp_client import LSPClient, LSPTools, LSPError

async def advanced_example():
    try:
        async with LSPClient("pylsp") as client:
            await client.initialize("/path/to/project")
            
            # Use high-level tools
            tools = LSPTools(client)
            
            # Get detailed symbol definition
            definition = await tools.get_symbol_definition_content("MyClass")
            print("Definition:", definition)
            
            # Get formatted diagnostics
            diagnostics = await tools.get_diagnostics_report("script.py")
            print("Diagnostics:", diagnostics)
            
    except LSPError as e:
        print(f"LSP Error {e.code}: {e.message}")
    except Exception as e:
        print(f"General error: {e}")

asyncio.run(advanced_example())
```

### Multi-file Workspace Example

```python
import asyncio
from lsp_client import LSPClient

async def workspace_example():
    async with LSPClient("pylsp") as client:
        await client.initialize("/path/to/project")
        
        # Open multiple files
        files = ["module1.py", "module2.py", "main.py"]
        for file in files:
            await client.did_open(file)
        
        # Search workspace symbols
        symbols = await client.get_workspace_symbols("MyClass")
        for symbol in symbols:
            location = symbol.get('location', {})
            print(f"Found {symbol['name']} in {location.get('uri', 'unknown')}")
        
        # Get cross-file references
        references = await client.get_references("module1.py", 10, 5)
        for ref in references:
            print(f"Reference at {ref.uri}:{ref.range.start.line + 1}")

asyncio.run(workspace_example())
```

## Error Handling

The client includes comprehensive error handling:

### LSP Errors
```python
from lsp_client import LSPError

try:
    result = await client.get_definition("file.py", 10, 5)
except LSPError as e:
    print(f"LSP Error {e.code}: {e.message}")
    if e.data:
        print(f"Additional data: {e.data}")
```

### Timeout Handling
```python
# All requests have a 30-second timeout by default
try:
    result = await client.get_hover("file.py", 10, 5)
except LSPError as e:
    if "timeout" in e.message.lower():
        print("Request timed out")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Architecture

The LSP client is built with several key components:

- **LSPClient**: Core client handling LSP communication
- **Message Handling**: Async message processing with proper framing
- **Error Management**: Comprehensive error handling and recovery
- **File Tracking**: Automatic file state management
- **Diagnostic Caching**: Efficient diagnostic storage and retrieval
- **High-level Tools**: User-friendly operations built on core functionality

## License

This project is based on the MCP Language Server project and follows similar licensing terms. See the original project for license details.

## Troubleshooting

### Common Issues

1. **Language Server Not Found**
   ```
   LSPError: LSP command not found: pylsp
   ```
   Solution: Install the language server (`pip install python-lsp-server`)

2. **Connection Timeout**
   ```
   LSPError: Request timeout: textDocument/hover
   ```
   Solution: Check if the language server is compatible and running properly

3. **File Not Found**
   ```
   FileNotFoundError: File not found: example.py
   ```
   Solution: Ensure the file exists before calling `did_open()`

### Debug Mode

Enable debug logging to see detailed LSP communication:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your LSP client code here
```

### Performance Tips

1. **Reuse Client Instances**: Create one client per workspace and reuse it
2. **Batch Operations**: Open multiple files before performing operations
3. **Cache Results**: Store frequently accessed data like symbols and diagnostics
4. **Proper Cleanup**: Always close files and shutdown the client properly

## Roadmap

- [ ] Support for more LSP features (semantic tokens, inlay hints)
- [ ] Better error recovery and reconnection
- [ ] Performance optimizations
- [ ] Integration with popular editors
- [ ] More language server examples
- [ ] Testing framework integration

## Related Projects

- [MCP Language Server](https://github.com/isaacphi/mcp-language-server) - Original Go implementation
- [Language Server Protocol](https://microsoft.github.io/language-server-protocol/) - Official LSP specification
- [Python LSP Server](https://github.com/python-lsp/python-lsp-server) - Python language server
