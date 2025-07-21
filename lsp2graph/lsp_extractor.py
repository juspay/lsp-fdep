import os
import json
import asyncio
import traceback
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from lsp_client import LSPClient

SYMBOL_KIND_MAP = {
    1: "File",
    2: "Module",
    3: "Namespace",
    4: "Package",
    5: "Class",
    6: "Method",
    7: "Property",
    8: "Field",
    9: "Constructor",
    10: "Enum",
    11: "Interface",
    12: "Function",
    13: "Variable",
    14: "Constant",
    15: "String",
    16: "Number",
    17: "Boolean",
    18: "Array",
    19: "Object",
    20: "Key",
    21: "Null",
    22: "EnumMember",
    23: "Struct",
    24: "Event",
    25: "Operator",
    26: "TypeParameter",
}


class AsyncUniversalExtractor:
    """Async universal code extractor using LSP"""

    def __init__(self):
        self.lsp_clients: Dict[str, LSPClient] = {}
        self.language_server_configs = {
            "python": ["pylsp"],  # python-lsp-server
            "javascript": ["typescript-language-server", "--stdio"],
            "typescript": ["typescript-language-server", "--stdio"],
            "java": ["jdtls"],  # Eclipse JDT Language Server
            "go": ["gopls"],
            "rust": ["rust-analyzer"],
            "cpp": ["clangd"],
            "c": ["clangd"],
        }

    async def cleanup(self):
        """Clean up all LSP clients and their subprocesses"""
        print("🧹 Cleaning up LSP clients...")
        for language, client in self.lsp_clients.items():
            try:
                print(f"  Stopping {language} LSP client...")
                await client.stop()
                print(f"  ✅ Stopped {language} LSP client")
            except Exception as e:
                print(f"  ❌ Error stopping {language} client: {e}")
        self.lsp_clients.clear()
        print("✅ All LSP clients cleaned up")

    def detect_language(self, file_path: str) -> str:
        """Detect language from file extension"""
        ext = Path(file_path).suffix.lower()

        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".hs": "haskell",
        }

        return language_map.get(ext, "unknown")

    async def get_or_create_lsp_client(
        self, language: str, workspace_path: str
    ) -> Optional[LSPClient]:
        """Get or create LSP client for a language"""
        if language in self.lsp_clients:
            return self.lsp_clients[language]

        if language not in self.language_server_configs:
            print(f"No LSP server config for {language}")
            return None

        try:
            command = self.language_server_configs[language][0]
            args = self.language_server_configs[language][1:]

            client = LSPClient(command, args, workspace_path)
            await client.start()
            await client.initialize(workspace_path)

            self.lsp_clients[language] = client
            print(f"✅ Initialized LSP client for {language}")
            return client

        except Exception as e:
            print(f"❌ Failed to initialize LSP client for {language}: {e}")
            return None

    def get_code_from_symbols(self, d, path, language):
        if language == "rust":
            file_path = path
            start_line = d.get("range").get("start").get("line")
            start_char = d.get("range").get("start").get("character")
            end_line = d.get("range").get("end").get("line")
            end_char = d.get("range").get("end").get("character")
        else:
            file_path = d.get("location").get("uri")
            start_line = d.get("location").get("range").get("start").get("line")
            start_char = d.get("location").get("range").get("start").get("character")
            end_line = d.get("location").get("range").get("end").get("line")
            end_char = d.get("location").get("range").get("end").get("character")

        with open(file_path.replace("file:///", "/"), "r") as f:
            lines = f.readlines()

        # Adjust lines because lists are 0-indexed
        selected_lines = lines[start_line : end_line + 1]

        # Trim the first and last lines by character indices
        if selected_lines:
            selected_lines[0] = selected_lines[0][start_char:]
            selected_lines[-1] = selected_lines[-1][:end_char]

        return ("".join(selected_lines), start_line, start_char, end_line, end_char)

    async def extract_module_from_file(self, file_path: str, workspace_path: str):
        """Extract a single module from a file using LSP"""
        try:
            language = self.detect_language(file_path)
            if language == "unknown":
                print(f"Unknown language for {file_path}")
                return None

            # Get or create LSP client
            lsp_client = await self.get_or_create_lsp_client(language, workspace_path)
            if not lsp_client:
                print(f"No LSP client available for {language}")
                return None

            # Get document symbols
            symbols = await lsp_client.get_document_symbols(file_path)
            for i in symbols:
                (i["code_string"], l1, c1, l2, c2) = self.get_code_from_symbols(
                    i, file_path, language
                )
                res = await lsp_client.get_references(
                    file_path, l1, c1, include_declaration=True
                )
                i["references"] = [i for i in res]
                i["kind"] = SYMBOL_KIND_MAP[i.get("kind")]

            return symbols

        except Exception as e:
            print(f"Error extracting from {file_path}: {e}")
            traceback.print_exc()
            return None

    async def extract_codebase(self, root_path: str):
        """Extract entire codebase from a directory"""
        try:
            file_patterns = [
                "*.py",
                "*.js",
                "*.ts",
                "*.java",
                "*.cpp",
                "*.c",
                "*.cs",
                "*.go",
                "*.rs",
            ]

            root_path_obj = Path(root_path)
            files_to_process = []
            codebase = []
            for pattern in file_patterns:
                files_to_process.extend(root_path_obj.rglob(pattern))
            # Filter out files in common ignore directories
            ignore_dirs = {
                ".git",
                "__pycache__",
                ".pytest_cache",
                "node_modules",
                ".venv",
                "venv",
                "env",
                "target",
                "build",
                "dist",
            }

            filtered_files = []
            for file_path in files_to_process:
                if file_path.is_file():
                    # Check if any parent directory is in ignore list
                    if not any(part in ignore_dirs for part in file_path.parts):
                        filtered_files.append(file_path)

            # Group files by language
            language_files = {}
            for file_path in filtered_files:
                language = self.detect_language(str(file_path))
                if language != "unknown":
                    if language not in language_files:
                        language_files[language] = []
                    language_files[language].append(file_path)

            print(f"Found files in languages: {list(language_files.keys())}")

            # Process files by language
            for language, files in language_files.items():
                print(f"Processing {len(files)} {language} files...")

                # Initialize LSP for this language
                lsp_client = await self.get_or_create_lsp_client(
                    language, str(root_path_obj.absolute())
                )
                if not lsp_client:
                    print(
                        f"Skipping {language} files due to LSP initialization failure"
                    )
                    continue

                # Process files
                successful = 0
                for file_path in files:
                    try:
                        print(file_path)
                        module = await self.extract_module_from_file(
                            str(file_path), str(root_path_obj.absolute())
                        )
                        if module:
                            codebase.append(module)
                            successful += 1
                            print(f"  ✅ {file_path.name}")
                        else:
                            print(f"  ❌ {file_path.name}")
                    except Exception as e:
                        print(f"  ❌ {file_path.name}: {e}")

                print(
                    f"Successfully processed {successful}/{len(files)} {language} files"
                )
            return codebase
        finally:
            # Always clean up LSP clients
            await self.cleanup()


async def extract_codebase_simple(root_path: str, output_file: str = None):
    """Simple synchronous function to extract codebase"""
    extractor = AsyncUniversalExtractor()
    try:
        codebase = await extractor.extract_codebase(root_path)
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(codebase, f, indent=2, default=str)
            print(f"Codebase model saved to {output_file}")
        return codebase
    except Exception as e:
        print(f"Error during extraction: {e}")
        # Ensure cleanup happens even if there's an error
        await extractor.cleanup()
        raise


codebase = ("/Users/eswar.tadiparth/Documents/open-source/fdep-rs/src",)
asyncio.run(extract_codebase_simple(codebase[0], "rust.json"))
