import asyncio
import os
import json
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import logging

from lsp_client import LSPClient, LSPTools, Position, Range, Location

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Represents a node in the dependency graph"""

    id: str  # Unique identifier
    name: str  # Display name
    kind: str  # Symbol kind (class, function, method, etc.)
    file_path: str  # Source file path
    line: int  # Line number (1-based)
    character: int  # Character position (0-based)
    namespace: Optional[str] = None  # Namespace/module
    parent: Optional[str] = None  # Parent node ID
    metadata: Dict[str, Any] = None  # Additional metadata

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Edge:
    """Represents an edge (dependency) in the graph"""

    source: str  # Source node ID
    target: str  # Target node ID
    relationship: str  # Type of relationship
    weight: int = 1  # Weight/strength of relationship
    metadata: Dict[str, Any] = None  # Additional metadata

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DependencyGraph:
    """Main dependency graph container"""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.relationships: Dict[str, List[Edge]] = defaultdict(list)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph"""
        self.edges.append(edge)
        self.relationships[edge.source].append(edge)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID"""
        return self.nodes.get(node_id)

    def get_dependencies(self, node_id: str) -> List[Edge]:
        """Get all dependencies (outgoing edges) for a node"""
        return self.relationships.get(node_id, [])

    def get_dependents(self, node_id: str) -> List[Edge]:
        """Get all dependents (incoming edges) for a node"""
        return [edge for edge in self.edges if edge.target == node_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization"""
        return {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "edges": [asdict(edge) for edge in self.edges],
            "stats": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": self._get_node_type_stats(),
                "relationship_types": self._get_relationship_type_stats(),
            },
        }

    def _get_node_type_stats(self) -> Dict[str, int]:
        """Get statistics about node types"""
        stats = defaultdict(int)
        for node in self.nodes.values():
            stats[node.kind] += 1
        return dict(stats)

    def _get_relationship_type_stats(self) -> Dict[str, int]:
        """Get statistics about relationship types"""
        stats = defaultdict(int)
        for edge in self.edges:
            stats[edge.relationship] += 1
        return dict(stats)


class DependencyAnalyzer:
    """Analyzes code dependencies using LSP client"""

    def __init__(self, lsp_client: LSPClient):
        self.client = lsp_client
        self.tools = LSPTools(lsp_client)
        self.graph = DependencyGraph()
        self.processed_files: Set[str] = set()
        self.symbol_cache: Dict[str, Dict] = {}

    async def analyze_workspace(
        self, workspace_path: str, file_patterns: List[str] = None
    ) -> DependencyGraph:
        """Analyze entire workspace and build dependency graph"""
        logger.info(f"Starting dependency analysis for workspace: {workspace_path}")

        # Default patterns for common file types
        if file_patterns is None:
            file_patterns = [
                "*.py",
                "*.js",
                "*.ts",
                "*.go",
                "*.rs",
                "*.java",
                "*.c",
                "*.cpp",
                "*.h",
                "*.hpp",
            ]

        # Find all source files
        source_files = self._find_source_files(workspace_path, file_patterns)
        logger.info(f"Found {len(source_files)} source files to analyze")

        # Phase 1: Collect all symbols and create nodes
        await self._collect_symbols_phase(source_files)

        # Phase 2: Analyze relationships and create edges
        await self._analyze_relationships_phase(source_files)

        # Phase 3: Enhance graph with additional analysis
        await self._enhance_graph_phase()

        logger.info(
            f"Analysis complete: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges"
        )
        return self.graph

    def _find_source_files(self, workspace_path: str, patterns: List[str]) -> List[str]:
        """Find all source files matching the patterns"""
        source_files = []
        workspace_path = Path(workspace_path)

        for pattern in patterns:
            files = workspace_path.rglob(pattern)
            for file_path in files:
                if file_path.is_file():
                    # Skip common non-source directories
                    path_str = str(file_path)
                    if not any(
                        skip in path_str
                        for skip in [
                            ".git",
                            "__pycache__",
                            "node_modules",
                            "target",
                            "build",
                        ]
                    ):
                        source_files.append(str(file_path))

        return sorted(source_files)

    async def _collect_symbols_phase(self, source_files: List[str]) -> None:
        """Phase 1: Collect all symbols and create nodes"""
        logger.info("Phase 1: Collecting symbols...")

        for file_path in source_files:
            try:
                await self._process_file_symbols(file_path)
                await asyncio.sleep(
                    0.01
                )  # Small delay to prevent overwhelming the LSP server
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")

    async def _process_file_symbols(self, file_path: str) -> None:
        """Process symbols in a single file"""
        if file_path in self.processed_files:
            return

        try:
            # Open file in LSP server
            await self.client.did_open(file_path)

            # Get document symbols
            symbols = await self.client.get_document_symbols(file_path)

            # Process symbols recursively
            await self._process_symbols_recursive(symbols, file_path)

            self.processed_files.add(file_path)

        except Exception as e:
            logger.warning(f"Error getting symbols for {file_path}: {e}")

    async def _process_symbols_recursive(
        self,
        symbols: List[Dict],
        file_path: str,
        parent_id: str = None,
        namespace: str = None,
    ) -> None:
        """Process symbols recursively to handle nested symbols"""
        for symbol in symbols:
            try:
                node = await self._create_node_from_symbol(
                    symbol, file_path, parent_id, namespace
                )
                if node:
                    self.graph.add_node(node)
                    self.symbol_cache[node.id] = symbol

                    # Process children if they exist
                    children = symbol.get("children", [])
                    if children:
                        child_namespace = (
                            f"{namespace}.{node.name}" if namespace else node.name
                        )
                        await self._process_symbols_recursive(
                            children, file_path, node.id, child_namespace
                        )

            except Exception as e:
                logger.warning(
                    f"Error processing symbol {symbol.get('name', 'unknown')}: {e}"
                )

    async def _create_node_from_symbol(
        self, symbol: Dict, file_path: str, parent_id: str = None, namespace: str = None
    ) -> Optional[Node]:
        """Create a node from an LSP symbol"""
        try:
            name = symbol.get("name", "unknown")
            kind = self._normalize_symbol_kind(symbol.get("kind", "unknown"))

            # Get position information
            location = symbol.get("location", {})
            if "range" in location:
                range_info = location["range"]
                line = range_info["start"]["line"] + 1  # Convert to 1-based
                character = range_info["start"]["character"]
            else:
                # Fallback for symbols without location
                range_info = symbol.get("range", {})
                if "start" in range_info:
                    line = range_info["start"]["line"] + 1
                    character = range_info["start"]["character"]
                else:
                    line = 1
                    character = 0

            # Create unique ID
            node_id = self._create_node_id(file_path, name, kind, line, namespace)

            # Extract additional metadata
            metadata = {
                "detail": symbol.get("detail", ""),
                "deprecated": symbol.get("deprecated", False),
                "tags": symbol.get("tags", []),
                "selection_range": symbol.get("selectionRange"),
                "symbol_info": symbol,
            }

            return Node(
                id=node_id,
                name=name,
                kind=kind,
                file_path=file_path,
                line=line,
                character=character,
                namespace=namespace,
                parent=parent_id,
                metadata=metadata,
            )

        except Exception as e:
            logger.warning(f"Error creating node from symbol: {e}")
            return None

    def _normalize_symbol_kind(self, kind: Any) -> str:
        """Normalize symbol kind to string"""
        # LSP symbol kinds (numbers to strings)
        kind_map = {
            1: "file",
            2: "module",
            3: "namespace",
            4: "package",
            5: "class",
            6: "method",
            7: "property",
            8: "field",
            9: "constructor",
            10: "enum",
            11: "interface",
            12: "function",
            13: "variable",
            14: "constant",
            15: "string",
            16: "number",
            17: "boolean",
            18: "array",
            19: "object",
            20: "key",
            21: "null",
            22: "enum_member",
            23: "struct",
            24: "event",
            25: "operator",
            26: "type_parameter",
        }

        if isinstance(kind, int):
            return kind_map.get(kind, f"unknown_{kind}")
        return str(kind).lower()

    def _create_node_id(
        self, file_path: str, name: str, kind: str, line: int, namespace: str = None
    ) -> str:
        """Create a unique node ID"""
        # Use relative path for cleaner IDs
        rel_path = os.path.relpath(file_path)
        if namespace:
            return f"{rel_path}::{namespace}.{name}:{kind}:{line}"
        else:
            return f"{rel_path}::{name}:{kind}:{line}"

    async def _analyze_relationships_phase(self, source_files: List[str]) -> None:
        """Phase 2: Analyze relationships between symbols"""
        logger.info("Phase 2: Analyzing relationships...")

        for file_path in source_files:
            try:
                await self._analyze_file_relationships(file_path)
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.warning(f"Error analyzing relationships in {file_path}: {e}")

    async def _analyze_file_relationships(self, file_path: str) -> None:
        """Analyze relationships in a single file"""
        try:
            # Get all nodes from this file
            file_nodes = [
                node
                for node in self.graph.nodes.values()
                if node.file_path == file_path
            ]

            for node in file_nodes:
                await self._analyze_node_relationships(node)

        except Exception as e:
            logger.warning(f"Error analyzing relationships in {file_path}: {e}")

    async def _analyze_node_relationships(self, node: Node) -> None:
        """Analyze relationships for a specific node"""
        try:
            # Get references to find usage relationships
            references = await self.client.get_references(
                node.file_path, node.line - 1, node.character
            )

            for ref in references:
                ref_file = self.client.uri_to_path(ref.uri)
                ref_line = ref.range.start.line + 1

                # Find the node that contains this reference
                referring_node = self._find_node_at_position(
                    ref_file, ref_line, ref.range.start.character
                )

                if referring_node and referring_node.id != node.id:
                    # Create "uses" relationship
                    edge = Edge(
                        source=referring_node.id,
                        target=node.id,
                        relationship="uses",
                        metadata={
                            "ref_location": {
                                "file": ref_file,
                                "line": ref_line,
                                "character": ref.range.start.character,
                            }
                        },
                    )
                    self.graph.add_edge(edge)

            # Analyze inheritance and other specific relationships
            await self._analyze_specific_relationships(node)

        except Exception as e:
            logger.warning(f"Error analyzing relationships for node {node.id}: {e}")

    def _find_node_at_position(
        self, file_path: str, line: int, character: int
    ) -> Optional[Node]:
        """Find the node that contains the given position"""
        # Find all nodes in the file
        file_nodes = [
            node for node in self.graph.nodes.values() if node.file_path == file_path
        ]

        # Sort by line and find the best match
        file_nodes.sort(
            key=lambda n: (n.line, -len(n.name))
        )  # Prefer more specific names

        best_match = None
        for node in file_nodes:
            if node.line <= line:
                best_match = node
            else:
                break

        return best_match

    async def _analyze_specific_relationships(self, node: Node) -> None:
        """Analyze specific relationship types based on node kind"""
        try:
            if node.kind == "class":
                await self._analyze_class_relationships(node)
            elif node.kind == "method":
                await self._analyze_method_relationships(node)
            elif node.kind == "function":
                await self._analyze_function_relationships(node)
            elif node.kind == "module":
                await self._analyze_module_relationships(node)

        except Exception as e:
            logger.warning(f"Error analyzing specific relationships for {node.id}: {e}")

    async def _analyze_class_relationships(self, node: Node) -> None:
        """Analyze class-specific relationships like inheritance"""
        try:
            # Try to get hover information to find inheritance info
            hover = await self.client.get_hover(
                node.file_path, node.line - 1, node.character
            )

            if hover and "contents" in hover:
                content = str(hover["contents"])
                # Look for inheritance patterns (this is language-specific)
                await self._extract_inheritance_from_hover(node, content)

        except Exception as e:
            logger.debug(f"Could not analyze class relationships for {node.id}: {e}")

    async def _extract_inheritance_from_hover(
        self, node: Node, hover_content: str
    ) -> None:
        """Extract inheritance information from hover content"""
        # This is a simplified example - would need language-specific parsing
        inheritance_keywords = ["extends", "inherits", "implements", ":", "->"]

        for keyword in inheritance_keywords:
            if keyword in hover_content.lower():
                # This would need more sophisticated parsing
                # For now, just mark that inheritance was detected
                node.metadata["has_inheritance"] = True
                break

    async def _analyze_method_relationships(self, node: Node) -> None:
        """Analyze method-specific relationships"""
        # Methods belong to classes - this relationship should be captured by parent
        if node.parent:
            parent_node = self.graph.get_node(node.parent)
            if parent_node and parent_node.kind == "class":
                edge = Edge(
                    source=node.id,
                    target=parent_node.id,
                    relationship="member_of",
                    metadata={"member_type": "method"},
                )
                self.graph.add_edge(edge)

    async def _analyze_function_relationships(self, node: Node) -> None:
        """Analyze function-specific relationships"""
        # Look for function calls within this function
        # This would require more detailed AST analysis
        pass

    async def _analyze_module_relationships(self, node: Node) -> None:
        """Analyze module-specific relationships like imports"""
        try:
            # Get all symbols in the module to find imports
            symbols = await self.client.get_document_symbols(node.file_path)

            # Look for import-like symbols
            for symbol in symbols:
                if symbol.get("name", "").startswith(
                    "import"
                ) or "import" in symbol.get("detail", ""):
                    # Create import relationship
                    imported_name = symbol.get("name", "").replace("import ", "")
                    if imported_name:
                        # Try to find the imported module
                        imported_node = self._find_node_by_name(imported_name, "module")
                        if imported_node:
                            edge = Edge(
                                source=node.id,
                                target=imported_node.id,
                                relationship="imports",
                                metadata={"import_type": "module"},
                            )
                            self.graph.add_edge(edge)

        except Exception as e:
            logger.debug(f"Could not analyze module relationships for {node.id}: {e}")

    def _find_node_by_name(self, name: str, kind: str = None) -> Optional[Node]:
        """Find a node by name and optionally kind"""
        for node in self.graph.nodes.values():
            if node.name == name and (kind is None or node.kind == kind):
                return node
        return None

    async def _enhance_graph_phase(self) -> None:
        """Phase 3: Enhance graph with additional analysis"""
        logger.info("Phase 3: Enhancing graph...")

        # Add parent-child relationships
        self._add_parent_child_relationships()

        # Calculate metrics
        self._calculate_node_metrics()

        # Detect patterns
        self._detect_design_patterns()

    def _add_parent_child_relationships(self) -> None:
        """Add explicit parent-child relationships"""
        for node in self.graph.nodes.values():
            if node.parent:
                edge = Edge(
                    source=node.id,
                    target=node.parent,
                    relationship="child_of",
                    metadata={"hierarchy_type": "containment"},
                )
                self.graph.add_edge(edge)

    def _calculate_node_metrics(self) -> None:
        """Calculate various metrics for nodes"""
        for node in self.graph.nodes.values():
            # Count dependencies and dependents
            dependencies = len(self.graph.get_dependencies(node.id))
            dependents = len(self.graph.get_dependents(node.id))

            node.metadata.update(
                {
                    "dependency_count": dependencies,
                    "dependent_count": dependents,
                    "complexity_score": dependencies + dependents,
                }
            )

    def _detect_design_patterns(self) -> None:
        """Detect common design patterns"""
        # This is a simplified pattern detection
        # Real implementation would need more sophisticated analysis

        # Detect singleton pattern
        for node in self.graph.nodes.values():
            if node.kind == "class":
                methods = [
                    n
                    for n in self.graph.nodes.values()
                    if n.parent == node.id and n.kind == "method"
                ]
                method_names = [m.name for m in methods]

                if "getInstance" in method_names or "get_instance" in method_names:
                    node.metadata["patterns"] = node.metadata.get("patterns", []) + [
                        "singleton"
                    ]


class GraphVisualizer:
    """Visualizes dependency graphs"""

    def __init__(self, graph: DependencyGraph):
        self.graph = graph

    def to_graphviz(
        self, output_file: str, filter_relationships: List[str] = None
    ) -> str:
        """Export graph to Graphviz DOT format"""
        dot_content = ["digraph DependencyGraph {"]
        dot_content.append("  rankdir=TB;")
        dot_content.append("  node [shape=box, style=filled];")

        # Define colors for different node types
        colors = {
            "module": "lightblue",
            "class": "lightgreen",
            "method": "lightyellow",
            "function": "lightcoral",
            "variable": "lightgray",
            "constant": "wheat",
        }

        # Add nodes
        for node in self.graph.nodes.values():
            color = colors.get(node.kind, "white")
            label = f"{node.name}\\n({node.kind})"
            dot_content.append(f'  "{node.id}" [label="{label}", fillcolor="{color}"];')

        # Add edges
        for edge in self.graph.edges:
            if (
                filter_relationships is None
                or edge.relationship in filter_relationships
            ):
                style = self._get_edge_style(edge.relationship)
                dot_content.append(
                    f'  "{edge.source}" -> "{edge.target}" [label="{edge.relationship}"{style}];'
                )

        dot_content.append("}")

        # Write to file
        with open(output_file, "w") as f:
            f.write("\n".join(dot_content))

        return "\n".join(dot_content)

    def _get_edge_style(self, relationship: str) -> str:
        """Get GraphViz style for different relationship types"""
        styles = {
            "uses": ", color=blue",
            "imports": ", color=green, style=dashed",
            "inherits": ", color=red, style=bold",
            "member_of": ", color=purple, style=dotted",
            "child_of": ", color=orange",
        }
        return styles.get(relationship, "")

    def to_json(self, output_file: str) -> Dict[str, Any]:
        """Export graph to JSON format"""
        graph_data = self.graph.to_dict()

        with open(output_file, "w") as f:
            json.dump(graph_data, f, indent=2)

        return graph_data

    def generate_report(self, output_file: str) -> str:
        """Generate a comprehensive text report"""
        report = []
        report.append("=== DEPENDENCY GRAPH ANALYSIS REPORT ===\n")

        # Summary statistics
        stats = self.graph.to_dict()["stats"]
        report.append("## Summary Statistics")
        report.append(f"Total Nodes: {stats['total_nodes']}")
        report.append(f"Total Edges: {stats['total_edges']}")
        report.append("")

        # Node type breakdown
        report.append("## Node Types")
        for node_type, count in stats["node_types"].items():
            report.append(f"  {node_type}: {count}")
        report.append("")

        # Relationship type breakdown
        report.append("## Relationship Types")
        for rel_type, count in stats["relationship_types"].items():
            report.append(f"  {rel_type}: {count}")
        report.append("")

        # Most connected nodes
        report.append("## Most Connected Nodes")
        nodes_by_connections = sorted(
            self.graph.nodes.values(),
            key=lambda n: n.metadata.get("complexity_score", 0),
            reverse=True,
        )

        for node in nodes_by_connections[:10]:
            deps = node.metadata.get("dependency_count", 0)
            dependents = node.metadata.get("dependent_count", 0)
            report.append(
                f"  {node.name} ({node.kind}): {deps} deps, {dependents} dependents"
            )
        report.append("")

        # Files with most symbols
        file_symbol_count = defaultdict(int)
        for node in self.graph.nodes.values():
            file_symbol_count[node.file_path] += 1

        report.append("## Files with Most Symbols")
        for file_path, count in sorted(
            file_symbol_count.items(), key=lambda x: x[1], reverse=True
        )[:10]:
            report.append(f"  {os.path.basename(file_path)}: {count} symbols")

        report_text = "\n".join(report)

        with open(output_file, "w") as f:
            f.write(report_text)

        return report_text


# Main analysis function
async def analyze_codebase(
    workspace_path: str,
    language_server_command: str,
    language_server_args: List[str] = None,
    output_dir: str = "dependency_analysis",
    file_patterns: List[str] = None,
) -> DependencyGraph:
    """
    Analyze a codebase and generate dependency graph

    Args:
        workspace_path: Path to the workspace/project root
        language_server_command: Command to start language server (e.g., 'pylsp', 'gopls')
        language_server_args: Additional arguments for language server
        output_dir: Directory to save analysis results
        file_patterns: File patterns to analyze (e.g., ['*.py', '*.go'])

    Returns:
        DependencyGraph: The analyzed dependency graph
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize LSP client
    async with LSPClient(language_server_command, language_server_args or []) as client:
        await client.initialize(workspace_path)

        # Create analyzer
        analyzer = DependencyAnalyzer(client)

        # Analyze workspace
        graph = await analyzer.analyze_workspace(workspace_path, file_patterns)

        # Generate outputs
        visualizer = GraphVisualizer(graph)

        # Export to various formats
        dot_file = os.path.join(output_dir, "dependency_graph.dot")
        json_file = os.path.join(output_dir, "dependency_graph.json")
        report_file = os.path.join(output_dir, "analysis_report.txt")

        visualizer.to_graphviz(dot_file)
        visualizer.to_json(json_file)
        visualizer.generate_report(report_file)

        logger.info(f"Analysis complete! Results saved to {output_dir}/")
        logger.info(f"  - GraphViz: {dot_file}")
        logger.info(f"  - JSON: {json_file}")
        logger.info(f"  - Report: {report_file}")

        return graph


if __name__ == "__main__":
    # Example usage
    async def main():
        # Analyze a Python project
        workspace = "/Users/eswar.tadiparth/Documents/open-source/lsp-2-graph/src/"
        graph = await analyze_codebase(
            workspace_path=workspace,
            language_server_command="pylsp",
            file_patterns=["*.py"],
            output_dir="python_analysis",
        )

        print(f"Analysis complete: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Run example
    asyncio.run(main())

    print("Dependency Graph Builder ready!")
    print("Usage:")
    print("  graph = await analyze_codebase('/path/to/project', 'pylsp')")
