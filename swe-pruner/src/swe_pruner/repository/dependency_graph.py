import logging
from typing import Dict, Set, List
from .repository_index import RepositoryIndex

logger = logging.getLogger(__name__)

class DependencyGraph:
    def __init__(self, repo_index: RepositoryIndex):
        self.repo_index = repo_index
        # Adjacency list: maps file_path to set of imported/dependent file_paths
        self.dependencies: Dict[str, Set[str]] = {}
        # Reverse mapping: maps file_path to set of files that import/depend on it
        self.dependents: Dict[str, Set[str]] = {}
        # Maps symbol name to list of file paths defining it
        self.symbol_definitions: Dict[str, List[str]] = {}

    def build_graph(self):
        """
        Builds the import/call dependency graph using the RepositoryIndex.
        """
        logger.info("Building dependency graph...")
        self.dependencies.clear()
        self.dependents.clear()
        self.symbol_definitions.clear()

        # Initialize collections
        for rel_path in self.repo_index.index:
            self.dependencies[rel_path] = set()
            self.dependents[rel_path] = set()
            
            # Map symbol definitions
            file_meta = self.repo_index.index[rel_path]
            for class_name in file_meta.get("classes", {}):
                self.symbol_definitions.setdefault(class_name, []).append(rel_path)
            for func_name in file_meta.get("functions", {}):
                self.symbol_definitions.setdefault(func_name, []).append(rel_path)

        # Resolve dependency edges (imports and function/method calls)
        for rel_path, file_meta in self.repo_index.index.items():
            # 1. Resolve direct imports
            for imp in file_meta.get("imports", []):
                # Try matching import name to repository files
                # e.g., if import is 'auth.service', matching path might be 'auth/service.py'
                imp_parts = imp.split('.')
                for target_path in self.repo_index.index:
                    target_parts = target_path.replace('.py', '').split('/')
                    # Match suffix, e.g., target 'src/auth/service.py' -> ['src', 'auth', 'service']
                    if len(imp_parts) <= len(target_parts):
                        if target_parts[-len(imp_parts):] == imp_parts:
                            self._add_edge(rel_path, target_path)

            # 2. Resolve calls to global/class symbols defined elsewhere in the repo
            for call_symbol in file_meta.get("calls", []):
                if call_symbol in self.symbol_definitions:
                    for target_path in self.symbol_definitions[call_symbol]:
                        if target_path != rel_path:
                            self._add_edge(rel_path, target_path)

        logger.info("Dependency graph built successfully.")

    def _add_edge(self, source: str, target: str):
        if source in self.dependencies:
            self.dependencies[source].add(target)
        if target in self.dependents:
            self.dependents[target].add(source)
