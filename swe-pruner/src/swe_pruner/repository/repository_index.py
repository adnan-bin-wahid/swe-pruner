import os
import logging
from pathlib import Path
from typing import Dict, Any
from .python_indexer import PythonASTIndexer

logger = logging.getLogger(__name__)

class RepositoryIndex:
    def __init__(self, workspace_root: str):
        self.workspace_root = Path(workspace_root).resolve()
        self.indexer = PythonASTIndexer()
        self.index: Dict[str, Any] = {}

    def build_index(self):
        """Walks the repository and indexes all Python files."""
        logger.info(f"Building repository AST index for {self.workspace_root}...")
        self.index.clear()
        
        exclude_dirs = {
            '.git', '.venv', 'venv', 'node_modules', '__pycache__', 
            '.vscode', '.idea', 'build', 'dist', 'carbon_artifacts'
        }

        for root, dirs, files in os.walk(self.workspace_root):
            # Modify dirs in-place to avoid traversing excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    full_path = Path(root) / file
                    try:
                        rel_path = str(full_path.relative_to(self.workspace_root)).replace('\\', '/')
                        file_meta = self.indexer.index_file(full_path)
                        file_meta["content"] = full_path.read_text(encoding='utf-8')
                        self.index[rel_path] = file_meta
                    except Exception as e:
                        logger.error(f"Error indexing {full_path}: {e}")
                        
        logger.info(f"Repository indexing complete. Total files indexed: {len(self.index)}")
