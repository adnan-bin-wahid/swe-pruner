import logging
from typing import List, Set
from ..repository.repository_index import RepositoryIndex

logger = logging.getLogger(__name__)

class LexicalRetriever:
    def __init__(self, index: RepositoryIndex):
        self.index = index

    def search_identifiers(self, identifiers: List[str]) -> Set[str]:
        """
        Scans class and function names in the AST index to find exact matches 
        with the specified list of identifiers.
        """
        matched_files = set()
        for rel_path, file_meta in self.index.index.items():
            for ident in identifiers:
                # Direct check on class definitions
                if ident in file_meta.get("classes", {}):
                    matched_files.add(rel_path)
                    logger.debug(f"Lexical match for class '{ident}' in {rel_path}")
                
                # Direct check on function definitions
                elif ident in file_meta.get("functions", {}):
                    matched_files.add(rel_path)
                    logger.debug(f"Lexical match for function '{ident}' in {rel_path}")
                    
        return matched_files
