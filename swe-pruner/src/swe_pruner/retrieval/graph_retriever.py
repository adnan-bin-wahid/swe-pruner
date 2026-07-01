import logging
from typing import Set, Dict
from ..repository.dependency_graph import DependencyGraph

logger = logging.getLogger(__name__)

class GraphRetriever:
    def __init__(self, dependency_graph: DependencyGraph):
        self.graph = dependency_graph

    def get_neighbors(self, seed_files: Set[str], max_hops: int = 1) -> Dict[str, int]:
        """
        Traverses the dependency/call graph starting from a set of seed files.
        Returns a dictionary mapping relative file paths to their minimum distance (hops) 
        from the seed set.
        """
        logger.info(f"Expanding candidate set from {len(seed_files)} seeds using call graph...")
        # Map relative path to distance from seed (0 means it is a seed itself)
        distances: Dict[str, int] = {f: 0 for f in seed_files}
        
        current_layer = set(seed_files)
        
        for hop in range(1, max_hops + 1):
            next_layer = set()
            for file in current_layer:
                # Add outgoing dependencies (imports, callees)
                if file in self.graph.dependencies:
                    for dep in self.graph.dependencies[file]:
                        if dep not in distances:
                            distances[dep] = hop
                            next_layer.add(dep)
                            
                # Add incoming dependencies (callers, dependents)
                if file in self.graph.dependents:
                    for dep in self.graph.dependents[file]:
                        if dep not in distances:
                            distances[dep] = hop
                            next_layer.add(dep)
                            
            current_layer = next_layer
            if not current_layer:
                break
                
        logger.info(f"Graph expansion complete. Total files in ego-graph: {len(distances)}")
        return distances
