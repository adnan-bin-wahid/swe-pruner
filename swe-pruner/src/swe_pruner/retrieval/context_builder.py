import logging
from typing import List, Dict, Any, Tuple
from ..prune_wrapper import prune_code_lines, aggregate_token_scores_to_lines, PruneRequest

logger = logging.getLogger(__name__)

class ContextBuilder:
    def __init__(self, token_budget: int = 8192):
        self.token_budget = token_budget

    def pack_context(
        self, 
        query: str,
        files_metadata: Dict[str, Dict[str, Any]], 
        graph_distances: Dict[str, int],
        ranked_scores: List[Tuple[str, float]],
        pruner_model: Any,
        threshold: float = 0.45
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Assembles the final pruned repository context.
        Allocates token budgets across three levels:
        - Tier 1: Target File / Distance 0 (light pruning)
        - Tier 2: Direct dependency (aggressive pruning + signature)
        - Tier 3: Transitive reference (signature only)
        """
        output_blocks = []
        file_summary_results = []
        
        # Determine tiers
        # Create map of scores for easier lookup
        score_map = {path: score for path, score in ranked_scores}
        
        for rel_path, file_meta in files_metadata.items():
            content = file_meta.get("content", "")
            distance = graph_distances.get(rel_path, 2)
            score = score_map.get(rel_path, -9.0)
            
            # Determine Tier
            if distance == 0:
                tier = 1
                rel_label = "active file"
            elif distance == 1 and score > -2.0:  # direct dependency with reasonable relevance
                tier = 2
                rel_label = "direct dependency"
            else:
                tier = 3
                rel_label = "transitive reference"
                
            # If the file contains tests, mark relation accordingly
            is_test_file = "test" in rel_path.lower()
            if is_test_file:
                rel_label = "related test"

            original_lines = content.splitlines()
            original_token_count = len(content.split()) # estimate
            
            print(f"ContextBuilder: processing {rel_path} (tier: {tier}, lines: {len(original_lines)})...", flush=True)
            pruned_content = ""
            
            if tier == 1:
                # Tier 1: Full/lightly pruned body
                # Use standard pruner with lenient threshold
                try:
                    req = PruneRequest(query=query, code=content, threshold=max(0.1, threshold - 0.15))
                    prune_res = pruner_model.prune(req)
                    pruned_content = prune_res.pruned_code
                    final_token_count = prune_res.left_token_cnt
                except Exception as e:
                    logger.error(f"Failed to prune Tier 1 file {rel_path}: {e}")
                    pruned_content = content
                    final_token_count = original_token_count
                    
            elif tier == 2:
                # Tier 2: Signature + aggressively pruned body
                # We'll extract class / function signatures, keep them, and heavily prune methods
                try:
                    req = PruneRequest(query=query, code=content, threshold=min(0.85, threshold + 0.15))
                    prune_res = pruner_model.prune(req)
                    pruned_content = prune_res.pruned_code
                    final_token_count = prune_res.left_token_cnt
                except Exception as e:
                    # Fallback to simple first-10-lines signature + body
                    logger.error(f"Failed to prune Tier 2 file {rel_path}: {e}")
                    pruned_content = "\n".join(original_lines[:10]) + "\n... (body truncated)"
                    final_token_count = len(pruned_content.split())

            else:
                # Tier 3: Signatures only (Classes and function declarations only)
                # Walk the index classes and methods
                signatures = []
                for class_name, cls_info in file_meta.get("classes", {}).items():
                    signatures.append(f"class {class_name}:")
                    for method in cls_info.get("methods", []):
                        signatures.append(f"    def {method}(self, ...): ...")
                for func_name in file_meta.get("functions", {}):
                    signatures.append(f"def {func_name}(...): ...")
                
                if not signatures:
                    # Simple fallback
                    pruned_content = f"# Module {rel_path} interface reference"
                else:
                    pruned_content = "\n".join(signatures)
                final_token_count = len(pruned_content.split())

            print(f"ContextBuilder: completed {rel_path}. Pruned tokens: {final_token_count}", flush=True)
            # Format block with headers and metadata
            header = f"### {rel_path}\n# Relation: {rel_label}\n# Tier: {tier} (original lines: 1-{len(original_lines)})\n"
            block = f"{header}```python\n{pruned_content}\n```"
            
            output_blocks.append(block)
            file_summary_results.append({
                "file_path": rel_path,
                "relation": rel_label,
                "tier": tier,
                "original_tokens": original_token_count,
                "pruned_tokens": final_token_count,
                "score": score
            })

        unified_prompt = "\n\n".join(output_blocks)
        return unified_prompt, file_summary_results
