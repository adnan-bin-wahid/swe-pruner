import torch
import logging
from typing import List, Tuple
from ..swepruner import SwePrunerForCodeCompression

logger = logging.getLogger(__name__)

class CandidateRanker:
    def __init__(self, pruner_model: SwePrunerForCodeCompression):
        self.model = pruner_model

    def rank_candidates(self, query: str, candidates: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """
        Reranks a list of candidate files by querying the pre-loaded 
        Qwen3-Reranker-0.6B model's score_logits output.
        
        Args:
            query: The structured goal synthesis string
            candidates: List of (file_path, file_content)
            
        Returns:
            List of (file_path, relevance_score) sorted descending by relevance.
        """
        ranked = []
        # Fallback if model is not loaded (safety safeguard)
        if not self.model:
            logger.warning("Reranker model is not loaded. Falling back to uniform score mapping.")
            return [(path, 0.5) for path, _ in candidates]

        self.model.eval()
        
        # Determine device
        device = next(self.model.parameters()).device

        for path, content in candidates:
            print(f"CandidateRanker: scoring {path} (length: {len(content)})...", flush=True)
            # Clean/truncate content to prevent out-of-memory or model constraint exceptions
            # LLMs generally accept up to 8192 context; we take first 12000 chars for scoring
            truncated_content = content[:12000]
            
            # Format using standard instruction query template
            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            instruction_text = f"<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {truncated_content}"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            
            try:
                inputs = self.model.tokenizer(
                    prefix + instruction_text + suffix,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                )
                
                # Move tensors to model device
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    # score_logits corresponds to the probability of generation 'yes' vs 'no'
                    score = float(outputs.score_logits[0].cpu().numpy())
                    ranked.append((path, score))
                print(f"CandidateRanker: completed scoring for {path}. Score: {score}", flush=True)
            except Exception as e:
                logger.error(f"Error scoring candidate {path}: {e}")
                ranked.append((path, -99.0)) # low score on failure
                
        # Sort descending by relevance score
        return sorted(ranked, key=lambda x: x[1], reverse=True)
