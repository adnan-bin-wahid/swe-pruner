---
license: mit
library_name: transformers
base_model: Qwen/Qwen3-Reranker-0.6B
pipeline_tag: token-classification
tags:
- code
- context-pruning
- agent
datasets:
- nick007x/github-code-2025
metrics:
- f1
- mse
---

# SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents

SWE-Pruner is a self-adaptive context pruning framework specifically designed for coding agents. It addresses the challenges of long interaction contexts, such as high API costs and latency, by performing task-aware adaptive pruning.

- **Paper:** [SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents](https://huggingface.co/papers/2601.16746)
- **Repository:** [https://github.com/Ayanami1314/swe-pruner](https://github.com/Ayanami1314/swe-pruner)

## Description
Inspired by how human programmers selectively skim code, SWE-Pruner enables agents to formulate explicit goals (e.g., "focus on error handling") which guide a lightweight neural skimmer (0.6B parameters). This skimmer dynamically selects relevant lines from the surrounding context, preserving critical implementation details while significantly reducing token usage.

Evaluations across benchmarks show that SWE-Pruner achieves 23-54% token reduction on agent tasks like SWE-Bench Verified and up to 14.84x compression on single-turn tasks like LongCodeQA with minimal performance impact.

## Model Usage
Given that we have made significant modifications to the model, its dual-head architecture and the complex compression head service code will be rather complex. 
Therefore, we recommend that you use the version we have released on [GitHub](https://github.com/Ayanami1314/swe-pruner) instead of attempting to use the original model on your own.


## Citation
If you find SWE-Pruner useful in your research, please cite:
```bibtex
@misc{wang2026sweprunerselfadaptivecontextpruning,
      title={SWE-Pruner: Self-Adaptive Context Pruning for Coding Agents}, 
      author={Yuhang Wang and Yuling Shi and Mo Yang and Rongrui Zhang and Shilin He and Heng Lian and Yuting Chen and Siyu Ye and Kai Cai and Xiaodong Gu},
      year={2026},
      eprint={2601.16746},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2601.16746},
}
```