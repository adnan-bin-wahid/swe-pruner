import os
# Force PyTorch, MKL, and OpenMP to run in single-threaded mode to prevent deadlocks in multithreaded web environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
import logging
import uvicorn
import typer
import torch
from pydantic import BaseModel

# Optimize PyTorch CPU execution threads for web server stability
torch.set_num_threads(1)
from .prune_wrapper import SwePrunerForCodePruning, PruneRequest, PruneResponse
from .carbon_estimator import (
    CarbonEstimateRequest,
    CarbonEstimateResponse,
    CarbonEstimator,
)
from .goal_models import StructuredGoal
from .goal_generator_client import LocalGoalGeneratorClient
from .goal_compiler import GoalCompiler
from .repository.repository_index import RepositoryIndex
from .repository.dependency_graph import DependencyGraph
from .retrieval.lexical_retriever import LexicalRetriever
from .retrieval.graph_retriever import GraphRetriever
from .retrieval.candidate_ranker import CandidateRanker
from .retrieval.context_builder import ContextBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Code Pruning Service")

# Global model and tokenizer
model: Optional[SwePrunerForCodePruning] = None
carbon_estimator = CarbonEstimator()

# Initialize Goal synthesizer components
generator_client = LocalGoalGeneratorClient()
goal_compiler = GoalCompiler(generator_client)

# Create Typer app
cli = typer.Typer(help="SwePruner code pruning service")

class WorkspacePruneRequest(BaseModel):
    query: str
    workspace_root: str
    active_file: str
    language: str
    current_symbol: Optional[str] = None
    selected_code: Optional[str] = None
    diagnostics: List[str] = []
    threshold: float = 0.45
    local_llm_url: Optional[str] = None
    local_llm_model: Optional[str] = None

class WorkspacePruneResponse(BaseModel):
    structured_goal: dict
    unified_prompt: str
    pruned_tokens: int
    original_tokens: int
    files: List[dict]

def check_model_path(model_path: str) -> bool:
    """Check if model directory exists and contains required files."""
    model_dir = Path(model_path)
    if not model_dir.exists():
        return False

    # Check for essential model files
    required_files = ["config.json", "model.safetensors"]
    for file in required_files:
        if not (model_dir / file).exists():
            return False
    return True


@app.on_event("startup")
async def startup_event():
    try:
        global model
        model_name_or_path = os.getenv("SWEPRUNER_MODEL_PATH", "./model")

        if not check_model_path(model_name_or_path):
            error_msg = (
                f"Model not found at {model_name_or_path}. "
                "The /prune endpoint will be disabled. /estimate-carbon is still available."
            )
            logger.warning(error_msg)
            model = None
        else:
            model = SwePrunerForCodePruning.from_pretrained(model_name_or_path)
            logger.info(f"Model loaded successfully from {model_name_or_path}")
    except Exception as e:
        logger.warning(f"Failed to load LLM model: {e}. The /prune endpoint will be disabled.")
        model = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/prune", response_model=PruneResponse)
async def prune_code(request: PruneRequest) -> PruneResponse | None:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    response = model.prune(request)
    return response


@app.post("/prune-workspace", response_model=WorkspacePruneResponse)
async def prune_workspace(request: WorkspacePruneRequest) -> WorkspacePruneResponse:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    workspace_root_path = Path(request.workspace_root).resolve()
    active_file_path = Path(request.active_file).resolve()
    try:
        active_rel_path = str(active_file_path.relative_to(workspace_root_path)).replace('\\', '/')
    except Exception:
        active_rel_path = request.active_file.replace('\\', '/')

    # 1. Synthesize structured goal
    goal = await goal_compiler.compile(
        query=request.query,
        active_file=active_rel_path,
        current_symbol=request.current_symbol,
        selected_code=request.selected_code,
        diagnostics=request.diagnostics,
        local_llm_url=request.local_llm_url,
        local_llm_model=request.local_llm_model
    )

    # 2. Build index and dependency graph
    repo_index = RepositoryIndex(request.workspace_root)
    repo_index.build_index()
    
    dep_graph = DependencyGraph(repo_index)
    dep_graph.build_graph()

    # 3. Retrieve candidates
    lex_retriever = LexicalRetriever(repo_index)
    seed_files = lex_retriever.search_identifiers(goal.identifiers)
    seed_files.add(active_rel_path)

    graph_retriever = GraphRetriever(dep_graph)
    distances = graph_retriever.get_neighbors(seed_files, max_hops=1)

    # Load file contents for ranking
    candidates = []
    for path in distances:
        if path in repo_index.index:
            file_path_abs = workspace_root_path / path
            if file_path_abs.exists():
                try:
                    content = file_path_abs.read_text(encoding='utf-8')
                    candidates.append((path, content))
                    repo_index.index[path]["content"] = content
                except Exception as e:
                    logger.error(f"Error reading file {file_path_abs}: {e}")

    # 4. Neural Reranking: Only rerank files that are active or match goal identifiers precisely
    rank_candidates = []
    skipped_candidates = []
    
    python_keywords = {
        'def', 'class', 'import', 'from', 'as', 'return', 'if', 'else', 'elif',
        'try', 'except', 'finally', 'for', 'while', 'in', 'is', 'not', 'and', 'or',
        'with', 'pass', 'break', 'continue', 'lambda', 'global', 'nonlocal', 'assert',
        'del', 'yield', 'raise', 'True', 'False', 'None', 'self', 'str', 'int', 'float',
        'list', 'dict', 'set', 'tuple', 'bool', 'type', 'print', 'len', 'range'
    }
    goal_idents = {ident for ident in goal.identifiers if ident not in python_keywords}

    for path, content in candidates:
        is_active = (path == active_rel_path)
        # Check if content has any matched identifier using word boundary check
        has_ident = False
        for ident in goal_idents:
            if re.search(r'\b' + re.escape(ident) + r'\b', content):
                has_ident = True
                break
        
        if is_active or has_ident:
            rank_candidates.append((path, content))
        else:
            skipped_candidates.append((path, -5.0)) # Default low score (Tier 3)

    print(f"WorkspacePrune: Found {len(rank_candidates)} rank candidates and {len(skipped_candidates)} skipped candidates.", flush=True)
    ranker = CandidateRanker(model)
    # Run CPU-bound candidate ranking in background threadpool
    ranked_scores = await asyncio.to_thread(ranker.rank_candidates, goal.objective, rank_candidates)
    
    # Merge both ranked and skipped candidates
    ranked_scores.extend(skipped_candidates)
    ranked_scores = sorted(ranked_scores, key=lambda x: x[1], reverse=True)
    print("WorkspacePrune: candidate ranking complete. Packaging context...", flush=True)

    # 5. Pack Context: Run CPU-bound context packing and pruning in background threadpool
    builder = ContextBuilder()
    unified_prompt, file_summaries = await asyncio.to_thread(
        builder.pack_context,
        goal.objective,
        repo_index.index,
        distances,
        ranked_scores,
        model,
        request.threshold
    )

    original_tokens = sum(f["original_tokens"] for f in file_summaries)
    pruned_tokens = sum(f["pruned_tokens"] for f in file_summaries)

    return WorkspacePruneResponse(
        structured_goal=goal.dict(),
        unified_prompt=unified_prompt,
        pruned_tokens=pruned_tokens,
        original_tokens=original_tokens,
        files=file_summaries
    )


@app.post("/estimate-carbon", response_model=CarbonEstimateResponse)
async def estimate_carbon(request: CarbonEstimateRequest) -> CarbonEstimateResponse:
    return carbon_estimator.estimate(request)


@cli.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0", "--host", "-h", help="Host to bind the server to"
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Port to run the server on"),
    model_path: Optional[str] = typer.Option(
        None,
        "--model-path",
        "-m",
        help="Path to model directory. Overrides SWEPRUNER_MODEL_PATH environment variable.",
    ),
):
    """Start the FastAPI server for code pruning."""
    if model_path:
        os.environ["SWEPRUNER_MODEL_PATH"] = model_path

    # Verify model exists before starting server
    final_model_path = os.getenv("SWEPRUNER_MODEL_PATH", "./model")
    if not check_model_path(final_model_path):
        typer.echo(
            f"Error: Model not found at {final_model_path}",
            err=True,
        )
        typer.echo(
            "Please download the model or set SWEPRUNER_MODEL_PATH environment variable.",
            err=True,
        )
        typer.echo("See README.md for instructions.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Starting server on {host}:{port}")
    typer.echo(f"Model path: {final_model_path}")
    uvicorn.run(app, host=host, port=port)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
