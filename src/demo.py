"""Interactive CLI Demo for local MRL embedding search.

Loads fine-tuned weights, dynamically truncates embeddings to --dim (default 128),
searches pre-embedded Hawaii businesses, prints ranked results with metadata,
and confirms sub-50ms search latency.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mrl.demo")


def load_corpus(corpus_path: Path | str) -> list[dict]:
    """Load business corpus metadata from JSON Lines."""
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}. Run preprocess.py first.")
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def get_corpus_embeddings(
    model: SentenceTransformer,
    corpus: list[dict],
    cache_path: Path,
    device: torch.device,
) -> torch.Tensor:
    """Load pre-computed full corpus embeddings or compute and cache them."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        logger.info(f"Loading cached corpus embeddings from {cache_path}...")
        embeds = torch.load(cache_path, map_location=device, weights_only=True)
        if embeds.size(0) == len(corpus):
            return embeds

    logger.info(f"Computing embeddings for {len(corpus)} businesses...")
    texts = [d["text"] for d in corpus]
    embeds = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeds.cpu(), cache_path)
    logger.info(f"Saved corpus embeddings to {cache_path}")
    return embeds.to(device)


def run_query(
    query_text: str,
    model: SentenceTransformer,
    corpus_dim: torch.Tensor,
    corpus: list[dict],
    dim: int,
    top_k: int,
    device: torch.device,
) -> tuple[list[dict], float]:
    """Execute dynamic MRL embedding search and measure elapsed latency."""
    if device.type == "mps":
        torch.mps.synchronize()
    start_time = time.perf_counter()

    # 1. Fast direct tokenization with fixed length to avoid Metal JIT recompilation
    tok = getattr(model, "tokenizer", None)
    with torch.no_grad():
        if tok is not None:
            inputs = tok([query_text], return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(inputs)
            query_raw = out.get("sentence_embedding", out.get("token_embeddings", None))
        else:
            query_raw = model.encode([query_text], convert_to_tensor=True, normalize_embeddings=False, device=str(device))

        # 2. Slice to requested Matryoshka dimension and L2-normalize
        query_dim = F.normalize(query_raw[:, :dim], p=2, dim=1)

        # 3. Vector dot product / cosine similarity against sliced corpus
        sims = torch.matmul(query_dim, corpus_dim.T)[0]

        # 4. Top-K ranking
        top_k_clamped = min(top_k, len(corpus))
        top_scores, top_indices = torch.topk(sims, k=top_k_clamped)

    if device.type == "mps":
        torch.mps.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    results = []
    for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
        doc = corpus[idx]
        results.append({
            "name": doc.get("name", "Unknown"),
            "address": doc.get("address", "N/A"),
            "category": doc.get("category", "N/A"),
            "score": round(score, 4),
            "description": doc.get("description", "None"),
            "avg_rating": doc.get("avg_rating", "N/A"),
            "doc_id": doc.get("doc_id", ""),
        })

    return results, latency_ms


def display_results(query: str, results: list[dict], latency_ms: float, dim: int):
    """Print beautifully formatted search results."""
    print("\n" + "-" * 75)
    print(f"QUERY: \"{query}\" (Dimension: {dim})")
    print(f"LATENCY: {latency_ms:.2f} ms " + ("[PASS < 50ms]" if latency_ms < 50.0 else "[EXCEEDED 50ms]"))
    print("-" * 75)
    for rank, res in enumerate(results, 1):
        print(f"[{rank}] {res['name']}  (Score: {res['score']:.4f})")
        print(f"    Address    : {res['address']}")
        print(f"    Categories : {res['category']}")
        print(f"    Rating     : {res['avg_rating']}")
        if res['description'] and res['description'] != "None":
            desc_preview = res['description'][:120] + "..." if len(res['description']) > 120 else res['description']
            print(f"    Details    : {desc_preview}")
        print()
    print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI Demo for local MRL retrieval.")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model", help="Path to fine-tuned model")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed data")
    parser.add_argument("--dim", type=int, default=128, help="Matryoshka embedding dimension (default: 128)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--query", type=str, default=None, help="Execute single query and exit")
    args = parser.parse_args()

    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Loading search engine on device: {device}...")

    # Load Model
    model = SentenceTransformer(args.model_path, device=str(device))
    get_dim_fn = getattr(model, "get_embedding_dimension", getattr(model, "get_sentence_embedding_dimension", None))
    base_dim = get_dim_fn() if get_dim_fn else 768
    dim = min(args.dim, base_dim)

    # Load Corpus & Embeddings
    data_dir = Path(args.data_dir)
    corpus = load_corpus(data_dir / "corpus.jsonl")
    corpus_embeds_raw = get_corpus_embeddings(
        model=model,
        corpus=corpus,
        cache_path=data_dir / "corpus_embeddings.pt",
        device=device,
    )

    # Pre-slice and L2-normalize corpus once for the session
    logger.info(f"Preparing corpus embeddings sliced at {dim} dimensions on {device}...")
    corpus_dim = F.normalize(corpus_embeds_raw[:, :dim], p=2, dim=1).to(device)

    # Warmup query
    _, _ = run_query("warmup search query", model, corpus_dim, corpus, dim, 1, device)

    # Single-shot query execution
    if args.query:
        results, latency_ms = run_query(args.query, model, corpus_dim, corpus, dim, args.top_k, device)
        display_results(args.query, results, latency_ms, dim)
        sys.exit(0)

    # Interactive REPL mode
    print("\n" + "=" * 75)
    print("HAWAII GOOGLE LOCAL - MRL SEARCH REPL")
    print(f"Model: {args.model_path} | Dimension: {dim} | Corpus Size: {len(corpus)}")
    print("Type your search query and press Enter. Type 'exit' or 'quit' to exit.")
    print("=" * 75 + "\n")

    try:
        while True:
            try:
                user_input = input("\nEnter search query (or 'exit'): ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting search demo. Aloha!")
                break

            results, latency_ms = run_query(user_input, model, corpus_dim, corpus, dim, args.top_k, device)
            display_results(user_input, results, latency_ms, dim)

    except KeyboardInterrupt:
        print("\nExiting search demo. Aloha!")
        sys.exit(0)


if __name__ == "__main__":
    main()
