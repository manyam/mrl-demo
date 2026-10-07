"""Interactive CLI Demo for local MRL embedding search with Generative Search Summarization.

Loads fine-tuned weights, dynamically truncates embeddings to --dim (default 128),
searches pre-embedded Hawaii businesses with sub-50ms latency, and generates an AI Search
Overview / Generative Search Summary of top-5 results using configurable local or API models.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.summary import BaseSearchSummarizer, create_summarizer
from src.utils.config import ConfigDict, load_config

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
    gazetteer: Any = None,
    corpus_coords: Any = None,
) -> tuple[list[dict], float, Any]:
    """Execute dynamic MRL embedding search with optional geo-proximity filtering."""
    if device.type == "mps":
        torch.mps.synchronize()
    start_time = time.perf_counter()

    # 1. Parse geographic proximity constraint from query
    import numpy as np
    from src.geo import parse_geo_query
    from src.geo.distance import compute_distances_vectorized
    geo_constraint = parse_geo_query(query_text, gazetteer=gazetteer)
    embed_query_text = geo_constraint.clean_query if geo_constraint.has_geo else query_text

    # 2. Fast direct tokenization with fixed length to avoid Metal JIT recompilation
    tok = getattr(model, "tokenizer", None)
    with torch.no_grad():
        if tok is not None:
            inputs = tok([embed_query_text], return_tensors="pt", padding="max_length", max_length=64, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            out = model(inputs)
            query_raw = out.get("sentence_embedding", out.get("token_embeddings", None))
        else:
            query_raw = model.encode([embed_query_text], convert_to_tensor=True, normalize_embeddings=False, device=str(device))

        # 3. Slice to requested Matryoshka dimension and L2-normalize
        query_dim = F.normalize(query_raw[:, :dim], p=2, dim=1)

        # 4. Vector dot product / cosine similarity
        sims = torch.matmul(query_dim, corpus_dim.T)[0]
        sims_np = sims.cpu().numpy()

    # 5. Fast spatial proximity filtering and top-k selection
    geo_distance_map = {}
    if geo_constraint.has_geo and corpus_coords is not None:
        dists = compute_distances_vectorized(corpus_coords, geo_constraint.latitude, geo_constraint.longitude)
        valid_mask = dists <= geo_constraint.radius_miles
        num_valid = int(valid_mask.sum())
        if num_valid > 0:
            sims_np[~valid_mask] = -1e9
            top_k_clamped = min(top_k, num_valid)
            top_indices_raw = np.argpartition(sims_np, -top_k_clamped)[-top_k_clamped:]
            top_indices = top_indices_raw[np.argsort(-sims_np[top_indices_raw])].tolist()
            top_scores = [float(sims_np[i]) for i in top_indices]
            for idx in top_indices:
                geo_distance_map[idx] = round(float(dists[idx]), 2)
        else:
            geo_constraint.notes.append(
                f"No businesses found within {geo_constraint.radius_miles:.1f} miles of {geo_constraint.anchor_name}."
            )
            top_k_clamped = min(top_k, len(corpus))
            top_indices_raw = np.argpartition(sims_np, -top_k_clamped)[-top_k_clamped:]
            top_indices = top_indices_raw[np.argsort(-sims_np[top_indices_raw])].tolist()
            top_scores = [float(sims_np[i]) for i in top_indices]
    else:
        top_k_clamped = min(top_k, len(corpus))
        top_indices_raw = np.argpartition(sims_np, -top_k_clamped)[-top_k_clamped:]
        top_indices = top_indices_raw[np.argsort(-sims_np[top_indices_raw])].tolist()
        top_scores = [float(sims_np[i]) for i in top_indices]

    if device.type == "mps":
        torch.mps.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    results = []
    from src.data.reviews import select_top_review
    primary_geo_note = geo_constraint.notes[0] if (geo_constraint and geo_constraint.notes) else None

    for score, idx in zip(top_scores, top_indices):
        doc = corpus[idx]
        top_rev = select_top_review(doc.get("top_reviews", []), query=embed_query_text)
        dist_val = geo_distance_map.get(idx)
        results.append({
            "name": doc.get("name", "Unknown"),
            "address": doc.get("address", "N/A"),
            "category": doc.get("category", "N/A"),
            "score": round(score, 4),
            "description": doc.get("description", "None"),
            "avg_rating": doc.get("avg_rating", "N/A"),
            "doc_id": doc.get("doc_id", ""),
            "top_review": top_rev,
            "distance_miles": dist_val,
            "distance_anchor": geo_constraint.anchor_name,
            "geo_note": primary_geo_note,
        })

    return results, latency_ms, geo_constraint


def display_results(
    query: str,
    results: list[dict],
    latency_ms: float,
    dim: int,
    geo_constraint: Any = None,
):
    """Print beautifully formatted ranked search results with optional distance metrics."""
    print("\n" + "-" * 75)
    print(f"QUERY: \"{query}\" (Dimension: {dim})")
    if geo_constraint and geo_constraint.has_geo:
        print(f"GEO CONSTRAINT   : {geo_constraint.describe()}")
        for note in geo_constraint.notes:
            print(f"[GEO NOTE]       : {note}")
    print(f"RETRIEVAL LATENCY: {latency_ms:.2f} ms " + ("[PASS < 50ms]" if latency_ms < 50.0 else "[EXCEEDED 50ms]"))
    print("-" * 75)
    for rank, res in enumerate(results, 1):
        print(f"[{rank}] {res['name']}  (Score: {res['score']:.4f})")
        if res.get("distance_miles") is not None:
            anchor_str = f" from {res['distance_anchor']}" if res.get("distance_anchor") else ""
            print(f"    Distance   : {res['distance_miles']:.2f} miles{anchor_str}")
        print(f"    Address    : {res['address']}")
        print(f"    Categories : {res['category']}")
        print(f"    Rating     : {res['avg_rating']}")
        if res.get("top_review") and res["top_review"].get("text"):
            rev = res["top_review"]
            rev_text = rev["text"].strip()
            preview = rev_text[:140] + "..." if len(rev_text) > 140 else rev_text
            rating_val = rev.get("rating", 5.0)
            print(f"    Top Review : \"{preview}\" (★ {rating_val})")
        elif res['description'] and res['description'] != "None":
            desc_preview = res['description'][:120] + "..." if len(res['description']) > 120 else res['description']
            print(f"    Details    : {desc_preview}")
        print()
    print("-" * 75)


def display_summary(
    query: str,
    results: list[dict],
    summarizer: BaseSearchSummarizer,
    stream: bool = True,
):
    """Generate and display the AI Search Overview synthesizing the top places."""
    print("\n" + "=" * 75)
    print(f"✦ AI SEARCH OVERVIEW (Model: {summarizer.model_name} | Backend: {summarizer.name})")
    print("=" * 75)

    start_time = time.perf_counter()
    full_text = ""
    token_count = 0

    try:
        if stream:
            for chunk in summarizer.stream_generate(query, results):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                full_text += chunk
                # Approximate token count by whitespace chunks
                token_count += len(chunk.split())
            print()
        else:
            full_text = summarizer.generate(query, results)
            print(full_text)
            token_count = len(full_text.split())

        elapsed = time.perf_counter() - start_time
        tok_per_sec = (token_count / elapsed) if elapsed > 0 else 0.0

        print("-" * 75)
        print(f"[Synthesis Latency: {elapsed:.2f}s | Speed: ~{tok_per_sec:.1f} words/s]")
        print("=" * 75 + "\n")

    except Exception as e:
        logger.error(f"Error during search summarization: {e}")
        print(f"\n[Unable to complete search summary: {e}]\n" + "=" * 75 + "\n")


def print_repl_help():
    """Print REPL interactive commands."""
    print("\nAvailable REPL Commands:")
    print("  /summary [on|off]              : Enable or disable AI Search Overview")
    print("  /backend [auto|ollama|google|local|extractive] : Switch summary backend")
    print("  /dim [64|128|256|512|768]      : Switch Matryoshka retrieval dimension")
    print("  /config                        : View currently loaded configuration")
    print("  /help                          : Show this help message")
    print("  exit / quit                    : Exit the interactive demo\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive CLI Demo for local MRL retrieval and generative search summary.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML configuration file")
    parser.add_argument("--model_path", type=str, default=None, help="Override retrieval model path")
    parser.add_argument("--data_dir", type=str, default=None, help="Override path to processed data")
    parser.add_argument("--dim", type=int, default=None, help="Matryoshka embedding dimension (default from config: 128)")
    parser.add_argument("--top_k", type=int, default=None, help="Number of nearest neighbors to retrieve (default from config: 5)")
    parser.add_argument("--query", type=str, default=None, help="Execute single query and exit")
    parser.add_argument("--summary", dest="summary_enabled", action="store_true", default=None, help="Enable generative search summary")
    parser.add_argument("--no-summary", dest="summary_enabled", action="store_false", help="Disable generative search summary")
    parser.add_argument("--summary_backend", type=str, default=None, choices=["auto", "ollama", "google", "local", "extractive"], help="Override summary backend")
    parser.add_argument("--no-stream", dest="stream", action="store_false", default=True, help="Disable real-time token streaming")
    args = parser.parse_args()

    # 1. Load YAML Configuration
    config = load_config(args.config)

    # 2. Resolve parameters (CLI overrides YAML config)
    retrieval_cfg = config.retrieval
    summary_cfg = config.summary

    model_path = args.model_path or retrieval_cfg.get("model_path", "checkpoints/best_model")
    data_dir_path = Path(args.data_dir or retrieval_cfg.get("data_dir", "data/processed"))
    target_dim = args.dim or retrieval_cfg.get("default_dimension", 128)
    top_k = args.top_k or retrieval_cfg.get("top_k", 5)

    summary_enabled = summary_cfg.get("enabled", True) if args.summary_enabled is None else args.summary_enabled
    backend_choice = args.summary_backend or summary_cfg.get("default_backend", "auto")
    stream_output = args.stream if args.stream is not None else summary_cfg.get("stream", True)

    # 3. Determine device
    device_str = retrieval_cfg.get("device", "mps")
    device = torch.device("mps" if (device_str == "mps" and torch.backends.mps.is_available()) else "cpu")
    logger.info(f"Loading search engine on device: {device}...")

    # 4. Load Model
    model = SentenceTransformer(model_path, device=str(device))
    get_dim_fn = getattr(model, "get_embedding_dimension", getattr(model, "get_sentence_embedding_dimension", None))
    base_dim = get_dim_fn() if get_dim_fn else 768
    dim = min(target_dim, base_dim)

    # 5. Load Corpus & Embeddings
    corpus = load_corpus(data_dir_path / "corpus.jsonl")
    from src.geo import SpatialGazetteer
    from src.geo.distance import build_corpus_coords_array
    gazetteer = SpatialGazetteer(corpus=corpus)
    corpus_coords = build_corpus_coords_array(corpus)

    corpus_embeds_raw = get_corpus_embeddings(
        model=model,
        corpus=corpus,
        cache_path=data_dir_path / "corpus_embeddings.pt",
        device=device,
    )

    # Pre-slice and L2-normalize corpus once for the session
    logger.info(f"Preparing corpus embeddings sliced at {dim} dimensions on {device}...")
    corpus_dim = F.normalize(corpus_embeds_raw[:, :dim], p=2, dim=1).to(device)

    # 6. Initialize Generative Search Summarizer
    summarizer = None
    if summary_enabled:
        summarizer = create_summarizer(config, backend_override=backend_choice)
        logger.info(f"Initialized search summarizer: {summarizer.name} ({summarizer.model_name})")

    # Warmup query
    _, _, _ = run_query("warmup search query", model, corpus_dim, corpus, dim, 1, device, gazetteer=gazetteer, corpus_coords=corpus_coords)

    # Single-shot query execution
    if args.query:
        results, latency_ms, geo_constraint = run_query(args.query, model, corpus_dim, corpus, dim, top_k, device, gazetteer=gazetteer, corpus_coords=corpus_coords)
        display_results(args.query, results, latency_ms, dim, geo_constraint=geo_constraint)
        if summary_enabled and summarizer is not None:
            display_summary(args.query, results, summarizer, stream=stream_output)
        sys.exit(0)

    # Interactive REPL mode
    print("\n" + "=" * 75)
    print("HAWAII GOOGLE LOCAL - MRL SEARCH & GENERATIVE SUMMARY REPL")
    print(f"Retrieval Model: {model_path} | Dimension: {dim} | Corpus Size: {len(corpus)}")
    if summary_enabled and summarizer:
        print(f"AI Overview    : Enabled ({summarizer.name} / {summarizer.model_name})")
    else:
        print("AI Overview    : Disabled")
    print("Type your search query and press Enter. Type '/help' for commands or 'exit' to quit.")
    print("=" * 75 + "\n")

    try:
        while True:
            try:
                user_input = input("\nEnter query (or '/help'): ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting search demo. Aloha!")
                break

            # Handle interactive REPL commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split()
                cmd = cmd_parts[0].lower()

                if cmd == "/help":
                    print_repl_help()
                    continue
                elif cmd == "/config":
                    print("\n--- Current Configuration (YAML) ---")
                    print(f"  Retrieval Model : {model_path}")
                    print(f"  Active Dimension: {dim}")
                    print(f"  Summary Enabled : {summary_enabled}")
                    if summarizer:
                        print(f"  Summary Backend : {summarizer.name}")
                        print(f"  Summary Model   : {summarizer.model_name}")
                    print(f"  1. Ollama Model : {summary_cfg.get('ollama', {}).get('model_name')}")
                    print(f"  2. Google Model : {summary_cfg.get('google', {}).get('model_name')}")
                    print(f"  3. HF Model     : {summary_cfg.get('local', {}).get('model_name')}")
                    print("------------------------------------\n")
                    continue
                elif cmd == "/summary":
                    if len(cmd_parts) > 1:
                        arg = cmd_parts[1].lower()
                        if arg in ("on", "true", "1", "enable"):
                            summary_enabled = True
                            if summarizer is None:
                                summarizer = create_summarizer(config, backend_override=backend_choice)
                            print(f"[AI Overview enabled: {summarizer.name} ({summarizer.model_name})]")
                        elif arg in ("off", "false", "0", "disable"):
                            summary_enabled = False
                            print("[AI Overview disabled]")
                        else:
                            print("Usage: /summary [on|off]")
                    else:
                        print(f"AI Overview is currently: {'ENABLED' if summary_enabled else 'DISABLED'}")
                    continue
                elif cmd == "/backend":
                    if len(cmd_parts) > 1:
                        new_backend = cmd_parts[1].lower()
                        backend_choice = new_backend
                        summarizer = create_summarizer(config, backend_override=new_backend)
                        summary_enabled = True
                        print(f"[Switched summarizer to: {summarizer.name} ({summarizer.model_name})]")
                    else:
                        current = summarizer.name if summarizer else "none"
                        print(f"Current backend: {current}. Usage: /backend [auto|ollama|google|local|extractive]")
                    continue
                elif cmd == "/dim":
                    if len(cmd_parts) > 1:
                        try:
                            new_dim = int(cmd_parts[1])
                            if new_dim > base_dim:
                                print(f"Dimension cannot exceed model base dimension {base_dim}.")
                                continue
                            dim = new_dim
                            print(f"Reslicing corpus to {dim} dimensions...")
                            corpus_dim = F.normalize(corpus_embeds_raw[:, :dim], p=2, dim=1).to(device)
                            print(f"[Dimension set to {dim}]")
                        except ValueError:
                            print("Usage: /dim [64|128|256|512|768]")
                    else:
                        print(f"Current dimension: {dim}")
                    continue
                else:
                    print(f"Unknown command '{cmd}'. Type '/help' for options.")
                    continue

            # Execute search
            results, latency_ms, geo_constraint = run_query(user_input, model, corpus_dim, corpus, dim, top_k, device, gazetteer=gazetteer, corpus_coords=corpus_coords)
            display_results(user_input, results, latency_ms, dim, geo_constraint=geo_constraint)

            # Generate AI search summary
            if summary_enabled and summarizer is not None:
                display_summary(user_input, results, summarizer, stream=stream_output)

    except KeyboardInterrupt:
        print("\nExiting search demo. Aloha!")
        sys.exit(0)


if __name__ == "__main__":
    main()
