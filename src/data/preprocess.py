"""Preprocessing and pair generation pipeline for UCSD Google Local Hawaii dataset.

Parses streaming JSON metadata and reviews, filters short reviews, enriches business
representations, deduplicates user-business interactions, and performs a leakage-free
stratified split by gmap_id (85% Train, 5% Val, 10% Test).
"""

import argparse
import gzip
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mrl.preprocess")


def open_file_stream(filepath: Path) -> Generator[str, None, None]:
    """Yield lines from either a compressed gzip or plain text/json file."""
    if filepath.suffix == ".gz":
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            for line in f:
                yield line
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                yield line


def format_category_list(cat: Any) -> str:
    """Format category attribute into a clean comma-separated string."""
    if not cat:
        return "None"
    if isinstance(cat, list):
        filtered = [str(c).strip() for c in cat if c and str(c).strip()]
        return ", ".join(filtered) if filtered else "None"
    return str(cat).strip() or "None"


def load_metadata(meta_path: Path) -> dict[str, dict[str, Any]]:
    """Parse business metadata streaming line-by-line."""
    logger.info(f"Parsing metadata from {meta_path}...")
    businesses = {}
    count = 0
    for line in open_file_stream(meta_path):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        gmap_id = record.get("gmap_id")
        if not gmap_id:
            continue

        name = record.get("name") or "Unknown Business"
        category_list = format_category_list(record.get("category"))
        raw_desc = record.get("description")
        description = "None"
        if raw_desc:
            if isinstance(raw_desc, list):
                description = " ".join([str(d).strip() for d in raw_desc if d])
            else:
                description = str(raw_desc).strip()
        if not description:
            description = "None"

        # Structured positive representation required by spec:
        # Business: {name} | Categories: {category_list} | Details: {description}
        enriched_representation = (
            f"Business: {name} | Categories: {category_list} | Details: {description}"
        )

        businesses[gmap_id] = {
            "gmap_id": gmap_id,
            "name": name,
            "address": record.get("address") or "N/A",
            "category": category_list,
            "description": description,
            "avg_rating": record.get("avg_rating") or record.get("rating"),
            "num_of_reviews": record.get("num_of_reviews"),
            "latitude": record.get("latitude"),
            "longitude": record.get("longitude"),
            "enriched_text": enriched_representation,
        }
        count += 1

    logger.info(f"Loaded {len(businesses)} businesses from {count} records in metadata.")
    return businesses


def parse_and_filter_reviews(
    review_path: Path,
    valid_business_ids: set[str],
    max_reviews: int | None = None,
) -> list[dict[str, Any]]:
    """Parse reviews, filter <20 characters, and deduplicate by (user_id, gmap_id)."""
    logger.info(f"Parsing reviews from {review_path}...")
    seen_pairs = set()
    cleaned_reviews = []
    total_lines = 0
    filtered_short = 0
    missing_meta = 0
    duplicates = 0

    for line in open_file_stream(review_path):
        total_lines += 1
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        text = record.get("text")
        # Filter out reviews where text is null, empty, or shorter than 20 characters
        if not text or not isinstance(text, str) or len(text.strip()) < 20:
            filtered_short += 1
            continue

        gmap_id = record.get("gmap_id")
        if not gmap_id or gmap_id not in valid_business_ids:
            missing_meta += 1
            continue

        user_id = record.get("user_id") or f"anon_{total_lines}"
        pair_key = (user_id, gmap_id)
        if pair_key in seen_pairs:
            duplicates += 1
            continue
        seen_pairs.add(pair_key)

        cleaned_reviews.append({
            "user_id": user_id,
            "gmap_id": gmap_id,
            "rating": record.get("rating"),
            "text": text.strip().replace("\n", " "),
            "time": record.get("time"),
        })

        if max_reviews and len(cleaned_reviews) >= max_reviews:
            logger.info(f"Reached review sample limit: {max_reviews}")
            break

    logger.info(
        f"Parsed {total_lines} raw review rows -> {len(cleaned_reviews)} valid deduplicated reviews. "
        f"Filtered (short/null: {filtered_short}, unmapped business: {missing_meta}, duplicate: {duplicates})."
    )
    return cleaned_reviews


def stratified_split_by_business(
    reviews: list[dict[str, Any]],
    businesses: dict[str, dict[str, Any]],
    train_ratio: float = 0.85,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Partition businesses uniquely into train (85%), val (5%), test (10%) to prevent data leakage."""
    rng = random.Random(seed)

    # Group reviews by business
    biz_to_reviews = defaultdict(list)
    for r in reviews:
        biz_to_reviews[r["gmap_id"]].append(r)

    # Sort businesses deterministically by review count descending to balance split distribution
    all_biz = list(biz_to_reviews.keys())
    rng.shuffle(all_biz)
    all_biz.sort(key=lambda b: len(biz_to_reviews[b]), reverse=True)

    total_reviews = len(reviews)
    target_train = int(total_reviews * train_ratio)
    target_val = int(total_reviews * val_ratio)

    train_biz = set()
    val_biz = set()
    test_biz = set()

    train_count = 0
    val_count = 0
    test_count = 0

    # Greedy allocation of businesses to meet target review proportions
    for b in all_biz:
        n_revs = len(biz_to_reviews[b])
        # Prioritize train until target reached
        if train_count + n_revs <= target_train or (val_count >= target_val and test_count > 0):
            train_biz.add(b)
            train_count += n_revs
        elif val_count + n_revs <= target_val:
            val_biz.add(b)
            val_count += n_revs
        else:
            test_biz.add(b)
            test_count += n_revs

    # Ensure at least 1 business in each set
    if not val_biz and len(train_biz) > 1:
        moved = train_biz.pop()
        val_biz.add(moved)
        train_count -= len(biz_to_reviews[moved])
        val_count += len(biz_to_reviews[moved])

    if not test_biz and len(train_biz) > 1:
        moved = train_biz.pop()
        test_biz.add(moved)
        train_count -= len(biz_to_reviews[moved])
        test_count += len(biz_to_reviews[moved])

    # Strict leakage assertion
    assert train_biz.isdisjoint(test_biz), "Leakage detected: train and test share businesses!"
    assert train_biz.isdisjoint(val_biz), "Leakage detected: train and val share businesses!"
    assert val_biz.isdisjoint(test_biz), "Leakage detected: val and test share businesses!"

    logger.info(
        f"Stratified Split: Train={len(train_biz)} businesses ({train_count} reviews, {train_count/total_reviews*100:.1f}%), "
        f"Val={len(val_biz)} businesses ({val_count} reviews, {val_count/total_reviews*100:.1f}%), "
        f"Test={len(test_biz)} businesses ({test_count} reviews, {test_count/total_reviews*100:.1f}%)"
    )

    train_pairs = []
    for b in train_biz:
        pos_text = businesses[b]["enriched_text"]
        for r in biz_to_reviews[b]:
            train_pairs.append({
                "anchor": r["text"],
                "positive": pos_text,
                "gmap_id": b,
                "user_id": r["user_id"],
            })

    val_pairs = []
    for b in val_biz:
        pos_text = businesses[b]["enriched_text"]
        for r in biz_to_reviews[b]:
            val_pairs.append({
                "anchor": r["text"],
                "positive": pos_text,
                "gmap_id": b,
                "user_id": r["user_id"],
            })

    test_queries = []
    qid = 1
    for b in test_biz:
        for r in biz_to_reviews[b]:
            test_queries.append({
                "query_id": f"q_{qid}",
                "query": r["text"],
                "gmap_id": b,
                "user_id": r["user_id"],
            })
            qid += 1

    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)
    rng.shuffle(test_queries)

    return train_pairs, val_pairs, test_queries, biz_to_reviews


def save_jsonl(records: list[dict[str, Any]], dest_path: Path):
    """Write records to JSON Lines format."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records to {dest_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and generate MRL training pairs.")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw data directory")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Path to output directory")
    parser.add_argument("--max_pairs", type=int, default=None, help="Cap total reviews processed for quick iteration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Locate metadata file
    meta_path = raw_dir / "meta-Hawaii.json.gz"
    if not meta_path.exists():
        meta_path = raw_dir / "meta-Hawaii.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found in {raw_dir}. Run download.py first.")

    # Locate review file
    review_path = raw_dir / "review-Hawaii.json.gz"
    if not review_path.exists():
        review_path = raw_dir / "review-Hawaii_10.json"
    if not review_path.exists():
        review_path = raw_dir / "review-Hawaii.json"
    if not review_path.exists():
        raise FileNotFoundError(f"Review file not found in {raw_dir}. Run download.py first.")

    businesses = load_metadata(meta_path)
    valid_biz_ids = set(businesses.keys())

    reviews = parse_and_filter_reviews(
        review_path=review_path,
        valid_business_ids=valid_biz_ids,
        max_reviews=args.max_pairs,
    )

    train_pairs, val_pairs, test_queries, biz_to_reviews = stratified_split_by_business(
        reviews=reviews,
        businesses=businesses,
        train_ratio=0.85,
        val_ratio=0.05,
        test_ratio=0.10,
        seed=args.seed,
    )

    # Corpus for retrieval evaluation and search demo
    from src.data.reviews import score_review
    corpus = []
    for b_id, b_data in businesses.items():
        biz_revs = biz_to_reviews.get(b_id, [])
        if biz_revs:
            biz_revs_sorted = sorted(biz_revs, key=lambda r: score_review(r), reverse=True)
            top_revs = [
                {"text": r["text"], "rating": r.get("rating", 5.0), "user_id": r.get("user_id", "")}
                for r in biz_revs_sorted[:3]
            ]
        else:
            top_revs = []

        corpus.append({
            "doc_id": b_id,
            "text": b_data["enriched_text"],
            "name": b_data["name"],
            "address": b_data.get("address", "N/A"),
            "category": b_data["category"],
            "description": b_data["description"],
            "avg_rating": b_data["avg_rating"],
            "num_of_reviews": b_data["num_of_reviews"],
            "latitude": b_data["latitude"],
            "longitude": b_data["longitude"],
            "top_reviews": top_revs,
        })

    # Test Qrels mapping: query_id -> {doc_id: 1}
    test_qrels = {
        q["query_id"]: [q["gmap_id"]]
        for q in test_queries
    }

    # Save all datasets
    save_jsonl(train_pairs, processed_dir / "train.jsonl")
    save_jsonl(val_pairs, processed_dir / "val.jsonl")
    save_jsonl(test_queries, processed_dir / "test_queries.jsonl")
    save_jsonl(corpus, processed_dir / "corpus.jsonl")

    with open(processed_dir / "test_qrels.json", "w", encoding="utf-8") as f:
        json.dump(test_qrels, f, indent=2)
    logger.info(f"Saved {len(test_qrels)} QREL mappings to test_qrels.json")

    logger.info("Data preprocessing completed successfully.")


if __name__ == "__main__":
    main()
