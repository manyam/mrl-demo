"""Review selection and scoring utilities for search result augmentation."""

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("mrl.reviews")


def extract_keywords(query: str | None) -> set[str]:
    """Extract lowercase significant keywords from a query string."""
    if not query:
        return set()
    stopwords = {
        "a", "an", "the", "in", "on", "at", "for", "with", "and", "or",
        "of", "to", "from", "by", "is", "are", "was", "were", "near",
        "near-by", "nearby", "best", "good", "place", "places", "shop", "shops"
    }
    words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
    return words - stopwords


def score_review(review: dict[str, Any], query_terms: set[str] | None = None) -> float:
    """Score a review based on informative length, rating, and query keyword overlap."""
    text = review.get("text", "")
    if not text:
        return -1.0
    text = text.strip()
    text_len = len(text)
    if text_len < 20:
        return -1.0

    # Gaussian curve centered around 180 characters
    len_score = math.exp(-((text_len - 180) ** 2) / (2 * (90 ** 2)))

    # Rating factor (prioritize high ratings 4.0 - 5.0)
    try:
        raw_rating = review.get("rating")
        rating = float(raw_rating) if raw_rating is not None else 4.0
    except (ValueError, TypeError):
        rating = 4.0
    rating_score = min(1.0, max(0.0, rating / 5.0))

    # Query keyword overlap
    overlap_score = 0.0
    if query_terms:
        tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
        matching = len(tokens & query_terms)
        overlap_score = min(1.0, matching * 0.35)

    return (0.45 * rating_score) + (0.25 * len_score) + (0.30 * overlap_score)


def select_top_review(
    candidate_reviews: list[dict[str, Any]],
    query: str | None = None,
) -> dict[str, Any] | None:
    """Select the single best review for a business, optionally conditioned on the query."""
    if not candidate_reviews:
        return None

    query_terms = extract_keywords(query)
    scored = []
    for r in candidate_reviews:
        s = score_review(r, query_terms)
        if s > 0:
            scored.append((s, r))

    if not scored:
        # Fall back to first review if none met score threshold
        first = candidate_reviews[0]
        return {
            "text": first.get("text", "").strip(),
            "rating": first.get("rating", 5.0),
            "user_id": first.get("user_id", ""),
        }

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    return {
        "text": best.get("text", "").strip(),
        "rating": best.get("rating", 5.0),
        "user_id": best.get("user_id", ""),
        "relevance_score": round(scored[0][0], 3),
    }


def build_top_reviews_map(
    review_path: Path,
    max_candidates_per_biz: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Scan raw reviews and accumulate top candidate reviews per business."""
    logger.info(f"Scanning reviews from {review_path} to build top reviews index...")
    biz_candidates = defaultdict(list)

    # Stream JSON Lines
    line_count = 0
    with open(review_path, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            gmap_id = rec.get("gmap_id")
            text = rec.get("text")
            if not gmap_id or not text or len(text.strip()) < 25:
                continue

            review_entry = {
                "text": text.strip(),
                "rating": rec.get("rating", 5.0),
                "user_id": rec.get("user_id", ""),
            }

            current_list = biz_candidates[gmap_id]
            current_list.append(review_entry)
            # Keep list bounded by highest base quality (without query terms)
            if len(current_list) > max_candidates_per_biz * 2:
                current_list.sort(key=lambda r: score_review(r), reverse=True)
                biz_candidates[gmap_id] = current_list[:max_candidates_per_biz]

    # Final trim
    result = {}
    for gmap_id, revs in biz_candidates.items():
        revs.sort(key=lambda r: score_review(r), reverse=True)
        result[gmap_id] = revs[:max_candidates_per_biz]

    logger.info(f"Built reviews index for {len(result)} businesses from {line_count} reviews.")
    return result


def enrich_corpus_file(corpus_path: Path, reviews_index: dict[str, list[dict[str, Any]]]) -> int:
    """Enrich corpus.jsonl in-place with top_reviews."""
    if not corpus_path.exists():
        logger.warning(f"Corpus file {corpus_path} not found.")
        return 0

    enriched_docs = []
    enriched_count = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc.get("doc_id")
            if doc_id in reviews_index:
                doc["top_reviews"] = reviews_index[doc_id]
                enriched_count += 1
            else:
                doc["top_reviews"] = []
            enriched_docs.append(doc)

    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in enriched_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    logger.info(f"Enriched {enriched_count}/{len(enriched_docs)} businesses in {corpus_path.name} with reviews.")
    return enriched_count
