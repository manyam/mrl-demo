"""Deterministic extractive search summarizer for offline fallback and unit testing."""

import time
from typing import Iterator

from src.summary.base import BaseSearchSummarizer


class ExtractiveSummarizer(BaseSearchSummarizer):
    """Heuristic summarizer synthesizing top-k documents without external LLM dependencies."""

    def __init__(self, model_name: str = "extractive-heuristic"):
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "extractive"

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> bool:
        return True

    def generate(self, query: str, documents: list[dict]) -> str:
        """Synthesize a structured summary from retrieved documents."""
        if not documents:
            return f"No matching places found for query: \"{query}\"."

        top_doc = documents[0]
        top_name = top_doc.get("name", "Top Match")
        top_addr = top_doc.get("address", "N/A")
        top_cat = top_doc.get("category", "General")
        top_rating = top_doc.get("avg_rating", "N/A")

        # Check for geographic note
        geo_note = documents[0].get("geo_note")
        summary_lines = []
        if geo_note:
            summary_lines.append(f"> ℹ️ **{geo_note}**\n")

        dist_str = ""
        if top_doc.get("distance_miles") is not None:
            anchor = top_doc.get("distance_anchor")
            anchor_part = f" from {anchor}" if anchor else ""
            dist_str = f", located {top_doc['distance_miles']:.2f} miles{anchor_part}"

        summary_lines.extend([
            f"Based on your search for **\"{query}\"**, the highest-matching destination in Hawaii is "
            f"**{top_name}** ([1]){dist_str} at *{top_addr}* ({top_cat}, rated {top_rating}/5).",
            "",
            "### Key Highlights & Recommendations:",
        ])

        # Process each place
        review_quotes = []
        for rank, doc in enumerate(documents, 1):
            name = doc.get("name", f"Place #{rank}")
            addr = doc.get("address", "Address unavailable")
            category = doc.get("category", "Local business")
            rating = doc.get("avg_rating", "N/A")
            desc = doc.get("description", "None")
            dist = doc.get("distance_miles")
            anchor = doc.get("distance_anchor")

            dist_header = f" | {dist:.2f} mi from {anchor}" if dist is not None and anchor else (f" | {dist:.2f} mi" if dist is not None else "")
            line = f"- **[{rank}] {name}** (Rating: {rating}/5{dist_header})\n  *Address*: {addr}\n  *Type*: {category}"
            if desc and desc != "None":
                short_desc = desc[:140] + "..." if len(desc) > 140 else desc
                line += f"\n  *Details*: {short_desc}"

            # Extract review if present
            top_rev = doc.get("top_review") or (doc["top_reviews"][0] if doc.get("top_reviews") else None)
            if top_rev and isinstance(top_rev, dict) and top_rev.get("text"):
                rev_text = top_rev["text"].strip()
                rev_rating = top_rev.get("rating", 5)
                short_rev = rev_text[:140] + "..." if len(rev_text) > 140 else rev_text
                line += f"\n  *Top Review*: \"{short_rev}\" (★ {rev_rating})"
                review_quotes.append(f"- **{name}** ([{rank}]): \"{short_rev}\"")

            summary_lines.append(line)

        # Synthesize customer consensus across options
        if review_quotes:
            summary_lines.append("")
            summary_lines.append("### Customer Consensus & Review Insights:")
            summary_lines.extend(review_quotes)

        # Synthesize category groupings
        categories = [d.get("category", "") for d in documents if d.get("category")]
        unique_cats = list(dict.fromkeys([c.split(",")[0].strip() for c in categories if c]))
        if len(unique_cats) > 1:
            cats_str = ", ".join(unique_cats[:3])
            summary_lines.append("")
            summary_lines.append(
                f"**Overall Options Summary**: Across the top {len(documents)} results, options span {cats_str}. "
                f"For the highest overall match, **{top_name}** ([1]) leads customer satisfaction, "
                f"while the other annotated options above provide great alternatives based on reviews and location."
            )

        return "\n".join(summary_lines)

    def stream_generate(self, query: str, documents: list[dict]) -> Iterator[str]:
        """Stream chunks of the extractive summary."""
        full_text = self.generate(query, documents)
        lines = full_text.split("\n")
        for i, line in enumerate(lines):
            yield line + ("\n" if i < len(lines) - 1 else "")
            time.sleep(0.01)
