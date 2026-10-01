"""Prompt engineering and context formatting utilities for generative search summarization."""

import re

SYSTEM_INSTRUCTION = (
    "You are a knowledgeable local search assistant specializing in Hawaii businesses and destinations.\n"
    "Your task is to synthesize a direct, well-structured, and helpful search overview answering the user's "
    "query based EXCLUSIVELY on the provided top-ranked businesses and their authentic customer reviews.\n\n"
    "Strict Requirements:\n"
    "1. Direct Answer: Answer the user's intent in the first 1-2 sentences, highlighting the best recommendations.\n"
    "2. Review Synthesis: Combine business attributes with customer review highlights to summarize what real visitors "
    "experience (e.g. standout dishes, seating, views, atmosphere, crowd favorites).\n"
    "3. Attribution: Every place or quote you mention MUST include its citation index in brackets, e.g., [1], [2].\n"
    "4. Actionable Details: Explicitly mention physical addresses, ratings, and categories so users can easily visit.\n"
    "5. Objective & Grounded: Do not invent details not present in the provided descriptions or reviews.\n"
    "6. Format: Use clean markdown with bold place names and concise bullet points or short paragraphs."
)


def format_documents_context(documents: list[dict]) -> str:
    """Format candidate business records and top customer reviews into a numbered context block."""
    lines = []
    for rank, doc in enumerate(documents, 1):
        name = doc.get("name", "Unknown Business")
        address = doc.get("address", "Address unavailable")
        category = doc.get("category", "General")
        rating = doc.get("avg_rating", "N/A")
        desc = doc.get("description", "None")
        if desc and desc != "None":
            details = desc.strip()
        else:
            details = "No additional description."

        block = [
            f"[{rank}] Name: {name}",
            f"    Address: {address}",
            f"    Categories: {category}",
            f"    Overall Rating: {rating}/5",
            f"    Details: {details}",
        ]

        # Attach top customer review if available
        top_rev = doc.get("top_review")
        if top_rev and isinstance(top_rev, dict) and top_rev.get("text"):
            rev_rating = top_rev.get("rating", "5")
            block.append(f"    Top Customer Review: \"{top_rev['text']}\" (Rated: {rev_rating}/5)")
        elif doc.get("top_reviews"):
            first_rev = doc["top_reviews"][0]
            if isinstance(first_rev, dict) and first_rev.get("text"):
                rev_rating = first_rev.get("rating", "5")
                block.append(f"    Top Customer Review: \"{first_rev['text']}\" (Rated: {rev_rating}/5)")

        lines.append("\n".join(block))
    return "\n\n".join(lines)


def build_summary_prompt(query: str, documents: list[dict]) -> str:
    """Construct full text prompt for causal completion models."""
    context = format_documents_context(documents)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"--- RETRIEVED BUSINESSES & TOP REVIEWS ---\n"
        f"{context}\n\n"
        f"--- USER QUERY ---\n"
        f"\"{query}\"\n\n"
        f"--- SEARCH OVERVIEW & REVIEW SYNTHESIS ---\n"
    )


def build_chat_messages(query: str, documents: list[dict]) -> list[dict[str, str]]:
    """Construct structured chat messages for chat/instruction-tuned models."""
    context = format_documents_context(documents)
    user_content = (
        f"Based on the following top retrieved Hawaii businesses and their authentic customer reviews, "
        f"synthesize a comprehensive search overview answering the query: \"{query}\".\n\n"
        f"Retrieved Businesses & Top Customer Reviews:\n{context}\n\n"
        f"Summarize the overall options using the reviews, provide bracketed citations [1]-[5], and include addresses."
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


def extract_citations(text: str) -> list[int]:
    """Extract all bracketed citation numbers (e.g. [1], [2]) found in the generated text."""
    matches = re.findall(r"\[(\d+)\]", text)
    return sorted(list(set(int(m) for m in matches)))
