"""Prompt engineering and context formatting utilities for generative search summarization."""

import re

SYSTEM_INSTRUCTION = (
    "You are a knowledgeable local search assistant specializing in Hawaii businesses and destinations.\n"
    "Your task is to synthesize a direct, well-structured, and helpful search overview answering the user's "
    "query based EXCLUSIVELY on the provided top-ranked businesses.\n\n"
    "Strict Requirements:\n"
    "1. Direct Answer: Answer the user's intent in the first 1-2 sentences, highlighting the best recommendations.\n"
    "2. Attribution: Every place you mention MUST include its citation index in brackets, e.g., [1], [2].\n"
    "3. Actionable Details: Explicitly mention physical addresses, ratings, categories, and key specialties or atmosphere.\n"
    "4. Objective & Grounded: Do not invent or hallucinate information not present in the provided descriptions.\n"
    "5. Format: Use clean markdown with bold place names and concise bullet points or short paragraphs."
)


def format_documents_context(documents: list[dict]) -> str:
    """Format candidate business records into a numbered context block for LLM prompts."""
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

        lines.append(
            f"[{rank}] Name: {name}\n"
            f"    Address: {address}\n"
            f"    Categories: {category}\n"
            f"    Rating: {rating}/5\n"
            f"    Details: {details}"
        )
    return "\n\n".join(lines)


def build_summary_prompt(query: str, documents: list[dict]) -> str:
    """Construct full text prompt for causal completion models."""
    context = format_documents_context(documents)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"--- RETRIEVED BUSINESSES ---\n"
        f"{context}\n\n"
        f"--- USER QUERY ---\n"
        f"\"{query}\"\n\n"
        f"--- SEARCH OVERVIEW ---\n"
    )


def build_chat_messages(query: str, documents: list[dict]) -> list[dict[str, str]]:
    """Construct structured chat messages for chat/instruction-tuned models."""
    context = format_documents_context(documents)
    user_content = (
        f"Based on the following top 5 retrieved Hawaii businesses, synthesize a comprehensive search summary "
        f"answering the query: \"{query}\".\n\n"
        f"Retrieved Businesses:\n{context}\n\n"
        f"Provide the summary with bracketed citations [1]-[5] and include place addresses."
    )
    return [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


def extract_citations(text: str) -> list[int]:
    """Extract all bracketed citation numbers (e.g. [1], [2]) found in the generated text."""
    matches = re.findall(r"\[(\d+)\]", text)
    return sorted(list(set(int(m) for m in matches)))
