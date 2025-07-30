import ollama
from functools import lru_cache
from config import LLM_MODEL

@lru_cache(maxsize=500)
def transform_query_with_ollama(user_query: str) -> str:
    system_prompt = (
        "You are a fashion search query expert. Your job is to rewrite **one** user query for visual product retrieval by:\n"
        "1. Adding visual descriptors (color, pattern, texture, silhouette, details)\n"
        "2. Including style context (casual, formal, vintage, modern, streetwear)\n"
        "3. Specifying fit and cut when relevant (slim, oversized, A-line, cropped)\n"
        "4. Adding seasonal/occasion context when implied\n"
        "5. Preserving gender and size information if mentioned\n"
        "6. Output a **single, natural-sounding sentence** — not a list\n\n"
        "Focus only on attributes that are visually identifiable in product images.\n\n"
        "IMPORTANT: Only return the rewritten sentence. Do NOT include explanations, notes, or any text other than the rewritten sentence.\n\n"
        "Examples:\n"
        "- 'dress' → 'flowy midi dress with floral print and short sleeves'\n"
        "- 'something for work' → 'professional navy blazer with tailored gray slacks and a crisp white shirt for a business setting'\n"
        "- 'black outfit for party' → 'black fitted blazer with matching slim trousers and dress shoes for formal evening wear'"
    )

    response = ollama.chat(
        model = LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )

    return response['message']['content']

def summarize_transformed_query(transformed_query: str) -> str:
    prompt = (
        "You are a friendly fashion assistant.\n"
        "Your job is to write a short, natural-sounding sentence that describes the fashion item being searched for based on the query below.\n\n"
        "Use a tone that's warm, helpful, and suitable for a shopping interface.\n"
        "Feel free to imagine the scenario where the item would be worn (e.g., to a beach, work, date night, etc.).\n"
        "Do NOT rephrase the query—just describe what kind of item is being looked for.\n\n"
        f"Query: {transformed_query}\n\n"
        "Output: "
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content'].strip()


