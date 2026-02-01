import json
def percentage(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return (part / whole) * 100
def get_comparison(value, average):
    """Helper to determine relative performance."""
    if average == 0: return "at" # Avoid division by zero
    ratio = value / average
    if ratio > 1: return "above"
    if ratio < 1: return "below"
    return "at"

def safe_parse_embedding(emb):
    """Ensures the vector is a numerical list, even if DB returns it as a string."""
    if isinstance(emb, list):
        return emb
    if isinstance(emb, str):
        try:
            # Handle Postgres bracket formats {} or []
            clean_emb = emb.replace('{', '[').replace('}', ']')
            return json.loads(clean_emb)
        except:
            return None
    return None