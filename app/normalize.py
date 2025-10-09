import re

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("“", "\"").replace("”", "\"").replace("’", "'")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
