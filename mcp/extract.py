"""Entity and relation extraction using GLiNER (fully local, no API key).

Extracts:
  entities  — named things (people, projects, tools, concepts, places)
  relations — directed co-occurrence triples within sentences

GLiNER model is lazy-loaded on first use (~200MB, cached by HuggingFace).
Falls back silently if extraction fails — never blocks a sync/upsert.
"""

import re

_ENTITY_LABELS = [
    "person", "project", "tool", "technology", "organisation", "concept", "place",
]
_THRESHOLD = 0.4

_model = None  # lazy-loaded


def _get_model():
    global _model
    if _model is None:
        from gliner import GLiNER
        _model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
    return _model


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]


def _derive_relations(entities: list[dict], text: str) -> list[dict]:
    """Co-occurrence-based relations: any two entities in the same sentence get a 'related_to' edge."""
    relations = []
    seen: set[tuple[str, str]] = set()
    sentences = _split_sentences(text)
    entity_names = {e["name"].lower(): e["name"] for e in entities}

    for sentence in sentences:
        sentence_lower = sentence.lower()
        present = [canon for lower, canon in entity_names.items() if lower in sentence_lower]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                pair = (present[i], present[j])
                if pair not in seen:
                    seen.add(pair)
                    relations.append({
                        "subject": present[i],
                        "predicate": "related_to",
                        "object": present[j],
                    })

    return relations[:30]


def extract_entities_and_relations(
    title: str, content: str
) -> tuple[list[dict], list[dict]]:
    """Returns (entities, relations). Both empty on failure."""
    text = f"{title}\n\n{content}"[:4000]

    try:
        model = _get_model()
        raw = model.predict_entities(text, _ENTITY_LABELS, threshold=_THRESHOLD)
        # Deduplicate by lowercased name, keep first seen type
        seen: dict[str, dict] = {}
        for ent in raw:
            key = ent["text"].lower()
            if key not in seen:
                seen[key] = {"name": ent["text"], "type": ent["label"]}
        entities = list(seen.values())[:20]
        relations = _derive_relations(entities, text)
        return entities, relations
    except Exception:
        return [], []
