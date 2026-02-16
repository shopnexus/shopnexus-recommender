import logging
from config import event_weights, NUM_INTERESTS
from typing import Dict, List
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


def get_event_weight(event_type: str) -> float:
    return event_weights.get(event_type, 0.0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def aggregate_product_weights(events: List[Dict]) -> Dict[str, float]:
    """Aggregate event weights per product from a list of events."""
    product_weights: Dict[str, float] = defaultdict(float)
    for e in events:
        item_id = e.get("ref_id")
        if item_id is None:
            continue
        ev_type = (e.get("event_type") or "").lower()
        product_weights[item_id] += get_event_weight(ev_type)
    return product_weights


def default_interests(dim: int):
    """Return empty interests and strengths."""
    return (
        [np.zeros(dim, dtype=np.float32) for _ in range(NUM_INTERESTS)],
        [0.0] * NUM_INTERESTS,
    )
