# Constants
MAX_LENGTH_EMBED = 4048
DESCRIPTION_LENGTH = 10 * 1024

event_weights = {
    # === HIGH INTENT EVENTS (Strong purchase/interest signals) ===
    "purchase": 0.8,  # Strongest positive signal - user bought it
    "add_to_cart": 0.5,  # Strong intent but not committed
    "view": 0.3,  # Strong intent but not committed
    # "checkout_initiated": 0.7,          # Started checkout process
    # "add_to_wishlist": 0.6,             # Save for future - strong interest
    "add_to_favorites": 0.6,  # Permanent save - strong signal
    # === SOCIAL & VALIDATION EVENTS ===
    "write_review": 0.5,  # Active engagement post-purchase
    "rating_high": 0.4,  # Rating 4-5 stars
    "rating_medium": 0.1,  # Rating 3 stars - neutral
    "ask_question": 0.25,  # Considering purchase
    # === DISCOVERY EVENTS (Lower but important) ===
    "click_from_recommendation": 0.15,  # Responded to algo suggestion
    "click_from_search": 0.2,  # Intent-driven discovery
    "click_from_category": 0.12,  # Browsing discovery
    "view_similar_products": 0.15,  # Exploring options
    # === NEGATIVE SIGNALS (Important to prevent bad recommendations) ===
    "remove_from_cart": -0.3,  # Changed mind
    "return_product": -0.6,  # Strong negative (post-purchase dissatisfaction)
    "refund_requested": -0.7,  # Stronger than return intent
    "cancel_order": -0.6,  # Order cancelled before fulfillment
    "rating_low": -0.5,  # Rating 1-2 stars
    "report_product": -1.2,  # Strongest explicit negative
    "dislike": -0.5,  # Explicit dislike/ thumbs-down
    "hide_item": -0.35,  # User hides/blocks the item
    "not_interested": -0.3,  # Explicit not interested
    "view_bounce": -0.1,  # Very short view / bounce
}

# Multi-interest configuration
NUM_INTERESTS = 4
MERGE_THRESHOLD = 0.7  # Cosine sim threshold to merge into existing interest

# Interest update parameters
MAX_STRENGTH = 20.0  # Cap per-interest strength so old interests don't dominate
MIN_ALPHA = 0.05  # Floor for EMA alpha so vectors stay responsive
