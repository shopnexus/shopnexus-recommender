import time
import logging
from datetime import datetime
from typing import Dict
from flask import request, jsonify
from service import Service

logger = logging.getLogger(__name__)


def register_routes(app, service: Service):
    """Register all Flask routes"""

    @app.route("/search", methods=["POST"])
    def search():
        """Product semantic search endpoint"""
        data = request.json
        query = data.get("query")
        if not query:
            return jsonify({"error": "Missing query"}), 400

        limit = data.get("limit", 10)
        weights = data.get("weights", {"dense": 1.0, "sparse": 1.0})
        offset = data.get("offset", 0)

        result = service.semantic_search(
            query,
            dense_weight=weights.get("dense", 1.0),
            sparse_weight=weights.get("sparse", 1.0),
            offset=offset,
            limit=limit,
        )

        return jsonify(result)

    @app.route("/recommend", methods=["GET"])
    def recommend():
        """Get recommendations for a user based on their preferences"""
        limit = request.args.get("limit", 10, type=int)
        account_id = request.args.get("account_id")

        results = service.recommend(account_id, limit=limit)
        return jsonify(results)

    @app.route("/events", methods=["POST"])
    def process_events():
        """Process analytics events endpoint"""
        """
        Example:
        [
            {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "account_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
                "account_number": 123,
                "event_type": "purchase",
                "ref_type": "Product",
                "ref_id": "b2c3d4e5-f6a7-8901-2345-678901bcdef0",
                "date_created": "2025-01-21T14:32:10.123Z",
                "metadata": {}
            }
        ]
        """
        data = request.json
        events = data.get("events", [])
        service.process_events(events)
        return jsonify({"message": "Successfully processed events"})

    @app.route("/products", methods=["POST"])
    def update_products():
        """Update products in Milvus collection"""
        data = request.json
        products = data.get("products", [])
        metadata_only = data.get("metadata_only", False)
        service.update_products(products, metadata_only=metadata_only)
        return jsonify({"message": "Successfully updated products"})

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
            }
        )
