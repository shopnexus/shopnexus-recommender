import logging
from datetime import datetime
from flask import request, jsonify
from service import Service

logger = logging.getLogger(__name__)


def register_routes(app, service: Service):
    """Register all Flask routes"""

    @app.route("/search", methods=["POST"])
    def search():
        data = request.json
        query = data.get("query")
        if not query:
            return jsonify({"error": "Missing query"}), 400

        result = service.semantic_search(
            query,
            dense_weight=data.get("weights", {}).get("dense", 1.0),
            sparse_weight=data.get("weights", {}).get("sparse", 1.0),
            offset=data.get("offset", 0),
            limit=data.get("limit", 10),
        )
        return jsonify(result)

    @app.route("/recommend", methods=["GET"])
    def recommend():
        account_id = request.args.get("account_id")
        limit = request.args.get("limit", 10, type=int)

        results = service.recommend(account_id, limit=limit)
        return jsonify(results)

    @app.route("/events", methods=["POST"])
    def process_events():
        data = request.json
        events = data.get("events", [])
        service.process_events(events)
        return jsonify({"message": "Successfully processed events"})

    @app.route("/products", methods=["POST"])
    def update_products():
        data = request.json
        products = data.get("products", [])
        metadata_only = data.get("metadata_only", False)
        service.update_products(products, metadata_only=metadata_only)
        return jsonify({"message": "Successfully updated products"})

    @app.route("/health", methods=["GET"])
    def health_check():
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
