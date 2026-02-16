"""Service layer for recommendation system"""

import logging
from typing import List, Dict
from collections import defaultdict
import numpy as np
from milvus import MilvusClient
from pymilvus import AnnSearchRequest, WeightedRanker
from embeddings import EmbeddingService
from utils import aggregate_product_weights, cosine_sim, default_interests
from config import NUM_INTERESTS, MERGE_THRESHOLD, POPULARITY_DECAY, PERSONAL_RATIO, POPULAR_RATIO, RANDOM_RATIO, MAX_STRENGTH, MIN_ALPHA, MAX_PURCHASED_IDS

logger = logging.getLogger(__name__)

INTEREST_FIELDS = [f"interest_{i+1}" for i in range(NUM_INTERESTS)]
STRENGTH_FIELDS = [f"strength_{i+1}" for i in range(NUM_INTERESTS)]
ACCOUNT_FIELDS = INTEREST_FIELDS + STRENGTH_FIELDS


class Service:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """Initialize service with Milvus connection and embedding services"""
        self.embedding_service = EmbeddingService()
        self.client = MilvusClient(
            content_dim=self.embedding_service.dense_dim,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
        )

    def semantic_search(
        self, query: str, dense_weight=1.0, sparse_weight=1.0, offset=0, limit=10
    ):
        """Semantic search using content_products collection"""
        query_embeddings = self.embedding_service.embed_text(query)

        results = self.client.semantic_search(
            content_vec=query_embeddings["dense"],
            sparse_vec=query_embeddings["sparse"],
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            offset=offset,
            limit=limit,
        )

        return [{"id": hit["id"], "score": float(hit.score)} for hit in results[0]]

    def recommend(self, account_id: str, limit: int = 10):
        """Recommend products: personalized + popular + random discovery"""
        personal_count = max(1, int(limit * PERSONAL_RATIO))
        popular_count = max(1, int(limit * POPULAR_RATIO))
        random_count = limit - personal_count - popular_count

        # --- Personalized slots (from interests) ---
        personal_results = []
        results = self.client.accounts_collection.query(
            expr=f"id == '{account_id}'",
            output_fields=ACCOUNT_FIELDS + ["purchased_ids"],
        )

        # Cold-start: no account or no interests → all popular
        if not results:
            popular_results = self._get_popular_products(limit)
            return {"personalized": [], "popular": popular_results, "random": []}

        account = results[0]
        exclude_ids = set(account.get("purchased_ids") or [])

        search_reqs = []
        weights = []
        for i in range(NUM_INTERESTS):
            strength = float(account.get(f"strength_{i+1}", 0.0))
            if strength > 0:
                interest_vec = np.array(account[f"interest_{i+1}"])
                req = AnnSearchRequest(
                    [interest_vec.tolist()],
                    "content_vector",
                    {"metric_type": "COSINE"},
                    limit=personal_count,
                    expr="is_active == true",
                )
                search_reqs.append(req)
                weights.append(strength)

        if not search_reqs:
            popular_results = self._get_popular_products(limit, exclude_ids=exclude_ids)
            return {"personalized": [], "popular": popular_results, "random": []}

        max_w = max(weights)
        normalized = [w / max_w for w in weights]
        rerank = WeightedRanker(*normalized)
        hits = self.client.products_collection.hybrid_search(
            search_reqs,
            rerank=rerank,
            limit=personal_count,
            output_fields=["id"],
        )
        personal_results = [
            {"id": hit["id"], "score": float(hit.score)}
            for hit in hits[0]
            if hit["id"] not in exclude_ids
        ]

        used_ids = exclude_ids | {r["id"] for r in personal_results}

        # --- Popular/trending slots ---
        popular_results = self._get_popular_products(popular_count, exclude_ids=used_ids)
        used_ids.update(r["id"] for r in popular_results)

        # --- Random discovery slots ---
        random_results = self._get_random_products(random_count, exclude_ids=used_ids)

        return {
            "personalized": personal_results,
            "popular": popular_results,
            "random": random_results,
        }

    def _get_popular_products(
        self, limit: int, exclude_ids: set = None
    ) -> List[Dict]:
        """Get popular products for exploration, with some randomness."""
        fetch_count = limit * 5
        rows = self.client.products_collection.query(
            expr="popularity > 0 and is_active == true",
            output_fields=["id", "popularity"],
            limit=fetch_count,
        )

        # Sort by popularity desc so we sample from actual top products
        rows.sort(key=lambda r: r.get("popularity", 0.0), reverse=True)

        if exclude_ids:
            rows = [r for r in rows if r["id"] not in exclude_ids]

        if not rows:
            return []

        # Weighted random sample by popularity
        popularities = np.array([r["popularity"] for r in rows])
        popularities = np.maximum(popularities, 0.0)
        total = popularities.sum()
        if total == 0:
            return []

        probs = popularities / total
        count = min(limit, len(rows))
        indices = np.random.choice(len(rows), size=count, replace=False, p=probs)

        return [{"id": rows[i]["id"], "score": rows[i]["popularity"]} for i in indices]

    def _get_random_products(
        self, limit: int, exclude_ids: set = None
    ) -> List[Dict]:
        """Get random products by searching with a random vector."""
        if limit <= 0:
            return []

        # Random unit vector → finds products from a random region of content space
        random_vec = np.random.randn(self.client.content_dim).astype(np.float32)
        random_vec /= np.linalg.norm(random_vec)

        fetch_count = limit * 2
        hits = self.client.search(
            collection=self.client.products_collection,
            anns_field="content_vector",
            vector=random_vec,
            limit=fetch_count,
            output_fields=["id"],
            expr="is_active == true",
        )

        results = []
        for hit in hits:
            if exclude_ids and hit["id"] in exclude_ids:
                continue
            results.append({"id": hit["id"], "score": float(hit.score)})
            if len(results) >= limit:
                break

        return results

    def update_products(self, products: List[Dict], metadata_only: bool):
        """
        Update products in Milvus collection.

        Expected product structure:
        - id: string (UUID)
        - number: int64
        - name: string
        - description: string
        - brand: dict with 'name' key
        - category: dict with 'name' key
        - is_active: bool
        - rating: dict with 'score' key
        - skus: list
        - specifications: dict
        """

        upsert_products = []
        non_exist_ids = self.client.get_non_existing_ids(
            self.client.products_collection, [product.get("id") for product in products]
        )
        embeddings = {}

        if not metadata_only:
            embeddings = self.embedding_service.embed_texts(
                [
                    f"{product.get('name')} {product.get('description')}"
                    for product in products
                ]
            )
            embeddings = {
                product.get("id"): embedding
                for product, embedding in zip(products, embeddings)
            }

        for product in products:
            brand = product.get("brand") or {}
            category = product.get("category") or {}
            rating = product.get("rating") or {}

            update = {
                "id": product.get("id"),
                "number": product.get("number", 0),
                "name": product.get("name", ""),
                "description": product.get("description", ""),
                "brand": brand.get("name", ""),
                "category": category.get("name", ""),
                "is_active": product.get("is_active", False),
                "rating": rating.get("score", 0.0),
                "skus": product.get("skus") or [],
                "specifications": product.get("specifications") or {},
            }

            # If metadata only, skip if product does not exist
            if metadata_only and product.get("id") in non_exist_ids:
                continue

            # New products need a default popularity value
            if product.get("id") in non_exist_ids:
                update["popularity"] = 0.0

            if not metadata_only:
                embedding = embeddings[product.get("id")]
                update["sparse_vector"] = embedding.get("sparse")
                update["content_vector"] = embedding.get("dense")

            upsert_products.append(update)

        self.client.upsert(
            self.client.products_collection,
            upsert_products,
            partial_update=True,
        )

    def process_events(self, events: List[Dict]):
        """Process analytics events to update user interests and product popularity"""
        dim = self.client.content_dim

        # Group events by account_id
        account_events = defaultdict(list)
        for event in events:
            account_events[event.get("account_id")].append(event)

        # Aggregate product popularity across ALL events in this batch
        all_product_weights = aggregate_product_weights(events)

        # Update product popularity
        self._update_popularity(all_product_weights)

        # Fetch content vectors for all referenced products
        item_ids = set(event.get("ref_id") for event in events)
        item_vectors = self.client.get_vectors(
            self.client.products_collection,
            anns_field="content_vector",
            ids=item_ids,
            not_found_callback=lambda _: None,
        )

        # Fetch existing account interests
        account_ids = list(account_events.keys())
        existing_accounts = {}
        if account_ids:
            rows = self.client.accounts_collection.query(
                expr=f"id in {account_ids}",
                output_fields=["id"] + ACCOUNT_FIELDS + ["purchased_ids"],
            )
            for r in rows:
                existing_accounts[r["id"]] = r

        update_rows = []

        for account_id, acct_events in account_events.items():
            account = existing_accounts.get(account_id)

            # Load existing interests or init empty
            interests, strengths = default_interests(dim)
            purchased_ids = []
            if account:
                for i in range(NUM_INTERESTS):
                    vec = account.get(f"interest_{i+1}")
                    if vec is not None:
                        interests[i] = np.array(vec, dtype=np.float32)
                    strengths[i] = float(account.get(f"strength_{i+1}", 0.0))
                purchased_ids = list(account.get("purchased_ids") or [])

            # Collect new purchase IDs from this batch
            for e in acct_events:
                if (e.get("event_type") or "").lower() == "purchase" and e.get("ref_id"):
                    pid = e["ref_id"]
                    if pid in purchased_ids:
                        purchased_ids.remove(pid)
                    purchased_ids.append(pid)

            # Keep only the last N
            purchased_ids = purchased_ids[-MAX_PURCHASED_IDS:]

            # Aggregate event weights per product for this account
            product_weights = aggregate_product_weights(acct_events)

            # Assign each product interaction to closest interest
            for product_id, weight in product_weights.items():
                product_vec = item_vectors.get(product_id)
                if product_vec is None:
                    continue

                if weight > 0:
                    self._assign_positive(interests, strengths, product_vec, weight)
                elif weight < 0:
                    self._assign_negative(interests, strengths, product_vec, weight)

            # Build update row
            update_row = {
                "id": account_id,
                "number": acct_events[0].get("account_number", 0),
                "purchased_ids": purchased_ids,
            }
            for i in range(NUM_INTERESTS):
                update_row[f"interest_{i+1}"] = interests[i]
                update_row[f"strength_{i+1}"] = strengths[i]

            update_rows.append(update_row)

        self.client.upsert(self.client.accounts_collection, update_rows)

    def _update_popularity(self, product_weights: Dict[str, float]):
        """Update popularity scores for products referenced in events."""
        if not product_weights:
            return

        product_ids = list(product_weights.keys())
        rows = self.client.products_collection.query(
            expr=f"id in {product_ids}",
            output_fields=["id", "number", "popularity"],
        )
        current = {r["id"]: r for r in rows}

        updates = []
        for product_id, weight in product_weights.items():
            row = current.get(product_id)
            if row is None:
                continue
            old = float(row.get("popularity", 0.0))
            new = old * POPULARITY_DECAY + weight
            updates.append({"id": product_id, "number": row["number"], "popularity": max(0.0, new)})

        self.client.upsert(
            self.client.products_collection, updates, partial_update=True
        )

    @staticmethod
    def _assign_positive(
        interests: List[np.ndarray],
        strengths: List[float],
        product_vec: np.ndarray,
        weight: float,
    ):
        """Assign a positive interaction to the closest interest or create new."""
        best_idx, best_sim, empty_idx = -1, -1.0, -1

        for i in range(NUM_INTERESTS):
            if strengths[i] == 0:
                if empty_idx == -1:
                    empty_idx = i
                continue
            sim = cosine_sim(product_vec, interests[i])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim > MERGE_THRESHOLD:
            # Merge into existing interest
            alpha = max(MIN_ALPHA, weight / (strengths[best_idx] + weight))
            interests[best_idx] = alpha * product_vec + (1 - alpha) * interests[best_idx]
            strengths[best_idx] = min(strengths[best_idx] + weight, MAX_STRENGTH)
        elif empty_idx >= 0:
            # Create new interest
            interests[empty_idx] = product_vec.copy()
            strengths[empty_idx] = weight
        else:
            # All slots full → blend into weakest instead of hard replace
            weakest_idx = int(np.argmin(strengths))
            alpha = max(MIN_ALPHA, weight / (strengths[weakest_idx] + weight))
            interests[weakest_idx] = alpha * product_vec + (1 - alpha) * interests[weakest_idx]
            strengths[weakest_idx] = min(strengths[weakest_idx] + weight, MAX_STRENGTH)

    @staticmethod
    def _assign_negative(
        interests: List[np.ndarray],
        strengths: List[float],
        product_vec: np.ndarray,
        weight: float,
    ):
        """Reduce strength and push interest vector away from negative product."""
        best_idx, best_sim = -1, -1.0

        for i in range(NUM_INTERESTS):
            if strengths[i] == 0:
                continue
            sim = cosine_sim(product_vec, interests[i])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim > MERGE_THRESHOLD:
            # Push vector away from negative product
            alpha = max(MIN_ALPHA, abs(weight) / (strengths[best_idx] + abs(weight)))
            adjusted = interests[best_idx] - alpha * product_vec
            norm = np.linalg.norm(adjusted)
            if norm > 1e-9:
                interests[best_idx] = adjusted / norm
            # Reduce strength
            strengths[best_idx] = max(0.0, strengths[best_idx] + weight)
