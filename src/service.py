"""Service layer for embedding-based search and recommendation"""

import logging
from typing import List, Dict
from collections import defaultdict
import numpy as np
from milvus import MilvusClient
from pymilvus import AnnSearchRequest, WeightedRanker
from embeddings import EmbeddingService
from utils import aggregate_product_weights, cosine_sim, default_interests
from config import NUM_INTERESTS, MERGE_THRESHOLD, MAX_STRENGTH, MIN_ALPHA

logger = logging.getLogger(__name__)

INTEREST_FIELDS = [f"interest_{i+1}" for i in range(NUM_INTERESTS)]
STRENGTH_FIELDS = [f"strength_{i+1}" for i in range(NUM_INTERESTS)]
ACCOUNT_FIELDS = INTEREST_FIELDS + STRENGTH_FIELDS


class Service:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        self.embedding_service = EmbeddingService()
        self.client = MilvusClient(
            content_dim=self.embedding_service.dense_dim,
            milvus_host=milvus_host,
            milvus_port=milvus_port,
        )

    def semantic_search(
        self, query: str, dense_weight=1.0, sparse_weight=1.0, offset=0, limit=10
    ):
        """Hybrid semantic search using dense + sparse vectors."""
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
        """Recommend products based on user interest vectors."""
        results = self.client.accounts_collection.query(
            expr=f"id == '{account_id}'",
            output_fields=ACCOUNT_FIELDS,
        )

        if not results:
            return []

        account = results[0]
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
                    limit=limit,
                    expr="is_active == true",
                )
                search_reqs.append(req)
                weights.append(strength)

        if not search_reqs:
            return []

        max_w = max(weights)
        normalized = [w / max_w for w in weights]
        rerank = WeightedRanker(*normalized)

        hits = self.client.products_collection.hybrid_search(
            search_reqs,
            rerank=rerank,
            limit=limit,
            output_fields=["id"],
        )

        return [{"id": hit["id"], "score": float(hit.score)} for hit in hits[0]]

    def update_products(self, products: List[Dict], metadata_only: bool):
        """Embed and upsert products into Milvus."""
        upsert_products = []
        non_exist_ids = self.client.get_non_existing_ids(
            self.client.products_collection,
            [p.get("id") for p in products],
        )
        embeddings = {}

        if not metadata_only:
            embeddings = self.embedding_service.embed_texts(
                [f"{p.get('name')} {p.get('description')}" for p in products]
            )
            embeddings = {
                p.get("id"): emb for p, emb in zip(products, embeddings)
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

            if metadata_only and product.get("id") in non_exist_ids:
                continue

            if not metadata_only:
                embedding = embeddings[product.get("id")]
                update["sparse_vector"] = embedding.get("sparse")
                update["content_vector"] = embedding.get("dense")

            upsert_products.append(update)

        self.client.upsert(
            self.client.products_collection, upsert_products, partial_update=True
        )

    def process_events(self, events: List[Dict]):
        """Process events to update user interest vectors."""
        dim = self.client.content_dim

        # Group events by account
        account_events = defaultdict(list)
        for event in events:
            account_events[event.get("account_id")].append(event)

        # Fetch content vectors for all referenced products
        item_ids = set(e.get("ref_id") for e in events)
        item_vectors = self.client.get_vectors(
            self.client.products_collection,
            anns_field="content_vector",
            ids=item_ids,
            not_found_callback=lambda _: None,
        )

        # Fetch existing accounts
        account_ids = list(account_events.keys())
        existing_accounts = {}
        if account_ids:
            rows = self.client.accounts_collection.query(
                expr=f"id in {account_ids}",
                output_fields=["id"] + ACCOUNT_FIELDS,
            )
            existing_accounts = {r["id"]: r for r in rows}

        update_rows = []

        for account_id, acct_events in account_events.items():
            account = existing_accounts.get(account_id)

            interests, strengths = default_interests(dim)
            if account:
                for i in range(NUM_INTERESTS):
                    vec = account.get(f"interest_{i+1}")
                    if vec is not None:
                        interests[i] = np.array(vec, dtype=np.float32)
                    strengths[i] = float(account.get(f"strength_{i+1}", 0.0))

            # Aggregate event weights per product
            product_weights = aggregate_product_weights(acct_events)

            for product_id, weight in product_weights.items():
                product_vec = item_vectors.get(product_id)
                if product_vec is None:
                    continue

                if weight > 0:
                    self._assign_positive(interests, strengths, product_vec, weight)
                elif weight < 0:
                    self._assign_negative(interests, strengths, product_vec, weight)

            update_row = {
                "id": account_id,
                "number": acct_events[0].get("account_number", 0),
            }
            for i in range(NUM_INTERESTS):
                update_row[f"interest_{i+1}"] = interests[i]
                update_row[f"strength_{i+1}"] = strengths[i]

            update_rows.append(update_row)

        self.client.upsert(self.client.accounts_collection, update_rows)

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
            alpha = max(MIN_ALPHA, weight / (strengths[best_idx] + weight))
            interests[best_idx] = alpha * product_vec + (1 - alpha) * interests[best_idx]
            strengths[best_idx] = min(strengths[best_idx] + weight, MAX_STRENGTH)
        elif empty_idx >= 0:
            interests[empty_idx] = product_vec.copy()
            strengths[empty_idx] = weight
        else:
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
        """Push interest vector away from negative product."""
        best_idx, best_sim = -1, -1.0

        for i in range(NUM_INTERESTS):
            if strengths[i] == 0:
                continue
            sim = cosine_sim(product_vec, interests[i])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_idx >= 0 and best_sim > MERGE_THRESHOLD:
            alpha = max(MIN_ALPHA, abs(weight) / (strengths[best_idx] + abs(weight)))
            adjusted = interests[best_idx] - alpha * product_vec
            norm = np.linalg.norm(adjusted)
            if norm > 1e-9:
                interests[best_idx] = adjusted / norm
            strengths[best_idx] = max(0.0, strengths[best_idx] + weight)
