import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Set
from config import NUM_INTERESTS
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)

logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(
        self,
        content_dim: int,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
    ):
        self.content_dim = content_dim
        connections.connect(host=milvus_host, port=milvus_port)
        self._setup_collections()

    def _setup_collections(self):
        self.products_schema = {
            "name": "products",
            "description": "Products with embeddings for semantic search",
            "schema": [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True, auto_id=False),
                FieldSchema(name="number", dtype=DataType.INT64),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=10240),
                FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="is_active", dtype=DataType.BOOL),
                FieldSchema(name="rating", dtype=DataType.FLOAT),
                FieldSchema(name="skus", dtype=DataType.JSON),
                FieldSchema(name="specifications", dtype=DataType.JSON),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=self.content_dim),
            ],
            "indexes": [
                {"field_name": "sparse_vector", "index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"},
                {"field_name": "content_vector", "index_type": "AUTOINDEX", "metric_type": "COSINE"},
            ],
        }

        interest_fields = []
        interest_indexes = []
        for i in range(1, NUM_INTERESTS + 1):
            interest_fields.append(FieldSchema(name=f"interest_{i}", dtype=DataType.FLOAT_VECTOR, dim=self.content_dim))
            interest_fields.append(FieldSchema(name=f"strength_{i}", dtype=DataType.FLOAT))
            interest_indexes.append({"field_name": f"interest_{i}", "index_type": "AUTOINDEX", "metric_type": "COSINE"})

        self.accounts_schema = {
            "name": "accounts",
            "description": "Accounts with multi-interest embeddings",
            "schema": [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True, auto_id=False),
                FieldSchema(name="number", dtype=DataType.INT64),
                *interest_fields,
            ],
            "indexes": interest_indexes,
        }

        self.products_collection = self._setup_collection(self.products_schema)
        self.accounts_collection = self._setup_collection(self.accounts_schema)

    def _setup_collection(self, schema: dict):
        collection = Collection(
            name=schema["name"],
            schema=CollectionSchema(schema["schema"], description=schema["description"]),
        )
        for index in schema["indexes"]:
            collection.create_index(
                index["field_name"],
                {"index_type": index["index_type"], "metric_type": index["metric_type"]},
            )
        collection.load()
        return collection

    def semantic_search(
        self, content_vec, sparse_vec, dense_weight=1.0, sparse_weight=1.0, offset=0, limit=10,
    ):
        """Hybrid search combining dense + sparse vectors."""
        dense_req = AnnSearchRequest([content_vec], "content_vector", {"metric_type": "COSINE"}, limit=limit)
        sparse_req = AnnSearchRequest([sparse_vec], "sparse_vector", {"metric_type": "IP"}, limit=limit)

        return self.products_collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=WeightedRanker(sparse_weight, dense_weight),
            limit=limit,
            output_fields=["id"],
            **{"offset": offset},
        )

    def get_vectors(
        self,
        collection: Collection,
        anns_field: str,
        ids: Set[str],
        not_found_callback: Callable[[str], Optional[np.ndarray]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """Get vectors by IDs. Uses not_found_callback for missing IDs."""
        results = collection.query(
            expr=f"id in {list(ids)}",
            output_fields=["id", anns_field],
        )
        found = {r["id"]: np.array(r[anns_field]) for r in results}
        not_found_ids = ids - set(found.keys())

        if not_found_ids:
            logger.warning(f"IDs not found in collection: {not_found_ids}")
            if not_found_callback:
                for nf_id in not_found_ids:
                    found[nf_id] = not_found_callback(nf_id)
            else:
                raise ValueError(f"IDs not found in collection: {not_found_ids}")

        return found

    def upsert(self, collection: Collection, entities: List[Dict], partial_update: bool = True):
        if not entities:
            return
        collection.upsert(entities, None, None, **{"partial_update": partial_update})
        collection.flush()

    def get_non_existing_ids(self, collection: Collection, ids: List[str]) -> set[str]:
        results = collection.query(expr=f"id in {ids}", output_fields=["id"])
        existing = {r["id"] for r in results}
        return set(ids) - existing
