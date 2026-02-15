import logging
import numpy as np
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
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
    """Milvus client for managing connections, collections, and operations"""

    def __init__(
        self,
        cf_dim: int,
        content_dim: int,
        fused_dim: int,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
    ):
        """Initialize Milvus connection and setup collections"""
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port

        self.cf_dim = cf_dim
        self.content_dim = content_dim
        self.fused_dim = fused_dim

        # Setup connection and collections
        connections.connect(host=self.milvus_host, port=self.milvus_port)

        self._setup_collections()

    def _setup_collections(self):
        self.products_schema = {
            "name": "products",
            "description": "Products with embeddings for semantic search and recommendation",
            "schema": [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True, auto_id=False
                ),
                FieldSchema(name="number", dtype=DataType.INT64),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(
                    name="description", dtype=DataType.VARCHAR, max_length=10240
                ),
                FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="is_active", dtype=DataType.BOOL),
                FieldSchema(name="rating", dtype=DataType.FLOAT),
                FieldSchema(name="skus", dtype=DataType.JSON),
                FieldSchema(name="specifications", dtype=DataType.JSON),
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
                FieldSchema(
                    name="content_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.content_dim,
                ),
                FieldSchema(
                    name="cf_vector", dtype=DataType.FLOAT_VECTOR, dim=self.cf_dim
                ),
                FieldSchema(
                    name="fused_vector", dtype=DataType.FLOAT_VECTOR, dim=self.fused_dim
                ),
            ],
            "indexes": [
                {
                    "field_name": "sparse_vector",
                    "index_type": "SPARSE_INVERTED_INDEX",
                    "metric_type": "IP",
                },
                {
                    "field_name": "content_vector",
                    "index_type": "AUTOINDEX",
                    "metric_type": "COSINE",
                },
                {
                    "field_name": "cf_vector",
                    "index_type": "AUTOINDEX",
                    "metric_type": "COSINE",
                },
                {
                    "field_name": "fused_vector",
                    "index_type": "AUTOINDEX",
                    "metric_type": "COSINE",
                },
            ],
        }

        self.accounts_schema = {
            "name": "accounts",
            "description": "Accounts with embeddings for recommendations",
            "schema": [
                FieldSchema(
                    name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True, auto_id=False
                ),
                FieldSchema(name="number", dtype=DataType.INT64),
                FieldSchema(
                    name="cf_vector", dtype=DataType.FLOAT_VECTOR, dim=self.cf_dim
                ),
                FieldSchema(
                    name="fused_vector", dtype=DataType.FLOAT_VECTOR, dim=self.fused_dim
                ),
            ],
            "indexes": [
                {
                    "field_name": "cf_vector",
                    "index_type": "AUTOINDEX",
                    "metric_type": "COSINE",
                },
                {
                    "field_name": "fused_vector",
                    "index_type": "AUTOINDEX",
                    "metric_type": "COSINE",
                },
            ],
        }

        self.products_collection = self._setup_collection(self.products_schema)
        self.accounts_collection = self._setup_collection(self.accounts_schema)

    def _setup_collection(self, schema: dict):
        """Setup collection for user vectors (fused embeddings)"""

        collection = Collection(
            name=schema["name"],
            schema=CollectionSchema(
                schema["schema"], description=schema["description"]
            ),
        )
        for index in schema["indexes"]:
            collection.create_index(
                index["field_name"],
                {
                    "index_type": index["index_type"],
                    "metric_type": index["metric_type"],
                },
            )
        collection.load()
        return collection

    def semantic_search(
        self,
        content_vec,
        sparse_vec,
        dense_weight=1.0,
        sparse_weight=1.0,
        offset=0,
        limit=10,
    ):
        """Hybrid search in content_products collection"""
        dense_req = AnnSearchRequest(
            [content_vec], "content_vector", {"metric_type": "COSINE"}, limit=limit
        )
        sparse_req = AnnSearchRequest(
            [sparse_vec], "sparse_vector", {"metric_type": "IP"}, limit=limit
        )
        rerank = WeightedRanker(sparse_weight, dense_weight)

        return self.products_collection.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=limit,
            output_fields=["id"],
            **{"offset": offset},
        )

    def best_similar(
        self, collection: Collection, anns_field: str, vector: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Search best similar vector in collection"""
        return self.search(collection, anns_field, vector, limit=1)[0]

    def search(
        self,
        collection: Collection,
        anns_field: str,
        vector: np.ndarray,
        limit: int = 10,
        output_fields: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search similar vectors in collection"""
        results = collection.search(
            data=[vector.tolist()],
            anns_field=anns_field,
            param={"metric_type": "COSINE"},
            limit=limit,
            output_fields=output_fields or ["id", anns_field],
        )
        return results[0]

    def get_vectors(
        self,
        collection: Collection,
        anns_field: str,
        ids: Set[str],
        not_found_callback: Callable[[str], np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Get vectors from collection by list of IDs, if not found use the vector from not_found_callback"""
        results = collection.query(
            expr=f"id in {list(ids)}",
            output_fields=["id", anns_field],
        )
        not_found_ids = ids - set([result.get("id") for result in results])

        if len(not_found_ids) > 0:
            logger.warning(f"Some IDs are not found in collection: {not_found_ids}")
            if not_found_callback:
                not_found_vectors = {id: not_found_callback(id) for id in not_found_ids}
                return {
                    **{
                        result.get("id"): np.array(result.get(anns_field))
                        for result in results
                    },
                    **not_found_vectors,
                }
            else:
                raise ValueError(
                    f"Some IDs are not found in collection: {not_found_ids}"
                )

        return {
            result.get("id"): np.array(result.get(anns_field)) for result in results
        }

    def get_vector(
        self, collection: Collection, id: str, anns_field: str
    ) -> Optional[np.ndarray]:
        """Get vector from collection by ID and anns_field"""
        results = collection.query(expr=f"id == '{id}'", output_fields=[anns_field])
        return np.array(results[0].get(anns_field)) if results else None

    def upsert(
        self, collection: Collection, entities: List[List], partial_update: bool = True
    ):
        """Upsert entities into collection"""
        if len(entities) == 0:
            return
        collection.upsert(entities, None, None, **{"partial_update": partial_update})
        collection.flush()

    def get_rows(
        self, collection: Collection, ids: List[str], output_fields: List[str]
    ) -> List[Dict]:
        """Get rows from collection by list of IDs"""
        return collection.query(expr=f"id in {ids}", output_fields=output_fields)

    def get_non_existing_ids(self, collection: Collection, ids: List[str]) -> set[str]:
        """Get non-existing IDs from collection"""
        results = self.get_rows(collection, ids, ["id"])
        return set(
            [id for id in ids if id not in [result.get("id") for result in results]]
        )
