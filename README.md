# ShopNexus Embedding

A lightweight embedding microservice for ShopNexus. It exposes a single `/embed` endpoint that converts text into dense and sparse vectors using [BGE-M3](https://huggingface.co/BAAI/bge-m3).

## API

### `POST /embed`

Accepts a list of strings and returns dense + sparse embeddings for each.

**Request:**

```json
{
  "texts": ["running shoes", "wireless headphones"]
}
```

**Response:**

```json
{
  "embeddings": [
    {
      "dense": [0.012, -0.034, ...],
      "sparse": {"142": 0.85, "1024": 0.42, ...}
    },
    ...
  ]
}
```

### `GET /health`

Returns `{"status": "healthy"}`.

## Setup

Requires Python 3.13+.

```bash
# Install dependencies
uv sync

# Run the server
uv run python main.py
```

The server starts on `http://localhost:8000`.

## Tech Stack

- **Model:** BGE-M3 (dense + sparse embeddings)
- **Framework:** Flask
- **Dependencies:** FlagEmbedding, PyMilvus Model, Transformers, PyTorch

<img width="1200" height="630" alt="Image" src="https://github.com/user-attachments/assets/e8ea3d12-b545-4ad3-a78a-e431e95e9b52" />
