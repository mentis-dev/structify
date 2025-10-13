import asyncio
from typing import List, Dict, Any, Optional, Union
import numpy as np

class AsyncEmbeddingProcessor:
    """
    Asynchronous processor for generating embeddings efficiently.
    Handles batching, rate limiting, and retries.
    """

    def __init__(
            self,
            embedding_model,
            batch_size: int = 20,
            max_retries: int = 3,
            retry_delay: float = 1.0,
            max_concurrency: int = 5
    ):
        """
        Initialize the async embedding processor.

        Args:
            embedding_model: The embedding model to use (must have embed_documents method)
            batch_size: Size of batches to send to the embedding API
            max_retries: Maximum number of retries for failed requests
            retry_delay: Initial delay between retries (doubles each retry)
            max_concurrency: Maximum number of concurrent batches
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def embed_documents(
            self,
            documents: List[Dict],
            content_field: str = "page_content",
            id_field: str = "id"
    ) -> Dict[str, Any]:
        """
        Asynchronously embed multiple documents in optimized batches.

        Args:
            documents: List of document dictionaries
            content_field: Field containing the text to embed
            id_field: Field containing document ID

        Returns:
            Dictionary mapping document IDs to embeddings
        """
        # Group into batches
        batches = []
        current_batch = []

        for doc in documents:
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
            current_batch.append(doc)

        if current_batch:
            batches.append(current_batch)

        # Process batches concurrently with controlled concurrency
        tasks = []
        for batch in batches:
            tasks.append(self._process_batch(batch, content_field, id_field))

        # Wait for all batches to complete
        results = await asyncio.gather(*tasks)

        # Combine results from all batches
        combined_results = {}
        for result in results:
            combined_results.update(result)

        return combined_results

    async def _process_batch(
            self,
            batch: List[Dict],
            content_field: str,
            id_field: str
    ) -> Dict[str, Any]:
        """Process a single batch of documents."""
        async with self.semaphore:
            # Extract texts and IDs
            texts = [doc[content_field] for doc in batch if doc.get(content_field)]
            ids = [doc[id_field] for doc in batch if doc.get(id_field)]

            if not texts:
                return {}

            # Try to embed with retries
            for attempt in range(self.max_retries + 1):
                try:
                    # Some embedding models might have async methods
                    if hasattr(self.embedding_model, "aembed_documents"):
                        embeddings = await self.embedding_model.aembed_documents(texts)
                    else:
                        # Run in executor if only sync method available
                        loop = asyncio.get_event_loop()
                        embeddings = await loop.run_in_executor(
                            None, self.embedding_model.embed_documents, texts
                        )

                    # Create ID -> embedding mapping
                    result = {}
                    for doc_id, embedding in zip(ids, embeddings):
                        result[doc_id] = embedding

                    return result

                except Exception as e:
                    if attempt < self.max_retries:
                        # Exponential backoff
                        delay = self.retry_delay * (2 ** attempt)
                        print(f"Embedding attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s")
                        await asyncio.sleep(delay)
                    else:
                        print(f"All embedding attempts failed for batch: {str(e)}")
                        return {}  # Return empty dict on failure

    @staticmethod
    async def create_batches_by_token_count(
            documents: List[Dict],
            tokenizer,
            target_tokens: int = 8000,
            content_field: str = "page_content",
            max_batch_size: int = 50
    ) -> List[List[Dict]]:
        """
        Create batches optimized by token count rather than document count.
        Useful for embedding models that have token-based pricing.

        Args:
            documents: List of document dictionaries
            tokenizer: Tokenizer function/object with encode method
            target_tokens: Target token count per batch
            content_field: Field containing the text to embed
            max_batch_size: Maximum documents per batch regardless of tokens

        Returns:
            List of batches (each batch is a list of documents)
        """
        batches = []
        current_batch = []
        current_tokens = 0

        for doc in documents:
            content = doc.get(content_field, "")
            # Get token count
            token_count = len(tokenizer.encode(content))

            # If adding this document would exceed target or we hit max size,
            # finish current batch and start a new one
            if (current_tokens + token_count > target_tokens and current_batch) or \
                    len(current_batch) >= max_batch_size:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            # Add document to current batch
            current_batch.append(doc)
            current_tokens += token_count

        # Add the last batch if it's not empty
        if current_batch:
            batches.append(current_batch)

        return batches


class EmbeddingCache:
    """
    A memory-efficient cache for document embeddings with LRU eviction policy.
    Provides dictionary-like interface for storing and retrieving embeddings.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize the embedding cache.

        Args:
            max_size: Maximum number of embeddings to keep in cache
        """
        self._cache = {}  # doc_id -> embedding
        self.max_size = max_size
        self._access_order = []  # Least recently used at the start

    def get(self, doc_id: str) -> Optional[np.ndarray]:
        """Get embedding from cache if it exists."""
        if doc_id in self._cache:
            # Update access order (move to end)
            self._access_order.remove(doc_id)
            self._access_order.append(doc_id)
            return self._cache[doc_id]
        return None

    def put(self, doc_id: str, embedding: Union[List[float], np.ndarray]) -> None:
        """Add embedding to cache, evicting oldest if needed."""
        # Convert to numpy array if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)

        # Add to cache
        self._cache[doc_id] = embedding

        # Update access order
        if doc_id in self._access_order:
            self._access_order.remove(doc_id)
        self._access_order.append(doc_id)

        # Evict if over capacity
        while len(self._cache) > self.max_size:
            oldest_id = self._access_order.pop(0)
            if oldest_id in self._cache:
                del self._cache[oldest_id]

    def __len__(self) -> int:
        """Get number of items in cache."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()

    def memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        # Average embedding is 1536 floats * 4 bytes per float
        return len(self._cache) * (1536 * 4 + 64)  # Adding overhead for keys

    # Optional: Add dict-like methods for convenience
    def __getitem__(self, key):
        """Support dict-like access with [] operator."""
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __contains__(self, key):
        """Support 'in' operator for checking cache membership."""
        return key in self._cache

    def keys(self):
        """Return the keys in the cache."""
        return self._cache.keys()

    def values(self):
        """Return the values in the cache."""
        return self._cache.values()

    def items(self):
        """Return the items in the cache."""
        return self._cache.items()