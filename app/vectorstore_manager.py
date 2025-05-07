import os
import uuid
import pickle
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import concurrent.futures

import numpy as np
import faiss
from langchain.schema import Document as LangChainDocument
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from big_raptor.embedding_cache import EmbeddingCache, AsyncEmbeddingProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorStoreManager")


@dataclass
class Query:
    """Query object with text and optional parameters."""
    text: str
    top_k: Optional[int] = None
    metadata_filter: Optional[Dict[str, Any]] = None


class VectorStoreManager:
    """
    Manages FAISS vector indexes for efficient document storage and retrieval.
    Optimized for large corpora (tens of thousands of documents).
    """

    def __init__(
            self,
            index_name: str,
            embedding_dim: int = 1536,  # OpenAI embedding dimension
            embeddings: Optional[OpenAIEmbeddings] = None,
            embedding_model: str = "text-embedding-3-small",
            index_directory: str = "./vector_indexes",
            use_gpu: bool = False,
            chunk_size: int = 512,
            chunk_overlap: int = 50,
            cache_size: int = 10000,  # Size of embedding cache
            auto_save: bool = True,
            **kwargs
    ):
        """
        Initialize the FAISS vector store manager.

        :param index_name: Name for this index
        :param embedding_dim: Dimension of embeddings
        :param embedding_model: Model for embedding text (default: OpenAIEmbeddings)
        :param index_directory: Directory to store FAISS indexes
        :param use_gpu: Whether to use GPU for FAISS (if available)
        :param chunk_size: Default chunk size for document splitting
        :param chunk_overlap: Default chunk overlap for document splitting
        :param auto_save: Whether to automatically save after operations
        """
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.embeddings = embeddings or OpenAIEmbeddings(model=embedding_model)
        self.index_directory = index_directory
        self.use_gpu = use_gpu
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.auto_save = auto_save

        # Create storage directory
        os.makedirs(self.index_directory, exist_ok=True)

        # Initialize namespaces for organizing indexes
        self.namespaces = {}
        self.indexes = {}
        self.document_store = {}  # doc_id -> LangChainDocument mapping

        # Initialize splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # Initialize embedding cache
        self._embedding_cache = EmbeddingCache(max_size=cache_size)

        # Initialize async embedding processor
        self._async_embedder = AsyncEmbeddingProcessor(
            embedding_model=self.embeddings,
            batch_size=20,
            max_retries=3,
            max_concurrency=5
        )
        # Try to load existing index
        self._load_or_create_index()

        logger.info(f"Initialized VectorStoreManager with index '{index_name}'")

    def _get_index_path(self, namespace: str = "") -> str:
        """Get path for FAISS index file."""
        namespace = namespace or "default"
        return os.path.join(self.index_directory, f"{self.index_name}_{namespace}.faiss")

    def _get_metadata_path(self, namespace: str = "") -> str:
        """Get path for metadata store file."""
        namespace = namespace or "default"
        return os.path.join(self.index_directory, f"{self.index_name}_{namespace}_metadata.pkl")

    def _load_or_create_index(self):
        """Load existing indexes or create new ones."""
        # Check if index directory exists
        initial_namespaces = ["default"]

        # Look for existing index files
        for filename in os.listdir(self.index_directory):
            if filename.startswith(f"{self.index_name}_") and filename.endswith(".faiss"):
                # Extract namespace from filename
                namespace = filename[len(self.index_name) + 1:-6]  # Remove prefix and .faiss
                if namespace not in initial_namespaces:
                    initial_namespaces.append(namespace)

        # Load or create indexes for each namespace
        for namespace in initial_namespaces:
            self._load_or_create_namespace(namespace)

    def _load_or_create_namespace(self, namespace: str):
        """Load or create a specific namespace index."""
        index_path = self._get_index_path(namespace)
        metadata_path = self._get_metadata_path(namespace)

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                # Load existing index
                self.indexes[namespace] = faiss.read_index(index_path)

                # Load document store and metadata
                with open(metadata_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.document_store[namespace] = saved_data.get('documents', {})
                    self.namespaces[namespace] = saved_data.get('metadata', {})

                doc_count = len(self.document_store[namespace])
                logger.info(f"Loaded namespace '{namespace}' with {doc_count} documents")

            except Exception as e:
                logger.error(f"Error loading index for namespace '{namespace}': {str(e)}")
                self._create_new_index(namespace)
        else:
            self._create_new_index(namespace)

    def _create_new_index(self, namespace: str):
        """Create a new FAISS index for a namespace."""
        # Create a flat L2 index
        index = faiss.IndexFlatL2(self.embedding_dim)

        # Wrap with an IndexIDMap to store doc IDs
        index = faiss.IndexIDMap(index)

        # Use GPU if available and requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            try:
                gpu_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
                logger.info(f"Using GPU for FAISS index in namespace '{namespace}'")
            except Exception as e:
                logger.warning(f"Failed to use GPU for FAISS: {str(e)}")

        self.indexes[namespace] = index
        self.document_store[namespace] = {}
        self.namespaces[namespace] = {'total_docs': 0}

        logger.info(f"Created new index for namespace '{namespace}'")

    def persist(self):
        """Save all indexes and metadata to disk."""
        start_time = time.time()
        logger.info("Saving all indexes to disk...")

        for namespace in self.indexes:
            self._save_namespace(namespace)

        duration = time.time() - start_time
        logger.info(f"All indexes saved in {duration:.2f} seconds")

    def _save_namespace(self, namespace: str):
        """Save a specific namespace index and metadata."""
        if namespace not in self.indexes:
            logger.warning(f"Cannot save namespace '{namespace}': not found")
            return

        try:
            # Save FAISS index
            index_path = self._get_index_path(namespace)

            # Convert GPU index to CPU if needed
            index_to_save = self.indexes[namespace]
            if self.use_gpu:
                try:
                    index_to_save = faiss.index_gpu_to_cpu(index_to_save)
                except:
                    pass  # Already a CPU index or conversion failed

            faiss.write_index(index_to_save, index_path)

            # Save document store and metadata
            metadata_path = self._get_metadata_path(namespace)
            save_data = {
                'documents': self.document_store[namespace],
                'metadata': self.namespaces[namespace]
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(save_data, f)

            logger.info(f"Saved namespace '{namespace}' with {len(self.document_store[namespace])} documents")

        except Exception as e:
            logger.error(f"Error saving namespace '{namespace}': {str(e)}")

    def _generate_doc_id(self) -> int:
        """Generate a unique numeric document ID for FAISS."""
        # FAISS requires integer IDs
        return int(uuid.uuid4().int & 0xFFFFFFFF)

    def _chunk_document(self, document: LangChainDocument) -> List[LangChainDocument]:
        """Split a document into chunks."""
        chunks = self.text_splitter.split_text(document.page_content)
        chunked_docs = []

        for i, chunk in enumerate(chunks):
            # Create a new document for each chunk
            doc_id = document.id or document.metadata.get("document_id", str(uuid.uuid4()))
            chunk_id = f"{doc_id}_chunk_{i}"

            # Copy metadata and add chunk info
            metadata = document.metadata.copy() if document.metadata else {}
            metadata["chunk_id"] = chunk_id
            metadata["chunk_index"] = i
            metadata["source_id"] = doc_id
            metadata["total_chunks"] = len(chunks)

            chunked_docs.append(
                LangChainDocument(page_content=chunk, metadata=metadata)
            )

        return chunked_docs

    async def upsert_documents(
            self,
            documents: List[LangChainDocument],
            namespace: str = "",
            do_chunking: bool = True,  # Default True, but now configurable
            auto_delete: bool = False,  # Default False, but now configurable
            level: Optional[int] = None,
            show_progress: bool = True,
            max_workers: int = 8,
            batch_size: int = 50,  # Size of batches for embedding API calls
            save_interval: int = 1000,  # Save after processing this many docs
            enable_cache: bool = True
    ) -> Tuple[int, int]:
        """
        Add or update documents in the vector store with optimized performance.
        Maintains strict ID consistency between documents and index.

        Args:
            documents: List of documents to add
            namespace: Target namespace
            do_chunking: Whether to split documents into chunks
            auto_delete: If True, delete existing docs with same IDs
            level: If specified, set this level value in metadata
            show_progress: Show progress bars
            max_workers: Maximum number of parallel workers for chunking and processing
            batch_size: Size of batches for embedding API calls
            save_interval: How often to auto-save during large operations
            enable_cache: Whether to use embedding cache

        Returns:
            Tuple containing (number of documents processed, number of embeddings generated)
        """
        start_time = time.time()
        namespace = namespace or "default"
        embedding_cache = {}  # Local cache for this operation
        documents_processed = 0
        embeddings_generated = 0

        # Ensure namespace exists
        if namespace not in self.indexes:
            self._create_new_index(namespace)

        # 1. Validate input documents
        if not documents:
            logger.warning("No documents provided to upsert")
            return 0, 0

        logger.info(f"Processing {len(documents)} documents for namespace '{namespace}'")
        logger.info(f"Settings: do_chunking={do_chunking}, auto_delete={auto_delete}")

        # 2. Process documents in parallel (chunking if needed)
        all_docs = []
        original_ids = {}  # Track original doc IDs for chunks

        if do_chunking and len(documents) > 1:
            # First ensure all source documents have IDs
            for i, doc in enumerate(documents):
                if not doc.id:
                    if "document_id" in doc.metadata:
                        doc.id = doc.metadata["document_id"]
                    else:
                        doc_id = str(uuid.uuid4())
                        doc.id = doc_id
                        doc.metadata["document_id"] = doc_id

                # Store original document ID for later reference
                original_ids[doc.id] = doc.id

            # Chunk in parallel for better performance
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                chunking_tasks = []
                for doc in documents:
                    chunking_tasks.append(executor.submit(self._chunk_document_with_id, doc))

                if show_progress:
                    chunked_results = list(
                        tqdm(
                            (task.result() for task in as_completed(chunking_tasks)),
                            total=len(chunking_tasks),
                            desc="Chunking documents"
                        )
                    )
                else:
                    chunked_results = [task.result() for task in chunking_tasks]

                for chunks in chunked_results:
                    all_docs.extend(chunks)
        else:
            # Either no chunking needed or just one document
            for doc in documents:
                # Ensure document has ID
                if not doc.id:
                    if "document_id" in doc.metadata:
                        doc.id = doc.metadata["document_id"]
                    else:
                        doc_id = str(uuid.uuid4())
                        doc.id = doc_id
                        doc.metadata["document_id"] = doc_id

                # Store original document ID
                original_ids[doc.id] = doc.id

                if do_chunking:
                    chunks = self._chunk_document_with_id(doc)
                    all_docs.extend(chunks)
                else:
                    all_docs.append(doc)

        if not all_docs:
            logger.warning("No valid documents after processing")
            return 0, 0

        documents_processed = len(all_docs)
        logger.info(f"Processing {documents_processed} document chunks")

        # 3. Update metadata and prepare documents
        # Set level if provided
        if level is not None:
            for doc in all_docs:
                doc.metadata["level"] = level

        # 4. Prepare document IDs and handle auto-delete
        doc_id_map = {}  # Map string IDs to FAISS integer IDs
        docs_by_id = {}  # Store docs by ID for batch processing

        # Tracking for existing vs. new docs
        existing_ids = set()
        to_update_ids = set()
        new_ids = set()

        # Check if documents are updates to existing ones
        for doc in all_docs:
            # Ensure the ID is properly set in both the document and its metadata
            doc_id = doc.id
            if not doc_id:
                # This shouldn't happen at this point, but just in case
                doc_id = str(uuid.uuid4())
                doc.id = doc_id

            # Ensure metadata has the same document_id
            doc.metadata["document_id"] = doc_id

            # Check if this document exists in the store
            if doc_id in self.document_store.get(namespace, {}):
                existing_ids.add(doc_id)
                if auto_delete:
                    to_update_ids.add(doc_id)
            else:
                new_ids.add(doc_id)

            docs_by_id[doc_id] = doc

        # 5. Handle deletions more efficiently
        if to_update_ids and auto_delete:
            # Remember FAISS IDs for documents being updated
            faiss_ids_to_remove = []
            for doc_id in to_update_ids:
                if doc_id in self.document_store.get(namespace, {}):
                    faiss_ids_to_remove.append(self.document_store[namespace][doc_id]['faiss_id'])

            # Batch remove from FAISS
            if faiss_ids_to_remove:
                try:
                    self.indexes[namespace].remove_ids(np.array(faiss_ids_to_remove, dtype='int64'))
                    logger.info(f"Removed {len(faiss_ids_to_remove)} existing documents for update")
                except Exception as e:
                    logger.error(f"Error removing existing documents: {str(e)}")

        # 6. Group documents by size for more efficient embedding
        # Small docs (<200 chars), medium (200-1000), large (>1000)
        small_docs, medium_docs, large_docs = [], [], []
        for doc_id, doc in docs_by_id.items():
            length = len(doc.page_content)
            if length < 200:
                small_docs.append((doc_id, doc))
            elif length < 1000:
                medium_docs.append((doc_id, doc))
            else:
                large_docs.append((doc_id, doc))

        # 7. Process documents in batches by size groups
        all_embeddings = []
        all_doc_ids = []
        all_faiss_ids = []
        all_content_docs = []

        # Process size groups separately with appropriate batch sizes
        for group_name, group_docs, group_batch_size in [
            ("small", small_docs, batch_size * 2),
            ("medium", medium_docs, batch_size),
            ("large", large_docs, max(1, batch_size // 2))
        ]:
            if not group_docs:
                continue

            logger.info(f"Processing {len(group_docs)} {group_name} documents")

            # Process in batches
            for i in range(0, len(group_docs), group_batch_size):
                batch = group_docs[i:i + group_batch_size]

                # Prepare batch data
                batch_doc_ids = []
                batch_faiss_ids = []
                batch_docs = []
                docs_to_embed = []

                for doc_id, doc in batch:
                    # Generate FAISS numeric ID
                    faiss_id = self._generate_doc_id()

                    # Check cache first if enabled
                    if enable_cache and hasattr(self, '_embedding_cache'):
                        cached_embedding = self._embedding_cache.get(doc_id) if hasattr(self,
                                                                                        '_embedding_cache') else None
                        if cached_embedding is not None:
                            # Use cached embedding
                            doc.metadata["embedding"] = cached_embedding

                    # Check if document already has embedding
                    if "embedding" not in doc.metadata or not isinstance(
                            doc.metadata["embedding"], (list, np.ndarray)
                    ):
                        if doc.page_content.strip():
                            docs_to_embed.append(doc)

                    batch_doc_ids.append(doc_id)
                    batch_faiss_ids.append(faiss_id)
                    batch_docs.append(doc)

                # Generate missing embeddings
                if docs_to_embed:
                    try:
                        # Get embeddings for docs missing them
                        texts_to_embed = [doc.page_content for doc in docs_to_embed]
                        raw_embeddings = self.embeddings.embed_documents(texts_to_embed)
                        embeddings_generated += len(raw_embeddings)

                        # Update documents with their embeddings
                        for j, (doc, embedding) in enumerate(zip(docs_to_embed, raw_embeddings)):
                            doc.metadata["embedding"] = embedding

                            # Update cache if enabled
                            if enable_cache and hasattr(self, '_embedding_cache') and doc.id:
                                self._embedding_cache.put(doc.id, embedding)
                    except Exception as e:
                        logger.error(f"Error embedding batch: {str(e)}")
                        # Skip documents that couldn't be embedded
                        failed_ids = {doc.id for doc in docs_to_embed}
                        batch_doc_ids = [doc_id for doc_id in batch_doc_ids if doc_id not in failed_ids]
                        batch_faiss_ids = [faiss_id for doc_id, faiss_id in zip(batch_doc_ids, batch_faiss_ids)
                                           if doc_id not in failed_ids]
                        batch_docs = [doc for doc in batch_docs if doc.id not in failed_ids]

                # Collect embeddings for FAISS
                batch_embeddings = []
                valid_indices = []
                valid_docs = []

                for i, doc in enumerate(batch_docs):
                    if "embedding" in doc.metadata and isinstance(doc.metadata["embedding"], (list, np.ndarray)):
                        emb = doc.metadata["embedding"]
                        if isinstance(emb, list):
                            emb = np.array(emb)
                        batch_embeddings.append(emb)
                        valid_indices.append(i)
                        valid_docs.append(doc)

                # Add only valid docs/embeddings to the final lists
                for i in valid_indices:
                    all_doc_ids.append(batch_doc_ids[i])
                    all_faiss_ids.append(batch_faiss_ids[i])
                    all_embeddings.append(batch_embeddings[i - valid_indices[0]])
                    all_content_docs.append(valid_docs[i - valid_indices[0]])

                # Periodically save for very large operations
                if len(all_embeddings) >= save_interval and self.auto_save:
                    self._add_batch_to_index(
                        all_embeddings, all_faiss_ids, all_doc_ids,
                        all_content_docs,
                        namespace
                    )
                    # Reset after saving
                    all_embeddings = []
                    all_faiss_ids = []
                    all_doc_ids = []
                    all_content_docs = []

        # 8. Add final batch to index
        if all_embeddings:
            self._add_batch_to_index(
                all_embeddings, all_faiss_ids, all_doc_ids,
                all_content_docs,
                namespace
            )

        duration = time.time() - start_time
        logger.info(
            f"Upsert completed in {duration:.2f}s: processed {documents_processed} docs, "
            f"generated {embeddings_generated} embeddings"
        )

        return documents_processed, embeddings_generated

    def _chunk_document_with_id(self, document: LangChainDocument) -> List[LangChainDocument]:
        """
        Split a document into chunks while maintaining ID consistency.
        Each chunk gets a unique ID but maintains a reference to the original document ID.
        """
        chunks = self.text_splitter.split_text(document.page_content)
        chunked_docs = []

        # Make sure document has an ID
        source_doc_id = document.id or document.metadata.get("document_id", str(uuid.uuid4()))
        if not document.id:
            document.id = source_doc_id
            document.metadata["document_id"] = source_doc_id

        for i, chunk in enumerate(chunks):
            # Create a unique ID for this chunk that includes the original ID
            chunk_id = f"{source_doc_id}_chunk_{i}"

            # Copy metadata and add chunk info
            metadata = document.metadata.copy() if document.metadata else {}
            metadata["chunk_id"] = chunk_id
            metadata["chunk_index"] = i
            metadata["source_id"] = source_doc_id
            metadata["document_id"] = chunk_id  # Set document_id to the chunk ID
            metadata["total_chunks"] = len(chunks)

            # Create the chunked document with proper IDs
            chunk_doc = LangChainDocument(
                page_content=chunk,
                metadata=metadata
            )
            # Set the document ID properly
            chunk_doc.id = chunk_id

            chunked_docs.append(chunk_doc)

        return chunked_docs

    def _add_batch_to_index(
            self, embeddings, faiss_ids, doc_ids, docs, namespace
    ):
        """
        Helper to add a batch of documents to the FAISS index.
        Ensures ID consistency between documents and the index.
        """
        if not embeddings:
            return

        try:
            # Convert to numpy arrays
            embeddings_np = np.array(embeddings).astype('float32')
            faiss_ids_np = np.array(faiss_ids).astype('int64')

            # Add to FAISS index
            self.indexes[namespace].add_with_ids(embeddings_np, faiss_ids_np)

            # Update document store with ID consistency
            for i, (doc_id, faiss_id, doc) in enumerate(zip(doc_ids, faiss_ids, docs)):
                # Ensure document ID matches what we're storing
                if doc.id != doc_id:
                    logger.warning(f"Document ID mismatch: {doc.id} vs {doc_id}, fixing...")
                    doc.id = doc_id

                # Ensure metadata has the same document_id
                if doc.metadata.get("document_id") != doc_id:
                    doc.metadata["document_id"] = doc_id

                # Store in document store
                self.document_store.setdefault(namespace, {})[doc_id] = {
                    'faiss_id': int(faiss_id),
                    'document': doc
                }

            # Update namespace metadata
            self.namespaces.setdefault(namespace, {})['total_docs'] = len(self.document_store[namespace])

            logger.info(f"Added {len(embeddings)} documents to namespace '{namespace}'")

            # Auto-save if configured
            if self.auto_save:
                self._save_namespace(namespace)

        except Exception as e:
            logger.error(f"Error adding to FAISS index: {str(e)}")
            # Could implement retry logic here

    def delete_documents(self, doc_ids: List[str], namespace: str = ""):
        """
        Delete documents by their IDs.

        :param doc_ids: List of document IDs to delete
        :param namespace: Target namespace
        """
        namespace = namespace or "default"

        if namespace not in self.indexes or namespace not in self.document_store:
            logger.warning(f"Namespace '{namespace}' does not exist, nothing to delete")
            return

        faiss_ids_to_remove = []

        for doc_id in doc_ids:
            if doc_id in self.document_store[namespace]:
                faiss_id = self.document_store[namespace][doc_id]['faiss_id']
                faiss_ids_to_remove.append(faiss_id)
                del self.document_store[namespace][doc_id]

        if faiss_ids_to_remove:
            try:
                # Remove IDs from FAISS index
                self.indexes[namespace].remove_ids(np.array(faiss_ids_to_remove, dtype='int64'))

                # Update namespace metadata
                self.namespaces[namespace]['total_docs'] = len(self.document_store[namespace])

                logger.info(f"Deleted {len(faiss_ids_to_remove)} documents from namespace '{namespace}'")

                # Auto-save if configured
                if self.auto_save:
                    self._save_namespace(namespace)

            except Exception as e:
                logger.error(f"Error removing documents from FAISS index: {str(e)}")

    def delete_all_docs(self, namespace: str = ""):
        """Delete all documents in a namespace."""
        namespace = namespace or "default"

        if namespace not in self.indexes:
            logger.warning(f"Namespace '{namespace}' does not exist, nothing to delete")
            return

        # Create a new empty index
        self._create_new_index(namespace)

        logger.info(f"Deleted all documents from namespace '{namespace}'")

        # Auto-save if configured
        if self.auto_save:
            self._save_namespace(namespace)

    def delete_with_metadata(self, metadata_filter: Dict[str, Any], namespace: str = ""):
        """
        Delete documents that match a metadata filter.

        :param metadata_filter: Metadata filter specification
        :param namespace: Target namespace
        """
        namespace = namespace or "default"

        if namespace not in self.document_store:
            logger.warning(f"Namespace '{namespace}' does not exist, nothing to delete")
            return

        # Find matching documents
        matching_ids = []

        for doc_id, doc_data in self.document_store[namespace].items():
            if self._matches_metadata_filter(doc_data['document'].metadata, metadata_filter):
                matching_ids.append(doc_id)

        if matching_ids:
            self.delete_documents(matching_ids, namespace)
            logger.info(f"Deleted {len(matching_ids)} documents matching filter in namespace '{namespace}'")

    def _matches_metadata_filter(self, metadata: Dict[str, Any], filter_spec: Dict[str, Any]) -> bool:
        """Check if document metadata matches a filter specification."""
        if not metadata or not filter_spec:
            return False

        for key, filter_value in filter_spec.items():
            if key not in metadata:
                return False

            doc_value = metadata[key]

            # Handle complex filter operations
            if isinstance(filter_value, dict):
                # Check for operators
                for op, op_value in filter_value.items():
                    if op == "$eq":
                        if doc_value != op_value:
                            return False
                    elif op == "$ne":
                        if doc_value == op_value:
                            return False
                    elif op == "$gt":
                        if not isinstance(doc_value, (int, float)) or doc_value <= op_value:
                            return False
                    elif op == "$gte":
                        if not isinstance(doc_value, (int, float)) or doc_value < op_value:
                            return False
                    elif op == "$lt":
                        if not isinstance(doc_value, (int, float)) or doc_value >= op_value:
                            return False
                    elif op == "$lte":
                        if not isinstance(doc_value, (int, float)) or doc_value > op_value:
                            return False
                    elif op == "$in":
                        if not isinstance(op_value, list) or doc_value not in op_value:
                            return False
                    elif op == "$nin":
                        if not isinstance(op_value, list) or doc_value in op_value:
                            return False
            else:
                # Simple equality check
                if doc_value != filter_value:
                    return False

        return True

    def query(
            self,
            queries: List[Query],
            namespace: str = "",
            include_embeddings: bool = False,
            fallback_to_metadata: bool = True  # New parameter to enable fallback to metadata query
    ) -> List[List[LangChainDocument]]:
        """
        Run multiple semantic search queries against the vector store with improved filtering.

        :param queries: List of Query objects
        :param namespace: Target namespace
        :param include_embeddings: Whether to include embeddings in returned documents
        :param fallback_to_metadata: If True, fall back to metadata-only search when semantic results don't match filter
        :return: List of lists of retrieved documents, one list per query
        """
        namespace = namespace or "default"

        if namespace not in self.indexes:
            logger.warning(f"Namespace '{namespace}' does not exist, creating empty one")
            self._create_new_index(namespace)
            return [[] for _ in queries]

        # Process each query
        results = []

        for query in queries:
            try:
                # Get embedding for query
                query_embedding = self.embeddings.embed_query(query.text)

                # Determine limit (top_k)
                top_k = query.top_k or 10

                # Increase k to ensure we have enough after filtering
                search_k = top_k
                if query.metadata_filter:
                    # Search much more if we have filters
                    search_k = min(2000, top_k * 10)  # Increased search space

                # Search the index
                D, I = self.indexes[namespace].search(
                    np.array([query_embedding]).astype('float32'),
                    k=search_k
                )

                # Convert results to documents
                docs = []
                seen_ids = set()  # For deduplication

                for score, idx in zip(D[0], I[0]):
                    if idx == -1:  # FAISS returns -1 for empty slots
                        continue

                    # Find document matching this FAISS ID
                    matching_doc = None
                    matching_id = None

                    for doc_id, data in self.document_store[namespace].items():
                        if data['faiss_id'] == idx:
                            matching_doc = data['document']
                            matching_id = doc_id
                            break

                    if not matching_doc:
                        continue

                    # Skip if we've seen this doc already
                    if matching_id in seen_ids:
                        continue

                    # Apply metadata filter if present
                    if query.metadata_filter and not self._matches_metadata_filter(
                            matching_doc.metadata, query.metadata_filter
                    ):
                        continue

                    # Add score to metadata
                    matching_doc.metadata["score"] = float(score)

                    # Remove embedding unless specifically requested
                    if not include_embeddings and "embedding" in matching_doc.metadata:
                        doc_copy = LangChainDocument(
                            page_content=matching_doc.page_content,
                            metadata={k: v for k, v in matching_doc.metadata.items() if k != "embedding"}
                        )
                        doc_copy.id = matching_doc.id
                        docs.append(doc_copy)
                    else:
                        docs.append(matching_doc)

                    seen_ids.add(matching_id)

                    # Stop once we have enough results
                    if len(docs) >= top_k:
                        break

                # IMPORTANT FIX: If we have a metadata filter but no results,
                # fall back to a direct metadata query if fallback is enabled
                if fallback_to_metadata and query.metadata_filter and not docs:
                    logger.info(f"Semantic search with filter {query.metadata_filter} returned no results. "
                                f"Falling back to metadata-only query.")

                    # Get documents by metadata filter
                    metadata_docs = self.query_by_metadata(
                        metadata_filter=query.metadata_filter,
                        namespace=namespace,
                        limit=top_k,
                        include_embeddings=include_embeddings
                    )

                    # Sort metadata results by relevance to query if possible
                    if metadata_docs:
                        # Get page content for sorting
                        texts = [doc.page_content for doc in metadata_docs]

                        try:
                            # Compute similarity to query
                            query_emb = np.array(query_embedding)
                            doc_embeddings = []

                            # If embeddings are already in metadata, use them
                            for doc in metadata_docs:
                                if "embedding" in doc.metadata and isinstance(doc.metadata["embedding"],
                                                                              (list, np.ndarray)):
                                    emb = doc.metadata["embedding"]
                                    if isinstance(emb, list):
                                        emb = np.array(emb)
                                    doc_embeddings.append(emb)
                                else:
                                    # We'll need to compute embeddings - get them in batch
                                    break

                            # If we didn't collect all embeddings, compute them in one batch
                            if len(doc_embeddings) < len(metadata_docs):
                                doc_embeddings = self.embeddings.embed_documents(texts)

                            # Calculate similarities using dot product
                            similarities = []
                            for doc_emb in doc_embeddings:
                                doc_emb_array = np.array(doc_emb)
                                # Normalize vectors before dot product
                                query_norm = query_emb / np.linalg.norm(query_emb)
                                doc_norm = doc_emb_array / np.linalg.norm(doc_emb_array)
                                similarity = np.dot(query_norm, doc_norm)
                                similarities.append(similarity)

                            # Sort by similarity (highest first)
                            sorted_docs = [doc for _, doc in sorted(
                                zip(similarities, metadata_docs),
                                key=lambda pair: pair[0],
                                reverse=True
                            )]

                            # Add scores to metadata
                            for doc, score in zip(sorted_docs, sorted(similarities, reverse=True)):
                                doc.metadata["score"] = float(score)

                            docs = sorted_docs[:top_k]

                        except Exception as e:
                            logger.error(f"Error computing similarities for metadata results: {str(e)}")
                            # Just use metadata docs without sorting
                            docs = metadata_docs[:top_k]
                            for doc in docs:
                                doc.metadata["score"] = 0.0  # Default score

                results.append(docs)

            except Exception as e:
                logger.error(f"Error processing query: {str(e)}")
                results.append([])

        return results

    def query_by_metadata(
            self,
            metadata_filter: Dict[str, Any],
            namespace: str = "",
            limit: int = 1000,
            include_embeddings: bool = False
    ) -> List[LangChainDocument]:
        """
        Retrieve documents by metadata filter without semantic search.

        :param metadata_filter: Metadata filter specification
        :param namespace: Target namespace
        :param limit: Maximum number of documents to return
        :param include_embeddings: Whether to include embeddings in returned documents
        :return: List of matching documents
        """
        namespace = namespace or "default"

        if namespace not in self.document_store:
            logger.warning(f"Namespace '{namespace}' does not exist")
            return []

        # Find matching documents
        matching_docs = []

        for doc_id, doc_data in self.document_store[namespace].items():
            if self._matches_metadata_filter(doc_data['document'].metadata, metadata_filter):
                doc_copy = doc_data['document']

                # Remove embedding unless specifically requested
                if not include_embeddings and "embedding" in doc_copy.metadata:
                    doc_copy_meta = doc_copy.metadata.copy()
                    doc_copy_meta.pop("embedding", None)
                    doc_copy.metadata = doc_copy_meta

                matching_docs.append(doc_copy)

                if len(matching_docs) >= limit:
                    break

        return matching_docs

    def hybrid_search(
            self,
            query_text: str,
            metadata_filter: Optional[Dict[str, Any]] = None,
            namespace: str = "",
            top_k: int = 10,
            alpha: float = 0.5  # Weight between semantic (0) and keyword (1)
    ) -> List[LangChainDocument]:
        """
        Perform hybrid search combining semantic and keyword matching.

        :param query_text: Query text
        :param metadata_filter: Optional metadata filter
        :param namespace: Target namespace
        :param top_k: Number of results to return
        :param alpha: Weight between semantic and keyword (0=semantic only, 1=keyword only)
        :return: List of documents
        """
        namespace = namespace or "default"

        if namespace not in self.document_store:
            logger.warning(f"Namespace '{namespace}' does not exist")
            return []

        # 1. Semantic search
        semantic_query = Query(
            text=query_text,
            top_k=min(100, top_k * 3),  # Get more results for re-ranking
            metadata_filter=metadata_filter
        )
        semantic_results = self.query([semantic_query], namespace=namespace)[0]

        # 2. Get all potentially matching documents by metadata filter
        if metadata_filter:
            keyword_candidates = self.query_by_metadata(
                metadata_filter,
                namespace=namespace,
                limit=1000  # Get a larger set for keyword matching
            )
        else:
            # If no metadata filter, use all documents in the namespace (up to a limit)
            keyword_candidates = list(self.document_store[namespace].values())[:1000]
            keyword_candidates = [data['document'] for data in keyword_candidates]

        # 3. Simple keyword matching on query terms
        query_terms = set(query_text.lower().split())
        keyword_scores = {}

        for doc in keyword_candidates:
            doc_id = doc.id

            # Calculate keyword match score
            doc_text = doc.page_content.lower()
            term_matches = sum(term in doc_text for term in query_terms)
            if len(query_terms) > 0:
                keyword_scores[doc_id] = term_matches / len(query_terms)
            else:
                keyword_scores[doc_id] = 0

        # 4. Combine scores
        combined_scores = {}

        # Add semantic results
        for doc in semantic_results:
            doc_id = doc.id
            semantic_score = 1.0 - (doc.metadata.get("score", 0) / 2.0)  # Convert distance to similarity
            keyword_score = keyword_scores.get(doc_id, 0)
            combined_scores[doc_id] = {
                'doc': doc,
                'score': (1 - alpha) * semantic_score + alpha * keyword_score
            }

        # Add keyword results not already in combined
        for doc in keyword_candidates:
            doc_id = doc.id
            if doc_id not in combined_scores and keyword_scores.get(doc_id, 0) > 0:
                keyword_score = keyword_scores[doc_id]
                combined_scores[doc_id] = {
                    'doc': doc,
                    'score': alpha * keyword_score  # No semantic score
                }

        # 5. Sort and return top results
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return [item['doc'] for item in sorted_results[:top_k]]


def list_indexes(index_folder: str = "faiss_indexes") -> List[str]:
    """
    List all FAISS indexes in the specified folder.
    """
    if not os.path.exists(index_folder):
        return []

    indexes = []
    for file in os.listdir(index_folder):
        if file.endswith(".index"):
            indexes.append(file.replace(".index", ""))
    return indexes