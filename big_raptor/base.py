"""
Optimized Raptor with multiprocessing fixes.
Includes proper initialization and protection for multiprocessing.
"""

import asyncio
import uuid
import os
import time
import multiprocessing
from enum import Enum
from typing import Any, List, Optional, Union, Dict, Tuple, Callable
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
import functools

import numpy as np
import tiktoken
from langchain.schema import Document as LangChainDocument
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Import optimized clustering - assumed to be in 'big_raptor.optimized_clustering'
# We'll need to add proper protection to this module as well
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential

#from big_raptor.raptor_visualization import RaptorVisualizationHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OptimizedRaptor")

# Determine number of available CPUs - do this safely
CPU_COUNT = multiprocessing.cpu_count()
logger.info(f"System has {CPU_COUNT} CPU cores available")

DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following text, including the most important facts, concepts, and relationships:\n\n{context}"
)


# Here's the fixed code for the clustering helper function that properly handles coroutines
async def process_clustering_async(
        docs: List[LangChainDocument],
        embedding_map: Dict[str, np.ndarray],
        max_length_in_cluster: int,
        tokenizer,
        reduction_dimension: int,
        threshold: float,
        max_workers: int,
        process_pool
) -> List[List[LangChainDocument]]:
    """
    Async wrapper for processing clustering that correctly handles the event loop.
    """
    from big_raptor.optimized_clustering import batch_cluster_documents, parallel_cluster_documents

    # For smaller batches, do synchronous processing
    if len(docs) < 500:
        return batch_cluster_documents(
            docs,
            embedding_map,
            max_length_in_cluster,
            tokenizer,
            reduction_dimension,
            threshold
        )
    else:
        # For larger batches, use process pool but safely
        args_tuple = (
            docs,
            embedding_map,
            max_length_in_cluster,
            tokenizer,
            reduction_dimension,
            threshold,
            max_workers
        )

        # Use run_in_executor to run the CPU-bound task without blocking the event loop
        loop = asyncio.get_event_loop()
        from big_raptor.optimized_clustering import _parallel_cluster_documents_helper
        return await loop.run_in_executor(
            process_pool,
            _parallel_cluster_documents_helper,
            args_tuple
        )


# Now fix the methods in OptimizedRaptorRetriever to correctly await coroutines


class QueryModes(str, Enum):
    """Query modes."""
    tree_traversal = "tree_traversal"
    collapsed = "collapsed"
    hybrid = "hybrid"  # New hybrid mode


# Helper functions for multiprocessing need to be at module level
def _cluster_partition_helper(args):
    """
    Helper function for clustering partitions that can be safely pickled for multiprocessing.
    Args is a tuple containing all necessary parameters.
    """
    from big_raptor.optimized_clustering import cluster_partition
    return cluster_partition(*args)


def _parallel_cluster_documents_helper(args):
    """
    Helper function for parallel document clustering that can be safely pickled.
    Args is a tuple with (docs, embedding_map, max_length, tokenizer, dim, threshold, max_workers)
    """
    from big_raptor.optimized_clustering import parallel_cluster_documents
    return parallel_cluster_documents(*args)


class SummaryModule:
    def __init__(
            self,
            llm: Optional[OpenAI] = None,
            summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
            num_workers: Optional[int] = None,  # Will determine based on CPU count
            max_tokens_per_cluster: int = 12000,
            show_progress: bool = True,
    ) -> None:
        """
        A module responsible for generating summaries of documents/clusters.
        Optimized to use all available CPU cores.
        """
        self.llm = llm or ChatOpenAI(temperature=0, request_timeout=60)
        self.summary_prompt = summary_prompt

        # Auto-determine number of workers based on CPU count if not specified
        if num_workers is None:
            # Use 75% of available cores for summary generation
            # This leaves resources for other operations
            self.num_workers = max(1, int(CPU_COUNT * 0.75))
        else:
            self.num_workers = num_workers

        logger.info(f"Summary module initialized with {self.num_workers} workers")

        self.max_tokens_per_cluster = max_tokens_per_cluster
        self.show_progress = show_progress
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.summary_prompt)]
        )
        self.llm_chain = create_stuff_documents_chain(self.llm, prompt_template)

    def _truncate_if_needed(self, docs: List[LangChainDocument]) -> List[LangChainDocument]:
        """Truncate document content if total exceeds max token limit."""
        total_tokens = sum(len(self.tokenizer.encode(doc.page_content)) for doc in docs)

        if total_tokens <= self.max_tokens_per_cluster:
            return docs

        logger.warning(f"Cluster exceeds token limit ({total_tokens} > {self.max_tokens_per_cluster}), truncating...")

        # Calculate how much to keep (as a ratio)
        keep_ratio = self.max_tokens_per_cluster / total_tokens
        truncated_docs = []

        for doc in docs:
            tokens = self.tokenizer.encode(doc.page_content)
            # Keep at least 100 tokens or the calculated ratio, whichever is greater
            keep_tokens = max(100, int(len(tokens) * keep_ratio))
            truncated_content = self.tokenizer.decode(tokens[:keep_tokens])

            # Create new doc with truncated content but copy metadata
            truncated_doc = LangChainDocument(
                page_content=truncated_content,
                metadata=doc.metadata.copy()
            )
            truncated_docs.append(truncated_doc)

        return truncated_docs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry_error_callback=lambda _: None  # Return None on failure
    )
    async def _summarize_with_retry(self, docs: List[LangChainDocument]) -> str:
        """Summarize with retry logic and error handling."""
        try:
            # Safety check - truncate if too many tokens
            safe_docs = self._truncate_if_needed(docs)
            return await self.llm_chain.ainvoke({"context": safe_docs})
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            # Return a minimal summary if all retries fail
            return f"Collection of {len(docs)} documents [summary failed]"

    async def generate_summaries(
            self, documents_per_cluster: List[List[LangChainDocument]]
    ) -> List[str]:
        """
        Generate summaries for each cluster of documents asynchronously.
        Uses available CPU cores efficiently.
        """
        start_time = time.time()

        # Error handling: ensure documents_per_cluster is a list of document lists
        if not isinstance(documents_per_cluster, list):
            logger.error(f"Invalid input to generate_summaries: {type(documents_per_cluster)}")
            return ["Error: Invalid document clusters format"]

        # Check if the list is empty
        if len(documents_per_cluster) == 0:
            logger.warning("No document clusters provided to generate summaries")
            return []

        # Validate that each item is a list of documents
        for i, cluster in enumerate(documents_per_cluster):
            if not isinstance(cluster, list):
                logger.error(f"Cluster at index {i} is not a list: {type(cluster)}")
                documents_per_cluster[i] = [cluster] if isinstance(cluster, LangChainDocument) else []

        # Filter out empty clusters
        documents_per_cluster = [cluster for cluster in documents_per_cluster if cluster]

        if not documents_per_cluster:
            logger.warning("No valid document clusters after filtering")
            return []

        # Now proceed with summarization
        semaphore = asyncio.Semaphore(self.num_workers)

        async def worker(cluster_docs, cluster_idx):
            async with semaphore:
                summary = await self._summarize_with_retry(cluster_docs)
                return (cluster_idx, summary)

        # Process clusters in batches to avoid too many concurrent tasks
        # The batch size is scaled based on available cores
        batch_size = max(10, 5 * self.num_workers)
        all_tasks = []

        # Create tasks for all clusters with their index
        for i, cluster_docs in enumerate(documents_per_cluster):
            all_tasks.append(worker(cluster_docs, i))

        # Execute tasks in batches
        results = []
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i + batch_size]

            if self.show_progress:
                logger.info(f"Processing summary batch {i // batch_size + 1}/{math.ceil(len(all_tasks) / batch_size)}")

            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)

        # Sort results by original cluster index
        results.sort(key=lambda x: x[0])
        summaries = [result[1] for result in results]

        duration = time.time() - start_time
        logger.info(
            f"Generated {len(summaries)} summaries in {duration:.2f} seconds ({duration / max(1, len(summaries)):.2f}s per summary)")
        return summaries


class OptimizedRaptorRetriever:
    """
    RaptorRetriever optimized for large corpora and parallel processing.
    Uses all available CPU cores for maximum performance.
    """

    def __init__(
            self,
            vectorstore_manager,
            tree_depth: int = 3,
            similarity_top_k: int = 5,
            llm: Optional[OpenAI] = None,
            summary_module: Optional[SummaryModule] = None,
            mode: QueryModes = QueryModes.hybrid,
            batch_size: Optional[int] = None,
            cpu_utilization: float = 0.8,
            max_workers: Optional[int] = None,
            verbose: bool = True,
            cache_embeddings: bool = True,
            do_chunking: bool = True,
            auto_delete: bool = False,
            max_length_in_cluster: int = 10000,
            reduction_dimension: int = 5,
            clustering_threshold: float = 0.1,
            visualization_handler: Optional[Callable] = None,  # New parameter
            visualization_output_path: str = "./raptor_visualizations",  # New parameter
            **kwargs: Any,
    ) -> None:
        """
        Initialize the CPU-optimized retriever.

        :param vectorstore_manager: FAISS vector store manager
        :param tree_depth: Number of hierarchical clustering levels
        :param similarity_top_k: K for retrieval at each level
        :param llm: Language model for summary generation
        :param summary_module: Optional custom summary module
        :param mode: Query mode (tree_traversal, collapsed, or hybrid)
        :param cpu_utilization: Fraction of CPU cores to use (0.0-1.0)
        :param max_workers: Maximum number of workers for parallel tasks
        :param batch_size: Batch size for document processing
        :param verbose: Whether to show detailed logging
        :param cache_embeddings: Whether to cache embeddings in memory
        :param do_chunking: Whether to chunk documents during processing
        :param auto_delete: Whether to automatically delete existing documents
        :param max_length_in_cluster: Maximum token length per cluster
        :param reduction_dimension: Dimension for UMAP reduction
        :param clustering_threshold: Threshold for clustering similarity
        :param visualization_handler: Optional function to handle visualizations
        :param visualization_output_path: Path to save visualizations
        """
        # Existing initialization code here...

        # Add the visualization parameters
        self.visualization_handler = visualization_handler
        self.visualization_output_path = visualization_output_path

        # Ensure visualization output directory exists
        if self.visualization_handler:
            os.makedirs(self.visualization_output_path, exist_ok=True)
            # Validate parameters
        if cpu_utilization < 0.1 or cpu_utilization > 1.0:
            logger.warning(f"Invalid cpu_utilization {cpu_utilization}, using default 0.8")
            cpu_utilization = 0.8

        if tree_depth < 1:
            logger.warning(f"Invalid tree_depth {tree_depth}, using default 3")
            tree_depth = 3

        if similarity_top_k < 1:
            logger.warning(f"Invalid similarity_top_k {similarity_top_k}, using default 5")
            similarity_top_k = 5

        if reduction_dimension < 2:
            logger.warning(f"Invalid reduction_dimension {reduction_dimension}, using default 5")
            reduction_dimension = 5

        if clustering_threshold <= 0 or clustering_threshold >= 1.0:
            logger.warning(f"Invalid clustering_threshold {clustering_threshold}, using default 0.1")
            clustering_threshold = 0.1

        self.manager = vectorstore_manager
        self.tree_depth = tree_depth
        self.similarity_top_k = similarity_top_k
        self.mode = mode
        self._verbose = verbose
        self.cache_embeddings = cache_embeddings
        self._embedding_cache = {}

        # Store the newly added parameters
        self.do_chunking = do_chunking
        self.auto_delete = auto_delete
        self.max_length_in_cluster = max_length_in_cluster
        self.reduction_dimension = reduction_dimension
        self.clustering_threshold = clustering_threshold

        # Determine number of workers based on available CPUs
        self.cpu_utilization = min(1.0, max(0.1, cpu_utilization))
        self.max_workers = max_workers or max(1, int(CPU_COUNT * self.cpu_utilization))

        # Calculate optimal batch size based on CPU count if not specified
        # More CPUs = larger batch size is possible
        if batch_size is None:
            self.batch_size = max(500, 250 * self.max_workers)
        else:
            self.batch_size = batch_size

        if self._verbose:
            logger.info(f"Using {self.max_workers} workers with batch size {self.batch_size}")

        # Create process and thread pools - deferred until needed to avoid multiprocessing issues
        self.process_pool = None
        self.thread_pool = None

        # Summary module with CPU-aware worker count
        self.summary_module = summary_module or SummaryModule(
            llm=llm,
            num_workers=max(1, self.max_workers // 2)  # Use half for summaries to avoid API rate limits
        )

        # Use OpenAIEmbeddings by default
        self.embed_model = OpenAIEmbeddings()

        # Create model folder
        self.model_folder = f"models/{self.manager.index_name}"
        os.makedirs(self.model_folder, exist_ok=True)

        if self._verbose:
            logger.info(f"OptimizedRaptor: Using model folder: {self.model_folder}")

    def _setup_process_pools(self):
        """Initialize process and thread pools for parallel execution if not already created."""
        if self.process_pool is None:
            # Use 'spawn' context for better cross-platform compatibility
            # This avoids the multiprocessing bootstrap issue
            ctx = multiprocessing.get_context('spawn')
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=ctx
            )
            logger.info(f"Created process pool with {self.max_workers} workers")

        if self.thread_pool is None:
            # Thread pool for I/O-bound tasks (like API calls)
            self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers * 2)
            logger.info(f"Created thread pool with {self.max_workers * 2} workers")

    def __del__(self):
        """Clean up resources when the object is destroyed."""
        try:
            if self.process_pool:
                self.process_pool.shutdown(wait=False)
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
        except:
            pass

    def wipe_all_docs(self, namespace: str = "") -> None:
        """Wipe all docs from the vector store."""
        if self._verbose:
            logger.info(f"Wiping all docs in namespace '{namespace}'")
        self.manager.delete_all_docs(namespace=namespace)
        self._embedding_cache = {}  # Clear the cache

    def _ensure_embedding(self, doc: LangChainDocument) -> np.ndarray:
        """Get embedding from cache or compute if needed."""
        # Use document ID as cache key
        doc_id = doc.id or doc.metadata.get("document_id", str(uuid.uuid4()))

        if self.cache_embeddings and doc_id in self._embedding_cache:
            return self._embedding_cache[doc_id]

        if "embedding" in doc.metadata and isinstance(doc.metadata["embedding"], list):
            emb = np.array(doc.metadata["embedding"])
        else:
            emb = self.embed_model.embed_query(doc.page_content)
            doc.metadata["embedding"] = emb

        if self.cache_embeddings:
            self._embedding_cache[doc_id] = emb

        return np.array(emb)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
    )
    def _get_documents_by_metadata(
            self, key: str, value: Union[str, int, Dict], namespace: str = ""
    ) -> List[LangChainDocument]:
        """Retrieve docs by metadata filter with retry."""
        metadata_filter = {key: value} if not isinstance(value, dict) else value
        docs = self.manager.query_by_metadata(metadata_filter=metadata_filter, namespace=namespace)

        if not docs and not isinstance(value, dict):  # Only log errors for simple queries
            logger.warning(f"No docs found by metadata filter {key}={value}")

        return docs

    async def _embed_batch_parallel(
            self,
            batch_docs: List[LangChainDocument]
    ) -> Dict[str, np.ndarray]:
        """
        Embed a batch of documents in parallel using multiple threads.
        Returns a mapping of document IDs to embeddings.
        Fixed to properly handle the EmbeddingCache object.
        """
        embedding_map = {}

        # First check cache and existing embeddings
        docs_to_embed = []
        doc_ids_to_embed = []

        for doc in batch_docs:
            doc_id = doc.id or doc.metadata.get("document_id")
            if not doc_id:
                doc_id = str(uuid.uuid4())
                doc.id = doc_id
                doc.metadata["document_id"] = doc_id

            # Already in cache?
            if self.cache_embeddings:
                # Check if we're using the EmbeddingCache object or a simple dict
                if hasattr(self, '_embedding_cache'):
                    if isinstance(self._embedding_cache, dict):
                        # Dictionary-based cache
                        cached_embedding = self._embedding_cache.get(doc_id)
                    elif hasattr(self._embedding_cache, 'get'):
                        # Object with get method (like our EmbeddingCache class)
                        cached_embedding = self._embedding_cache.get(doc_id)
                    else:
                        cached_embedding = None

                    if cached_embedding is not None:
                        embedding_map[doc_id] = cached_embedding
                        continue

            # Already has embedding?
            if "embedding" in doc.metadata and isinstance(doc.metadata["embedding"], (list, np.ndarray)):
                emb = doc.metadata["embedding"]
                if isinstance(emb, list):
                    emb = np.array(emb)
                embedding_map[doc_id] = emb

                # Update cache if we can
                if self.cache_embeddings and hasattr(self, '_embedding_cache'):
                    if isinstance(self._embedding_cache, dict):
                        self._embedding_cache[doc_id] = emb
                    elif hasattr(self._embedding_cache, 'put'):
                        self._embedding_cache.put(doc_id, emb)

                continue

            # Need to compute embedding
            docs_to_embed.append(doc)
            doc_ids_to_embed.append(doc_id)

        if not docs_to_embed:
            return embedding_map

        # Split into smaller sub-batches to avoid rate limits
        # and to optimize parallelism
        sub_batch_size = 20  # Adjust based on API rate limits

        async def embed_sub_batch(sub_docs, sub_ids):
            try:
                contents = [doc.page_content for doc in sub_docs]
                embeddings = self.embed_model.embed_documents(contents)

                # Update documents with embeddings
                for i, (doc, emb) in enumerate(zip(sub_docs, embeddings)):
                    doc.metadata["embedding"] = emb

                # Return mapping of doc_id -> embedding
                return {doc_id: emb for doc_id, emb in zip(sub_ids, embeddings)}
            except Exception as e:
                logger.error(f"Error embedding sub-batch: {str(e)}")
                # Return empty dict on failure
                return {}

        # Create embedding tasks
        tasks = []
        for i in range(0, len(docs_to_embed), sub_batch_size):
            sub_docs = docs_to_embed[i:i + sub_batch_size]
            sub_ids = doc_ids_to_embed[i:i + sub_batch_size]
            tasks.append(embed_sub_batch(sub_docs, sub_ids))

        # Run embedding tasks concurrently
        results = await asyncio.gather(*tasks)

        # Merge results and update cache
        for result_dict in results:
            embedding_map.update(result_dict)

            # Update cache if we can
            if self.cache_embeddings and hasattr(self, '_embedding_cache'):
                for doc_id, emb in result_dict.items():
                    if isinstance(self._embedding_cache, dict):
                        self._embedding_cache[doc_id] = emb
                    elif hasattr(self._embedding_cache, 'put'):
                        self._embedding_cache.put(doc_id, emb)

        return embedding_map

    async def _process_level_batch_parallel(
            self,
            level: int,
            batch_docs: List[LangChainDocument],
            namespace: str,
            do_chunking: bool = None,  # Allow configuring chunking per operation
            auto_delete: bool = None,  # Allow configuring auto-delete per operation
            max_length_in_cluster: int = None  # Allow customizing cluster size
    ) -> List[LangChainDocument]:
        """
        Process a batch of documents at a given level using parallel processing.
        Fixed to handle errors, edge cases, and properly use the embedding cache.
        """

        _do_chunking = self.do_chunking if do_chunking is None else do_chunking
        _auto_delete = self.auto_delete if auto_delete is None else auto_delete
        _max_length_in_cluster = max_length_in_cluster or 10000

        try:
            # Safety check for empty batch
            if not batch_docs:
                logger.warning(f"Empty batch received at level {level}, skipping processing")
                return []

            # 1) Generate embeddings in parallel
            embedding_map = await self._embed_batch_parallel(batch_docs)

            if not embedding_map:
                logger.warning("Failed to generate embeddings for batch, skipping clustering")
                # Treat each document as its own cluster if we can't embed
                clusters = [[doc] for doc in batch_docs]
            elif len(batch_docs) > 10:  # Only cluster if enough documents
                try:
                    # Ensure process pool is initialized
                    self._setup_process_pools()

                    # Use the async wrapper for clustering that correctly handles coroutines
                    clusters = await process_clustering_async(
                        batch_docs,
                        embedding_map,
                        _max_length_in_cluster,  # max_length_in_cluster
                        tiktoken.get_encoding("cl100k_base"),
                        5,  # reduction_dimension
                        0.1,  # threshold
                        self.max_workers,
                        self.process_pool
                    )

                    # Verify clusters is a valid list of document lists
                    if not isinstance(clusters, list):
                        logger.error(f"Clustering returned invalid type: {type(clusters)}")
                        clusters = [[doc] for doc in batch_docs]
                    else:
                        # Ensure each cluster is a list of documents
                        valid_clusters = []
                        for cluster in clusters:
                            if isinstance(cluster, list) and all(isinstance(d, LangChainDocument) for d in cluster):
                                valid_clusters.append(cluster)
                            elif isinstance(cluster, LangChainDocument):
                                valid_clusters.append([cluster])

                        clusters = valid_clusters if valid_clusters else [[doc] for doc in batch_docs]
                except Exception as e:
                    logger.error(f"Error in clustering at level {level}: {str(e)}")
                    # Fall back to treating each doc as its own cluster
                    clusters = [[doc] for doc in batch_docs]
            else:
                # For small batches, just treat as one cluster
                clusters = [batch_docs]

            # Remove any empty clusters
            clusters = [cluster for cluster in clusters if cluster]

            if not clusters:
                logger.warning("No valid clusters created, skipping summarization")
                return []

            # 3) Generate summaries for clusters using our CPU-optimized SummaryModule
            try:
                summaries = await self.summary_module.generate_summaries(clusters)

                # Verify we got the right number of summaries
                if len(summaries) != len(clusters):
                    logger.warning(f"Summary count mismatch: got {len(summaries)}, expected {len(clusters)}")
                    # Pad with generic summaries if needed
                    while len(summaries) < len(clusters):
                        cluster_idx = len(summaries)
                        if cluster_idx < len(clusters):
                            summaries.append(f"Collection of {len(clusters[cluster_idx])} documents")
            except Exception as e:
                logger.error(f"Error generating summaries at level {level}: {str(e)}")
                # Create basic summaries as fallback
                summaries = [f"Collection of {len(cluster)} documents" for cluster in clusters]

            # 4) Create summary docs for next level
            new_nodes = []
            for idx, (summary_text, cluster_docs) in enumerate(zip(summaries, clusters)):
                # Validate summary text
                if not isinstance(summary_text, str) or not summary_text.strip():
                    summary_text = f"Collection of {len(cluster_docs)} documents at level {level}"

                summary_doc = LangChainDocument(
                    page_content=summary_text,
                    metadata={
                        "level": level + 1,
                        "document_id": str(uuid.uuid4()),
                        "child_count": len(cluster_docs),
                        "summary_index": idx
                    }
                )

                # Link child docs with parent_id
                for doc_ in cluster_docs:
                    doc_.metadata["parent_id"] = summary_doc.metadata["document_id"]

                new_nodes.append(summary_doc)

            # 5) Upsert updated docs and new summary nodes
            to_upsert = batch_docs + new_nodes

            try:
                # Batch upsert to reduce API calls
                await self.manager.upsert_documents(
                    to_upsert,
                    namespace=namespace,
                    do_chunking=_do_chunking,
                    auto_delete=_auto_delete,
                    show_progress=False
                )
            except Exception as e:
                logger.error(f"Error upserting documents: {str(e)}")

            return new_nodes

        except Exception as e:
            logger.error(f"Critical error in _process_level_batch_parallel: {str(e)}")
            # Return empty list on critical error
            return []

    # And also ensure we're properly awaiting in the main loop
    async def _process_level_parallel(
            self,
            level: int,
            all_level_docs: List[LangChainDocument],
            namespace: str
    ) -> List[LangChainDocument]:
        """
        Process all documents at a given level in parallel batches.
        Enhanced with safer document counting for debugging.
        """
        total_docs = len(all_level_docs)
        logger.info(f"Processing {total_docs} documents at level {level} using {self.max_workers} workers")

        # Log document metadata analytics in a safer way
        try:
            # Count documents with various metadata fields
            has_doc_id = sum(1 for doc in all_level_docs if doc.metadata.get("document_id"))
            has_parent_id = sum(1 for doc in all_level_docs if doc.metadata.get("parent_id"))
            has_level = sum(1 for doc in all_level_docs if "level" in doc.metadata)
            level_match = sum(1 for doc in all_level_docs if doc.metadata.get("level") == level)

            # Log counts
            logger.info(f"Document metadata stats:")
            logger.info(
                f"- has_document_id: {has_doc_id}/{total_docs} ({(has_doc_id / total_docs) * 100 if total_docs else 0:.1f}%)")
            logger.info(
                f"- has_parent_id: {has_parent_id}/{total_docs} ({(has_parent_id / total_docs) * 100 if total_docs else 0:.1f}%)")
            logger.info(
                f"- has_level: {has_level}/{total_docs} ({(has_level / total_docs) * 100 if total_docs else 0:.1f}%)")
            logger.info(
                f"- correct_level: {level_match}/{total_docs} ({(level_match / total_docs) * 100 if total_docs else 0:.1f}%)")
        except Exception as e:
            logger.error(f"Error analyzing document metadata: {str(e)}")

        # Determine batch size based on document count and available cores
        effective_batch_size = min(
            self.batch_size,
            max(100, total_docs // (self.max_workers * 2))
        )

        batch_count = math.ceil(total_docs / effective_batch_size)
        logger.info(f"Dividing into {batch_count} batches of ~{effective_batch_size} documents each")

        # Process in batches
        all_summary_docs = []

        # Create tasks for each batch
        tasks = []
        for i in range(0, total_docs, effective_batch_size):
            end_idx = min(i + effective_batch_size, total_docs)
            batch_docs = all_level_docs[i:end_idx]
            actual_batch_size = len(batch_docs)

            logger.info(
                f"Created batch {i // effective_batch_size + 1}/{batch_count} with {actual_batch_size} documents")
            # Store the coroutine directly in tasks
            tasks.append((i // effective_batch_size, self._process_level_batch_parallel(level, batch_docs, namespace)))

        # Track batch completion
        completed_batches = 0
        failed_batches = 0
        total_summary_docs = 0
        batch_processing_times = []

        # Process batches with concurrency limited by CPU count
        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(batch_idx, task):
            nonlocal completed_batches, failed_batches, total_summary_docs

            batch_number = batch_idx + 1
            async with semaphore:
                batch_start = time.time()
                try:
                    logger.info(f"Starting batch {batch_number}/{batch_count} at level {level}")
                    # Await the coroutine directly
                    result = await task

                    batch_duration = time.time() - batch_start
                    batch_processing_times.append(batch_duration)

                    if result:
                        summary_count = len(result)
                        total_summary_docs += summary_count
                        logger.info(
                            f"Batch {batch_number}/{batch_count} completed in {batch_duration:.2f}s with {summary_count} summaries")
                    else:
                        logger.warning(
                            f"Batch {batch_number}/{batch_count} completed in {batch_duration:.2f}s but returned no summaries")

                    completed_batches += 1
                    return result
                except Exception as e:
                    batch_duration = time.time() - batch_start
                    logger.error(f"Batch {batch_number}/{batch_count} failed after {batch_duration:.2f}s: {str(e)}")
                    failed_batches += 1
                    return []

        # Create and gather limited concurrent tasks
        concurrent_tasks = [process_with_semaphore(idx, task) for idx, task in tasks]

        # Create a progress tracker
        if batch_count > 10:  # Only show periodic updates for large batch counts
            async def log_progress():
                last_completed = 0
                while completed_batches + failed_batches < batch_count:
                    current_completed = completed_batches
                    new_completed = current_completed - last_completed
                    if new_completed > 0:
                        logger.info(
                            f"Progress update: {current_completed}/{batch_count} batches complete ({(current_completed / batch_count) * 100:.1f}%)")
                        last_completed = current_completed
                    await asyncio.sleep(10)  # Update every 10 seconds

            # Start progress tracker as a background task
            progress_task = asyncio.create_task(log_progress())
        else:
            progress_task = None

        try:
            # Await all concurrent tasks
            batch_results = await asyncio.gather(*concurrent_tasks)
        except Exception as e:
            logger.error(f"Error in gathering batch results: {str(e)}")
            batch_results = []

        # Cancel progress tracker if it exists
        if progress_task:
            progress_task.cancel()

        # Combine results
        for result in batch_results:
            if result:
                all_summary_docs.extend(result)

        # Calculate statistics safely
        try:
            avg_processing_time = sum(batch_processing_times) / len(
                batch_processing_times) if batch_processing_times else 0
            min_processing_time = min(batch_processing_times) if batch_processing_times else 0
            max_processing_time = max(batch_processing_times) if batch_processing_times else 0
        except Exception as e:
            logger.error(f"Error calculating batch statistics: {str(e)}")
            avg_processing_time = min_processing_time = max_processing_time = 0

        # Log detailed metrics
        logger.info(f"Level {level} processing metrics:")
        logger.info(f"- Total batches: {batch_count}")
        logger.info(f"- Completed batches: {completed_batches}")
        logger.info(f"- Failed batches: {failed_batches}")
        logger.info(
            f"- Processing times (sec): avg={avg_processing_time:.2f}, min={min_processing_time:.2f}, max={max_processing_time:.2f}")
        logger.info(f"- Input documents: {total_docs}")
        logger.info(f"- Output summary documents: {len(all_summary_docs)}")

        if len(all_summary_docs) > 0:
            logger.info(f"- Compression ratio: {total_docs / len(all_summary_docs):.2f}:1")
        else:
            logger.info(f"- Compression ratio: N/A (no summaries produced)")

        # Check if number of summaries matches expectations
        expected_summary_count = total_summary_docs
        actual_summary_count = len(all_summary_docs)
        if expected_summary_count != actual_summary_count:
            logger.warning(
                f"Summary count discrepancy - Expected: {expected_summary_count}, Actual: {actual_summary_count}")

        logger.info(f"Level {level} complete: created {len(all_summary_docs)} summary documents for level {level + 1}")

        # Log distribution of summaries by length in a safer way
        if all_summary_docs:
            try:
                # Calculate length statistics
                summary_lengths = [len(doc.page_content) for doc in all_summary_docs]
                avg_length = sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0
                min_length = min(summary_lengths) if summary_lengths else 0
                max_length = max(summary_lengths) if summary_lengths else 0

                # Count by length buckets
                very_short = sum(1 for length in summary_lengths if length < 100)
                short = sum(1 for length in summary_lengths if 100 <= length < 500)
                medium = sum(1 for length in summary_lengths if 500 <= length < 1000)
                long = sum(1 for length in summary_lengths if 1000 <= length < 2000)
                very_long = sum(1 for length in summary_lengths if length >= 2000)

                # Log results
                logger.info(f"Summary length distribution:")
                logger.info(f"- Character length: avg={avg_length:.1f}, min={min_length}, max={max_length}")
                total = len(summary_lengths)
                logger.info(f"- very_short (<100 chars): {very_short}/{total} ({very_short / total * 100:.1f}%)")
                logger.info(f"- short (100-500 chars): {short}/{total} ({short / total * 100:.1f}%)")
                logger.info(f"- medium (500-1000 chars): {medium}/{total} ({medium / total * 100:.1f}%)")
                logger.info(f"- long (1000-2000 chars): {long}/{total} ({long / total * 100:.1f}%)")
                logger.info(f"- very_long (>2000 chars): {very_long}/{total} ({very_long / total * 100:.1f}%)")
            except Exception as e:
                logger.error(f"Error analyzing summary lengths: {str(e)}")

        return all_summary_docs

    def _log_metadata_distribution(self, documents: List[LangChainDocument], level: int) -> None:
        """
        Log distribution of key metadata fields to help with debugging.

        :param documents: List of documents to analyze
        :param level: Current processing level
        """
        if not documents:
            return

        # Track counts of various metadata fields
        metadata_counts = {
            "has_document_id": 0,
            "has_parent_id": 0,
            "has_embedding": 0,
            "has_child_count": 0,
            "has_chunk_id": 0,
            "level_mismatch": 0
        }

        # Track unique values
        unique_levels = set()

        for doc in documents:
            if not doc.metadata:
                continue

            # Check key fields
            if "document_id" in doc.metadata:
                metadata_counts["has_document_id"] += 1

            if "parent_id" in doc.metadata:
                metadata_counts["has_parent_id"] += 1

            if "embedding" in doc.metadata:
                metadata_counts["has_embedding"] += 1

            if "child_count" in doc.metadata:
                metadata_counts["has_child_count"] += 1

            if "chunk_id" in doc.metadata:
                metadata_counts["has_chunk_id"] += 1

            # Check level consistency
            doc_level = doc.metadata.get("level", -1)
            unique_levels.add(doc_level)

            if doc_level != level:
                metadata_counts["level_mismatch"] += 1

        # Log the results
        total_docs = len(documents)
        logger.info(f"Metadata distribution for {total_docs} documents at level {level}:")

        for field, count in metadata_counts.items():
            percentage = (count / total_docs) * 100 if total_docs > 0 else 0
            logger.info(f"- {field}: {count} ({percentage:.1f}%)")

        logger.info(f"- Unique levels found: {sorted(list(unique_levels))}")

    def _log_summary_length_distribution(self, summary_docs: List[LangChainDocument]) -> None:
        """
        Log distribution of summary lengths to help with debugging.

        :param summary_docs: List of summary documents to analyze
        """
        if not summary_docs:
            return

        # Calculate summary lengths
        summary_lengths = [len(doc.page_content) for doc in summary_docs]
        token_lengths = []

        # If tokenizer is available, calculate token lengths too
        if hasattr(self, 'summary_module') and hasattr(self.summary_module, 'tokenizer'):
            token_lengths = [len(self.summary_module.tokenizer.encode(doc.page_content)) for doc in summary_docs]

        # Calculate statistics
        avg_length = sum(summary_lengths) / len(summary_lengths)
        min_length = min(summary_lengths)
        max_length = max(summary_lengths)

        # Define length buckets
        buckets = {
            "very_short (< 100 chars)": 0,
            "short (100-500 chars)": 0,
            "medium (500-1000 chars)": 0,
            "long (1000-2000 chars)": 0,
            "very_long (> 2000 chars)": 0
        }

        # Count summaries in each bucket
        for length in summary_lengths:
            if length < 100:
                buckets["very_short (< 100 chars)"] += 1
            elif length < 500:
                buckets["short (100-500 chars)"] += 1
            elif length < 1000:
                buckets["medium (500-1000 chars)"] += 1
            elif length < 2000:
                buckets["long (1000-2000 chars)"] += 1
            else:
                buckets["very_long (> 2000 chars)"] += 1

        # Log the results
        logger.info(f"Summary length distribution for {len(summary_docs)} documents:")
        logger.info(f"- Character length: avg={avg_length:.1f}, min={min_length}, max={max_length}")

        if token_lengths:
            avg_tokens = sum(token_lengths) / len(token_lengths)
            min_tokens = min(token_lengths)
            max_tokens = max(token_lengths)
            logger.info(f"- Token length: avg={avg_tokens:.1f}, min={min_tokens}, max={max_tokens}")

        # Log distribution by bucket
        for bucket_name, count in buckets.items():
            percentage = (count / len(summary_docs)) * 100
            logger.info(f"- {bucket_name}: {count} ({percentage:.1f}%)")

    async def insert(
            self,
            documents: List[LangChainDocument],
            namespace: str = "",
            fresh_start: bool = False,
    ) -> None:
        """
        Optimized insert using all available CPU cores.
        Enhanced with safer document counting for debugging.

        :param documents: List of documents to insert
        :param namespace: Namespace to store documents in
        :param fresh_start: If True, wipe all existing data first
        """
        start_time = time.time()
        total_docs = len(documents)
        namespace = namespace or "default"

        if self._verbose:
            logger.info(
                f"Inserting {total_docs} documents into namespace '{namespace}' using {self.max_workers} workers")

        # Print initial document count
        initial_count = self._get_document_count(namespace)
        logger.info(f"Initial document count: {initial_count}")

        if fresh_start:
            logger.info(f"Fresh start: wiping all docs from namespace={namespace}")
            self.wipe_all_docs(namespace)
            after_wipe_count = self._get_document_count(namespace)
            logger.info(f"After wipe document count: {after_wipe_count}")
        else:
            # Only delete higher level documents (keep level 0)
            try:
                level_filter = {"level": {"$gte": 1}}
                self.manager.delete_with_metadata(level_filter, namespace=namespace)
                after_level_delete_count = self._get_document_count(namespace)
                logger.info(f"After higher level deletion document count: {after_level_delete_count}")
            except Exception as e:
                logger.error(f"Error deleting higher level documents: {str(e)}")
                # Continue processing if this fails

        # Ensure all documents have level=0 in metadata
        for doc in documents:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["level"] = 0
            if not doc.id:
                doc.id = str(uuid.uuid4())

        # Insert level 0 documents in optimal batch sizes
        effective_batch_size = min(
            self.batch_size,
            max(200, total_docs // (self.max_workers * 2))
        )

        logger.info(f"Using batch size of {effective_batch_size} for level 0 document insertion")

        # Track level 0 insertion progress
        level0_inserted = 0

        for i in range(0, len(documents), effective_batch_size):
            batch = documents[i:i + effective_batch_size]
            batch_size = len(batch)

            if self._verbose and i % (effective_batch_size * 5) == 0:
                logger.info(
                    f"Upserting batch {i // effective_batch_size + 1} of {math.ceil(total_docs / effective_batch_size)}")

            await self.manager.upsert_documents(
                batch,
                namespace=namespace,
                do_chunking=False,  # We handle chunking separately
                auto_delete=False,
                show_progress=False
            )

            level0_inserted += batch_size

            # Log document count after each batch
            if self._verbose:
                batch_count = self._get_document_count(namespace)
                logger.info(f"After batch {i // effective_batch_size + 1} document count: {batch_count}")

        # Log document count after level 0 insertion is complete
        level0_complete_count = self._get_document_count(namespace)
        logger.info(f"Level 0 insertion complete. Document count: {level0_complete_count}")
        logger.info(f"Inserted {level0_inserted} level 0 documents")

        # Process each level in parallel
        for level in range(self.tree_depth):
            level_start = time.time()

            if self._verbose:
                logger.info(f"Processing hierarchical level {level}")

            # Get all docs at current level - this is a potential bottleneck
            # We'll optimize by using multiple queries if needed
            all_level_docs = self._get_documents_by_metadata("level", level, namespace=namespace)
            total_level_docs = len(all_level_docs)

            if not all_level_docs:
                logger.info(f"No documents found at level {level}, stopping hierarchy building")
                break

            logger.info(f"Found {total_level_docs} documents at level {level}")

            # Log document count before processing this level
            before_level_count = self._get_document_count(namespace)
            logger.info(f"Before processing level {level} document count: {before_level_count}")

            # Process this level using parallel execution
            summary_docs = await self._process_level_parallel(level, all_level_docs, namespace)

            # Log document count after processing this level
            after_level_count = self._get_document_count(namespace)

            # Try to get level distribution
            try:
                level_distribution = self._get_level_distribution(namespace)
                level_dist_str = ", ".join([f"L{lvl}: {count}" for lvl, count in level_distribution.items()])
                logger.info(f"Document distribution by level: {level_dist_str}")
            except Exception as e:
                logger.error(f"Error getting level distribution: {str(e)}")

            level_duration = time.time() - level_start
            logger.info(f"Level {level} processed in {level_duration:.2f}s, created {len(summary_docs)} summary docs")
            logger.info(f"After processing level {level} document count: {after_level_count}")
            logger.info(f"Document count change: {after_level_count - before_level_count}")

        # Final document count
        final_count = self._get_document_count(namespace)

        total_duration = time.time() - start_time
        logger.info(f"Completed hierarchical processing of {total_docs} documents in {total_duration:.2f} seconds")
        logger.info(f"Performance: {total_docs / total_duration:.2f} documents per second")
        logger.info(f"Final document count: {final_count}")
        logger.info(f"Net document growth: {final_count - initial_count}")

        if self.visualization_handler:
            try:
                if self._verbose:
                    logger.info("Generating visualization of document hierarchy...")
                visualization_path = self.generate_visualization(namespace)
                if visualization_path:
                    logger.info(f"Visualization saved to: {visualization_path}")
            except Exception as e:
                logger.error(f"Error in visualization generation: {str(e)}")

    def _get_index_stats(self, namespace: str) -> Dict[str, Any]:
        """
        Get detailed statistics about the index for debugging.
        Adapted to work with VectorStoreManager instead of direct FAISS access.

        :param namespace: Target namespace
        :return: Dictionary with index statistics
        """
        namespace = namespace or "default"
        stats = {}

        try:
            # Get stats through the manager instead of direct access
            if hasattr(self.manager, 'indexes'):
                # If manager has direct indexes attribute
                if namespace in self.manager.indexes:
                    stats["faiss_ntotal"] = self.manager.indexes[namespace].ntotal
                else:
                    stats["faiss_ntotal"] = 0
            else:
                # Otherwise get document count from manager
                stats["faiss_ntotal"] = self.manager.namespaces.get(namespace, {}).get('total_docs', 0)

            # Document store stats
            if hasattr(self.manager, 'document_store'):
                if namespace in self.manager.document_store:
                    stats["doc_store_count"] = len(self.manager.document_store[namespace])
                else:
                    stats["doc_store_count"] = 0
            else:
                stats["doc_store_count"] = 0

            # Namespace metadata
            if hasattr(self.manager, 'namespaces'):
                if namespace in self.manager.namespaces:
                    stats["namespace_total_docs"] = self.manager.namespaces[namespace].get('total_docs', 0)
                else:
                    stats["namespace_total_docs"] = 0
            else:
                stats["namespace_total_docs"] = 0

            # Memory usage estimate for embedding cache
            if hasattr(self, '_embedding_cache'):
                stats["embedding_cache_count"] = len(self._embedding_cache)
                # Rough estimate of memory usage (1536-dim embeddings * 4 bytes per float)
                stats["embedding_cache_memory_mb"] = round(len(self._embedding_cache) * 1536 * 4 / (1024 * 1024), 2)

        except Exception as e:
            logger.error(f"Error getting index stats: {str(e)}")
            stats["error"] = str(e)

        return stats

    def _get_level_distribution(self, namespace: str) -> Dict[int, int]:
        """
        Get the distribution of documents by level.
        Adapted to work with VectorStoreManager.

        :param namespace: Target namespace
        :return: Dictionary with level -> count mapping
        """
        namespace = namespace or "default"
        level_counts = {}

        try:
            # Get documents by level through manager
            for level in range(self.tree_depth + 1):
                # Query the manager for documents at this level
                docs = self.manager.query_by_metadata({"level": level}, namespace=namespace)
                if docs:
                    level_counts[level] = len(docs)
                else:
                    level_counts[level] = 0
        except Exception as e:
            logger.error(f"Error getting level distribution: {str(e)}")
            level_counts["error"] = str(e)

        return level_counts

    def _get_document_count(self, namespace: str) -> int:
        """
        Get total document count in the index.
        A simpler alternative to _get_index_stats that's less likely to fail.

        :param namespace: Target namespace
        :return: Number of documents
        """
        namespace = namespace or "default"
        try:
            # Try several methods to get document count
            if hasattr(self.manager, 'namespaces'):
                # First try namespace metadata
                if namespace in self.manager.namespaces:
                    return self.manager.namespaces[namespace].get('total_docs', 0)

            # If that fails, try document store
            if hasattr(self.manager, 'document_store'):
                if namespace in self.manager.document_store:
                    return len(self.manager.document_store[namespace])

            # Last resort - query documents
            try:
                docs = self.manager.query_by_metadata({}, namespace=namespace, limit=1)
                if hasattr(docs, 'total_docs'):
                    return docs.total_docs
            except:
                pass

            # If all else fails, return 0
            return 0
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0

    def _retrieve_collapsed(
            self,
            query,
            namespace: str = "",
            ignore_hierarchy: bool = False
    ) -> List[List[LangChainDocument]]:
        """
        Flat retrieval across all documents with optional metadata filtering.

        :param query: Query object with text, top_k, and metadata_filter
        :param namespace: Target namespace
        :param ignore_hierarchy: If True, retrieves from all levels, not just level 0
        :return: List of lists of retrieved documents
        """
        try:
            # Configure base metadata filter
            base_filter = query.metadata_filter.copy() if query.metadata_filter else {}

            # Enforce level=0 unless ignoring hierarchy
            if not ignore_hierarchy:
                base_filter["level"] = 0

            # Create final query
            top_k = query.top_k or self.similarity_top_k
            final_query = type(query)(
                text=query.text,
                top_k=top_k,
                metadata_filter=base_filter
            )

            # Do the retrieval
            results = self.manager.query([final_query], namespace=namespace)

            if self._verbose:
                doc_count = sum(len(group) for group in results if group)
                logger.info(f"Collapsed retrieval returned {doc_count} documents")

            return results

        except Exception as e:
            logger.error(f"Error in collapsed retrieval: {str(e)}")
            return [[]]  # Return empty result on error

    def _retrieve_tree_traversal(self, query, namespace: str = "") -> List[List[LangChainDocument]]:
        """
        Hierarchical top-down retrieval starting from the top level and following parent-child links.

        :param query: Query object with text, top_k, and metadata_filter
        :param namespace: Target namespace
        :return: List of lists of retrieved documents
        """
        try:
            level = self.tree_depth
            parent_ids = None
            nodes = []
            base_filter = query.metadata_filter.copy() if query.metadata_filter else {}
            top_k = query.top_k or self.similarity_top_k

            # Start from top level and work down
            while level >= 0:
                if parent_ids is None or len(parent_ids) == 0:
                    # Get top level documents
                    combined_filter = {**base_filter, "level": level}
                    top_query = type(query)(
                        text=query.text,
                        top_k=top_k,
                        metadata_filter=combined_filter
                    )

                    try:
                        top_nodes = self.manager.query([top_query], namespace=namespace)
                    except Exception as e:
                        logger.error(f"Error querying top level nodes: {str(e)}")
                        top_nodes = [[]]

                    if self._verbose:
                        logger.info(
                            f"[tree_traversal] Found {len(top_nodes[0]) if top_nodes and top_nodes[0] else 0} docs at level {level}")

                    nodes = top_nodes
                    # Collect document_ids for the next level
                    parent_ids = []
                    if top_nodes and len(top_nodes) > 0 and len(top_nodes[0]) > 0:
                        for doc in top_nodes[0]:
                            doc_id = doc.metadata.get("document_id")
                            if doc_id:
                                parent_ids.append(doc_id)
                else:
                    # Find children of selected documents
                    child_nodes = []
                    for pid in parent_ids:
                        if pid:
                            combined_filter = {**base_filter, "parent_id": pid}
                            child_query = type(query)(
                                text=query.text,
                                top_k=top_k,
                                metadata_filter=combined_filter
                            )

                            try:
                                found_children = self.manager.query([child_query], namespace=namespace)
                                child_nodes.extend(found_children)
                            except Exception as e:
                                logger.error(f"Error querying children for parent {pid}: {str(e)}")

                    if self._verbose and child_nodes:
                        total_children = sum(len(group) for group in child_nodes if group)
                        logger.info(f"[tree_traversal] Found {total_children} child docs at level {level}")

                    nodes = child_nodes
                    # Get parent_ids for next level
                    parent_ids = []
                    for node_list in child_nodes:
                        for doc in node_list:
                            parent_id = doc.metadata.get("document_id")
                            if parent_id:
                                parent_ids.append(parent_id)

                level -= 1

            return nodes

        except Exception as e:
            logger.error(f"Error in tree traversal retrieval: {str(e)}")
            return [[]]  # Return empty result on error

    def _retrieve_hybrid(self, query, namespace: str = "") -> List[List[LangChainDocument]]:
        """
        Hybrid retrieval combining results from tree traversal and direct retrieval.
        This improves recall while preserving the hierarchical structure.

        :param query: Query object with text, top_k, and metadata_filter
        :param namespace: Target namespace
        :return: List of lists of retrieved documents
        """
        try:
            # 1. Get hierarchical results (may be slow but structured)
            tree_results = self._retrieve_tree_traversal(query, namespace=namespace)

            # 2. Get direct results from level 0 (faster, may find different matches)
            direct_query = type(query)(
                text=query.text,
                top_k=max(3, query.top_k // 2) if query.top_k else 3,  # Reduced top_k for direct
                metadata_filter={"level": 0}
            )

            try:
                direct_results = self.manager.query([direct_query], namespace=namespace)
            except Exception as e:
                logger.error(f"Error in direct retrieval for hybrid mode: {str(e)}")
                direct_results = [[]]

            # 3. Combine results, avoiding duplicates
            combined = []
            seen_ids = set()

            # Add tree results first (they follow the hierarchy)
            for result_list in tree_results:
                filtered_docs = []
                for doc in result_list:
                    # Get a unique identifier for the document
                    doc_id = doc.id
                    if not doc_id and doc.metadata:
                        doc_id = doc.metadata.get("document_id")

                    if not doc_id:
                        # Create a content hash if no ID exists
                        doc_id = f"content_{hash(doc.page_content)}"

                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        filtered_docs.append(doc)

                if filtered_docs:
                    combined.append(filtered_docs)

            # Add direct results, deduplicating
            for result_list in direct_results:
                filtered_docs = []
                for doc in result_list:
                    # Get a unique identifier for the document
                    doc_id = doc.id
                    if not doc_id and doc.metadata:
                        doc_id = doc.metadata.get("document_id")

                    if not doc_id:
                        # Create a content hash if no ID exists
                        doc_id = f"content_{hash(doc.page_content)}"

                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        filtered_docs.append(doc)

                if filtered_docs:
                    combined.append(filtered_docs)

            # Log results
            if self._verbose:
                total_docs = sum(len(group) for group in combined)
                logger.info(f"Hybrid retrieval returned {total_docs} documents from {len(combined)} groups")

            return combined

        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {str(e)}")
            return [[]]  # Return empty result on error

    def retrieve(
            self,
            query,
            mode: Optional[QueryModes] = None,
            namespace: str = "default",
            ignore_hierarchy: bool = False
    ) -> List[List[LangChainDocument]]:
        """
        Retrieve documents using specified mode.

        :param query: Query object with text and optional parameters
        :param mode: Retrieval mode (tree_traversal, collapsed, or hybrid)
        :param namespace: Target namespace
        :param ignore_hierarchy: For collapsed mode, retrieve from all levels
        :return: List of lists of retrieved documents
        """
        retrieval_start = time.time()
        mode = mode or self.mode

        try:
            if mode == QueryModes.tree_traversal:
                results = self._retrieve_tree_traversal(query, namespace=namespace)
            elif mode == QueryModes.collapsed:
                results = self._retrieve_collapsed(query, namespace=namespace, ignore_hierarchy=ignore_hierarchy)
            elif mode == QueryModes.hybrid:
                results = self._retrieve_hybrid(query, namespace=namespace)
            else:
                logger.error(f"Invalid retrieval mode: {mode}")
                results = [[]]

            # Ensure we have a valid result format
            if not isinstance(results, list):
                logger.error(f"Invalid retrieval result type: {type(results)}")
                results = [[]]

            # Ensure each inner item is a list
            for i in range(len(results)):
                if not isinstance(results[i], list):
                    logger.warning(f"Result item {i} is not a list, converting: {type(results[i])}")
                    if results[i] is None:
                        results[i] = []
                    else:
                        results[i] = [results[i]]
        except Exception as e:
            logger.error(f"Critical error in retrieve: {str(e)}")
            results = [[]]

        retrieval_time = time.time() - retrieval_start
        if self._verbose:
            total_docs = sum(len(group) for group in results if group)
            logger.info(f"Retrieved {total_docs} documents in {retrieval_time:.2f}s using {mode} mode")

        return results

    def persist(self) -> None:
        """Ensure FAISS indexes are saved to disk."""
        try:
            if self._verbose:
                logger.info("Persisting FAISS indexes to disk")
            self.manager.persist()
        except Exception as e:
            logger.error(f"Error persisting indexes: {str(e)}")

    def clear_caches(self) -> None:
        """Clear in-memory caches to free up RAM."""
        try:
            cache_size = len(self._embedding_cache)
            self._embedding_cache = {}
            if self._verbose:
                logger.info(f"Cleared embedding cache ({cache_size} entries)")
        except Exception as e:
            logger.error(f"Error clearing caches: {str(e)}")

    # Add the generate_visualization method to the class
    def generate_visualization(self, namespace: str = "") -> Optional[str]:
        """
        Generate a visualization of the document hierarchy.

        Args:
            namespace: Target namespace

        Returns:
            Path to the visualization file if created, None otherwise
        """
        try:
            if self.visualization_handler is None:
                logger.info("No visualization handler configured")
                return None

            if self._verbose:
                logger.info(f"Generating visualization for namespace '{namespace}'")

            # Generate hierarchy data
            hierarchy_data = RaptorVisualizationHandler.generate_hierarchy_data(self, namespace)

            # Call the visualization handler with the data
            result = self.visualization_handler(
                hierarchy_data=hierarchy_data,
                output_path=self.visualization_output_path,
                namespace=namespace
            )

            if self._verbose:
                logger.info(f"Visualization generated: {result}")

            return result
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return None
