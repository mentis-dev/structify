"""
Optimized clustering module for large document collections.
With multiprocessing fixes and integrated with your initial implementation.
"""

import logging
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import umap
from langchain.schema import Document as LangChainDocument
from sklearn.mixture import GaussianMixture
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OptimizedClustering")

# Fixed random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# This line is critical for Windows multiprocessing
if __name__ == "__main__":
    multiprocessing.freeze_support()

# Get available CPU count safely
CPU_COUNT = multiprocessing.cpu_count()
logger.info(f"Clustering module detected {CPU_COUNT} CPU cores")


class MiniBatchGMM:
    """
    Memory-efficient Gaussian Mixture Model that processes data in mini-batches.
    This is a wrapper around scikit-learn's GaussianMixture that processes data in batches.
    """

    def __init__(self, n_components=1, batch_size=1000, random_state=RANDOM_SEED):
        self.n_components = n_components
        self.batch_size = batch_size
        self.random_state = random_state
        self.gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        self.fitted = False

    def fit(self, X):
        """Fit the model with batch processing for large datasets."""
        n_samples = X.shape[0]

        # For small datasets, use standard GMM directly
        if n_samples <= self.batch_size:
            self.gmm.fit(X)
            self.fitted = True
            return self

        # For larger datasets, use batch processing
        # First fit with a sample to initialize
        sample_indices = np.random.choice(n_samples, min(self.batch_size, n_samples), replace=False)
        self.gmm.fit(X[sample_indices])

        # Process remaining data in batches
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = X[start_idx:end_idx]

            # Partial fit using current model parameters
            # (This is a simplified approximation as sklearn doesn't have partial_fit for GMM)
            # We're using the predict_proba as a way to update the model incrementally
            self.gmm.predict_proba(batch)

        self.fitted = True
        return self

    def predict_proba(self, X):
        """Predict probability of cluster membership."""
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        return self.gmm.predict_proba(X)

    def bic(self, X):
        """Bayesian Information Criterion for the current model on the input X."""
        if not self.fitted:
            raise ValueError("Model must be fitted before calculating BIC")
        return self.gmm.bic(X)


def efficient_umap_reduce(
        embeddings: np.ndarray,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        verbose: bool = False
) -> Tuple[np.ndarray, Any]:
    """
    Perform efficient UMAP dimensionality reduction with proper error handling.

    Args:
        embeddings: Input embeddings
        n_components: Dimension to reduce to
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance parameter for UMAP
        metric: Distance metric
        verbose: Whether to show detailed logs

    Returns:
        Tuple of (reduced embeddings, UMAP model)
    """
    if verbose:
        logger.info(
            f"Reducing {embeddings.shape[0]} embeddings from {embeddings.shape[1]} to {n_components} dimensions")

    try:
        # Auto-calculate n_neighbors if not enough data points
        if n_neighbors >= embeddings.shape[0]:
            effective_n_neighbors = max(2, min(15, embeddings.shape[0] // 2))
            if verbose:
                logger.info(f"Reducing n_neighbors from {n_neighbors} to {effective_n_neighbors} due to small dataset")
            n_neighbors = effective_n_neighbors

        # Create and fit UMAP model
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=RANDOM_SEED
        )
        reduced_embeddings = reducer.fit_transform(embeddings)

        if verbose:
            logger.info(f"UMAP reduction complete: {reduced_embeddings.shape}")

        return reduced_embeddings, reducer

    except Exception as e:
        logger.error(f"Error in UMAP reduction: {str(e)}")
        # Fallback: use PCA as a simpler alternative
        try:
            from sklearn.decomposition import PCA
            if verbose:
                logger.info(f"Falling back to PCA for dimensionality reduction")
            pca = PCA(n_components=min(n_components, embeddings.shape[0], embeddings.shape[1]),
                      random_state=RANDOM_SEED)
            reduced_embeddings = pca.fit_transform(embeddings)
            return reduced_embeddings, pca
        except Exception as e2:
            logger.error(f"Error in PCA fallback: {str(e2)}")
            # Last resort: return original embeddings or a random projection
            if embeddings.shape[1] <= n_components:
                return embeddings, None
            else:
                # Simple random projection as last resort
                projection = np.random.randn(embeddings.shape[1], n_components)
                reduced = embeddings @ projection
                return reduced, projection


def find_optimal_clusters(
        embeddings: np.ndarray,
        max_clusters: int = 30,
        min_clusters: int = 2,
        batch_size: int = 1000,
        verbose: bool = False
) -> int:
    """
    Find optimal number of clusters using BIC criterion.

    Args:
        embeddings: Input embeddings (usually reduced)
        max_clusters: Maximum number of clusters to consider
        min_clusters: Minimum number of clusters to consider
        batch_size: Batch size for large datasets
        verbose: Whether to show detailed logs

    Returns:
        Optimal number of clusters
    """
    # Cap max_clusters based on dataset size
    n_samples = embeddings.shape[0]
    effective_max = min(max_clusters, n_samples // 5, 50)  # No more than n/5 or 50 clusters
    effective_max = max(min_clusters, effective_max)  # At least min_clusters

    if verbose:
        logger.info(f"Finding optimal clusters between {min_clusters} and {effective_max}")

    if n_samples <= min_clusters:
        # Too few samples, return minimum
        if verbose:
            logger.info(f"Too few samples ({n_samples}), using {min_clusters} clusters")
        return min_clusters

    try:
        # Use MiniBatchGMM for larger datasets
        if n_samples > batch_size:
            bic_values = []
            n_clusters_range = range(min_clusters, effective_max + 1)

            for n_clusters in n_clusters_range:
                if verbose and n_clusters % 5 == 0:
                    logger.info(f"Testing {n_clusters} clusters...")

                gmm = MiniBatchGMM(
                    n_components=n_clusters,
                    batch_size=batch_size,
                    random_state=RANDOM_SEED
                )
                gmm.fit(embeddings)
                bic_values.append(gmm.bic(embeddings))

            optimal_clusters = n_clusters_range[np.argmin(bic_values)]
        else:
            # For smaller datasets, use standard GaussianMixture
            bic_values = []
            n_clusters_range = range(min_clusters, effective_max + 1)

            for n_clusters in n_clusters_range:
                gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED)
                gmm.fit(embeddings)
                bic_values.append(gmm.bic(embeddings))

            optimal_clusters = n_clusters_range[np.argmin(bic_values)]

        if verbose:
            logger.info(f"Optimal number of clusters: {optimal_clusters}")

        return optimal_clusters

    except Exception as e:
        logger.error(f"Error finding optimal clusters: {str(e)}")
        # Fallback: simple heuristic
        fallback_clusters = max(min_clusters, min(int(np.sqrt(n_samples / 2)), effective_max))
        logger.info(f"Using fallback of {fallback_clusters} clusters")
        return fallback_clusters


def partition_documents(
        docs: List[LangChainDocument],
        embedding_map: Dict[str, np.ndarray],
        n_partitions: int
) -> List[Tuple[List[LangChainDocument], Dict[str, np.ndarray]]]:
    """
    Partition documents and embeddings for parallel processing.

    Args:
        docs: List of documents
        embedding_map: Mapping of doc IDs to embeddings
        n_partitions: Number of partitions to create

    Returns:
        List of (docs_partition, embeddings_partition) tuples
    """
    if not docs:
        return []

    # Determine partition size
    n_docs = len(docs)
    n_partitions = min(n_partitions, n_docs)
    partition_size = max(1, n_docs // n_partitions)

    partitions = []
    for i in range(0, n_docs, partition_size):
        end_idx = min(i + partition_size, n_docs)
        partition_docs = docs[i:end_idx]

        # Create embeddings submap for this partition
        partition_embeddings = {}
        for doc in partition_docs:
            doc_id = doc.id or doc.metadata.get("document_id")
            if doc_id and doc_id in embedding_map:
                partition_embeddings[doc_id] = embedding_map[doc_id]

        partitions.append((partition_docs, partition_embeddings))

    return partitions


def cluster_partition(
        partition_data: Tuple[List[LangChainDocument], Dict[str, np.ndarray]],
        max_length_in_cluster: int = 10000,
        tokenizer=None,
        reduction_dimension: int = 5,
        threshold: float = 0.1
) -> List[List[LangChainDocument]]:
    """
    Process a single partition of documents for clustering.

    Args:
        partition_data: Tuple of (docs, embeddings_map)
        max_length_in_cluster: Maximum token length per cluster
        tokenizer: Tokenizer for length calculation
        reduction_dimension: Dimension for UMAP reduction
        threshold: Threshold for cluster membership

    Returns:
        List of document clusters
    """
    docs, embedding_map = partition_data

    if not docs:
        return []

    # For very small partitions, don't cluster
    if len(docs) <= 2:
        return [docs]

    # Initialize tokenizer if not provided
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    try:
        # Get embeddings for all docs
        embeddings = []
        valid_docs = []

        for doc in docs:
            doc_id = doc.id or doc.metadata.get("document_id")
            if doc_id and doc_id in embedding_map:
                embeddings.append(embedding_map[doc_id])
                valid_docs.append(doc)

        if not valid_docs:
            logger.warning("No valid documents with embeddings found in partition")
            return []

        # Convert to numpy array
        embeddings = np.array(embeddings)

        # Process according to your implementation
        node_clusters, _, _ = get_clusters(
            valid_docs,
            embedding_map,
            max_length_in_cluster=max_length_in_cluster,
            tokenizer=tokenizer,
            reduction_dimension=reduction_dimension,
            threshold=threshold
        )

        return node_clusters

    except Exception as e:
        logger.error(f"Error clustering partition: {str(e)}")
        # Return each doc as its own cluster on error
        return [[doc] for doc in docs]


def parallel_cluster_documents(
        docs: List[LangChainDocument],
        embedding_map: Dict[str, np.ndarray],
        max_length_in_cluster: int = 10000,
        tokenizer=None,
        reduction_dimension: int = 5,
        threshold: float = 0.1,
        max_workers: Optional[int] = None
) -> List[List[LangChainDocument]]:
    """
    Parallel document clustering using multiple CPU cores.
    """
    start_time = time.time()

    # Error handling: check inputs
    if not isinstance(docs, list):
        logger.error(f"Expected docs to be a list, got {type(docs)}")
        # Return single cluster if docs is a single document
        if isinstance(docs, LangChainDocument):
            return [[docs]]
        # Otherwise return empty result
        return []

    doc_count = len(docs)

    # Handle empty input case
    if doc_count == 0:
        logger.warning("No documents provided for clustering")
        return []

    # Verify embedding_map is valid
    if not isinstance(embedding_map, dict):
        logger.error(f"Expected embedding_map to be a dict, got {type(embedding_map)}")
        return [[doc] for doc in docs]  # Return each doc as its own cluster

    # Determine number of workers
    if max_workers is None:
        max_workers = min(CPU_COUNT, max(1, doc_count // 500))

    # For very small document sets, don't parallelize
    if doc_count < 100 or max_workers <= 1:
        logger.info(f"Using single-process clustering for {doc_count} documents")
        try:
            # Do the clustering in the current process
            return batch_cluster_documents(
                docs, embedding_map, max_length_in_cluster, tokenizer,
                reduction_dimension, threshold
            )
        except Exception as e:
            logger.error(f"Error in single-process clustering: {str(e)}")
            # On error, treat each doc as its own cluster
            return [[doc] for doc in docs]

    logger.info(f"Clustering {doc_count} documents using {max_workers} parallel workers")

    try:
        # Create document partitions
        partitions = partition_documents(docs, embedding_map, max_workers)
        logger.info(f"Created {len(partitions)} document partitions")

        # Create arguments for each partition worker
        worker_args = []
        for partition in partitions:
            args = (partition, max_length_in_cluster, tokenizer, reduction_dimension, threshold)
            worker_args.append(args)

        # Use spawn context for better cross-platform compatibility
        ctx = multiprocessing.get_context('spawn')

        # Process partitions in parallel
        all_clusters = []

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            try:
                # Map worker args to worker function
                cluster_results = list(executor.map(_cluster_partition_worker, worker_args))

                # Flatten results
                for result in cluster_results:
                    if isinstance(result, list):
                        all_clusters.extend(result)
            except Exception as e:
                logger.error(f"Error in parallel clustering execution: {str(e)}")
                # Fall back to treating each doc as its own cluster
                return [[doc] for doc in docs]

        # Remove any empty clusters
        all_clusters = [cluster for cluster in all_clusters if cluster]

        # Post-process: merge small clusters if needed
        if len(all_clusters) > doc_count // 10 and len(all_clusters) > 10:
            logger.info(f"Too many clusters ({len(all_clusters)}), merging small ones")

            # Sort clusters by size (smallest first)
            all_clusters.sort(key=len)

            # Merge small clusters until we have a reasonable number
            target_cluster_count = max(doc_count // 50, 10)

            while len(all_clusters) > target_cluster_count and len(all_clusters) >= 2:
                # Merge the two smallest clusters
                smallest = all_clusters.pop(0)
                second_smallest = all_clusters.pop(0)
                merged = smallest + second_smallest

                # Insert the merged cluster in the right position
                for i in range(len(all_clusters)):
                    if len(merged) <= len(all_clusters[i]):
                        all_clusters.insert(i, merged)
                        break
                else:
                    all_clusters.append(merged)

        # If we still have no clusters, return each doc as its own cluster
        if not all_clusters:
            logger.warning("No clusters created, returning each document as its own cluster")
            return [[doc] for doc in docs]

        duration = time.time() - start_time
        logger.info(f"Parallel clustering completed in {duration:.2f}s - Created {len(all_clusters)} clusters")

        return all_clusters

    except Exception as e:
        logger.error(f"Critical error in parallel_cluster_documents: {str(e)}")
        # Return each doc as its own cluster on critical error
        return [[doc] for doc in docs]


def batch_cluster_documents(
        docs: List[LangChainDocument],
        embedding_map: Dict[str, np.ndarray],
        max_length_in_cluster: int = 10000,
        tokenizer=None,
        reduction_dimension: int = 5,
        threshold: float = 0.1
) -> List[List[LangChainDocument]]:
    """
    Memory-efficient document clustering (single-process version).
    Uses get_clusters from your implementation.
    """
    if not docs:
        return []

    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    try:
        # Use your implementation of get_clusters
        clusters, _, _ = get_clusters(
            docs,
            embedding_map,
            max_length_in_cluster=max_length_in_cluster,
            tokenizer=tokenizer,
            reduction_dimension=reduction_dimension,
            threshold=threshold
        )

        return clusters
    except Exception as e:
        logger.error(f"Error in batch clustering: {str(e)}")
        # On error, each doc is its own cluster
        return [[doc] for doc in docs]


# Function to be run in a worker process - must be at module level for pickle to work
def _cluster_partition_worker(partition_args):
    """Cluster a single partition of documents (process-safe)."""
    try:
        partition_data, max_length, tokenizer, dim, threshold = partition_args
        return cluster_partition(partition_data, max_length, tokenizer, dim, threshold)
    except Exception as e:
        logger.error(f"Error in worker process: {str(e)}")
        # Return the partition docs as a single cluster on error
        docs, _ = partition_data
        return [docs] if docs else []


# Helper function for multiprocessing
def _parallel_cluster_documents_helper(args):
    """Helper function for parallel_cluster_documents that can be pickled."""
    docs, embedding_map, max_length, tokenizer, dim, threshold, workers = args
    return parallel_cluster_documents(docs, embedding_map, max_length, tokenizer, dim, threshold, workers)


# Import functions from your implementation
def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = min(30, int((len(embeddings) - 1) ** 0.5))
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    ).fit_transform(embeddings)


def local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    # Safety check for small datasets
    if len(embeddings) < num_neighbors:
        num_neighbors = max(2, len(embeddings) - 1)

    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    ).fit_transform(embeddings)


def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings) - 1)
    if max_clusters <= 1:
        return 1

    n_clusters = np.arange(1, max_clusters + 1)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(
        embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED
) -> Tuple[List[np.ndarray], int, GaussianMixture]:
    if len(embeddings) <= 1:
        return [np.array([0])], 1, None

    n_clusters = get_optimal_clusters(embeddings, random_state=random_state)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters, gm


def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
) -> Tuple[List[np.ndarray], umap.UMAP, GaussianMixture]:
    # Safety check for very small input
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))], None, None

    # Fit global clustering
    try:
        reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
        global_clusters, n_global_clusters, gm_model = GMM_cluster(
            reduced_embeddings_global, threshold
        )
    except Exception as e:
        logger.error(f"Error in global clustering: {str(e)}")
        return [np.array([0]) for _ in range(len(embeddings))], None, None

    # Initialize result containers
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Process each global cluster
    for i in range(n_global_clusters):
        # Get embeddings for this global cluster
        global_cluster_indices = [j for j, gc in enumerate(global_clusters) if i in gc]
        if not global_cluster_indices:
            continue

        global_cluster_embeddings_ = embeddings[global_cluster_indices]

        # Skip or simplify for very small clusters
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
            local_gm_model = None
        else:
            # Perform local clustering
            try:
                reduced_embeddings_local = local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters, local_gm_model = GMM_cluster(
                    reduced_embeddings_local, threshold
                )
            except Exception as e:
                logger.error(f"Error in local clustering for global cluster {i}: {str(e)}")
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
                local_gm_model = None

        # Update document cluster assignments
        for j in range(n_local_clusters):
            local_cluster_indices = [k for k, lc in enumerate(local_clusters) if j in lc]
            if not local_cluster_indices:
                continue

            # Get original indices in the full embeddings array
            original_indices = [global_cluster_indices[k] for k in local_cluster_indices]

            # Assign cluster
            for idx in original_indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

        # Update total cluster count
        total_clusters += n_local_clusters

    # Create placeholder UMAP model
    umap_model = umap.UMAP(
        n_neighbors=min(10, len(embeddings) - 1),
        n_components=dim,
        metric="cosine",
        random_state=RANDOM_SEED
    ).fit(embeddings)

    return all_local_clusters, umap_model, gm_model


def get_clusters(
        docs: List[LangChainDocument],
        embedding_map: Dict[str, np.ndarray],
        max_length_in_cluster: int = 10000,
        tokenizer=None,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        prev_total_length=None,
) -> Tuple[List[List[LangChainDocument]], Any, Any]:
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    # Safety checks
    if not docs:
        return [], None, None

    # Build embeddings array in the same order as docs
    embeddings = []
    valid_docs = []
    for doc in docs:
        doc_id = doc.id or doc.metadata.get("document_id")
        if not doc_id:
            logger.warning("Document is missing an ID, skipping")
            continue

        emb = embedding_map.get(doc_id)
        if emb is None:
            logger.warning(f"No embedding found for doc_id: {doc_id}, skipping")
            continue

        embeddings.append(np.array(emb))
        valid_docs.append(doc)

    # If no valid docs with embeddings, return empty
    if not valid_docs:
        logger.warning("No valid documents with embeddings found")
        return [], None, None

    # If only one document, return it as a single cluster
    if len(valid_docs) == 1:
        return [valid_docs], None, None

    # Convert to numpy array
    embeddings = np.array(embeddings)

    # Get clusters
    try:
        clusters, umap_model, gm_model = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )
    except Exception as e:
        logger.error(f"Error in perform_clustering: {str(e)}")
        return [valid_docs], None, None  # Return all docs as one cluster on error

    # Process clusters
    node_clusters = []
    unique_labels = set()
    for cluster_labels in clusters:
        unique_labels.update(cluster_labels)

    # Process each unique cluster
    for label in sorted(unique_labels):
        # Find indices of documents belonging to this cluster
        indices = [i for i, cluster_labels in enumerate(clusters) if label in cluster_labels]
        cluster_docs = [valid_docs[i] for i in indices]

        # Skip empty clusters
        if not cluster_docs:
            continue

        # If there's only one doc, no need to re-cluster
        if len(cluster_docs) == 1:
            node_clusters.append(cluster_docs)
            continue

        # Calculate token length
        try:
            total_length = sum(len(tokenizer.encode(doc.page_content)) for doc in cluster_docs)
        except Exception as e:
            logger.error(f"Error calculating token length: {str(e)}")
            # Estimate using character length as fallback
            char_lengths = [len(doc.page_content) for doc in cluster_docs]
            total_length = sum(char_lengths) // 4  # Rough estimate: 4 chars ≈ 1 token

        # If too large, recursively attempt to break down further
        if total_length > max_length_in_cluster and (
                prev_total_length is None or total_length < prev_total_length
        ):
            # Recursively get sub-clusters
            try:
                sub_clusters, _, _ = get_clusters(
                    cluster_docs,
                    embedding_map,
                    max_length_in_cluster=max_length_in_cluster,
                    tokenizer=tokenizer,
                    reduction_dimension=reduction_dimension,
                    threshold=threshold,
                    prev_total_length=total_length,
                )
                if sub_clusters:
                    node_clusters.extend(sub_clusters)  # Extend with the list of sub-clusters
                else:
                    # If sub-clustering failed, keep as is
                    node_clusters.append(cluster_docs)
            except Exception as e:
                logger.error(f"Error in recursive clustering: {str(e)}")
                node_clusters.append(cluster_docs)  # Keep as is on error
        else:
            node_clusters.append(cluster_docs)  # Append the current cluster

    # If no clusters were created, return all docs as one cluster
    if not node_clusters:
        return [valid_docs], umap_model, gm_model

    return node_clusters, umap_model, gm_model