import random
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import tiktoken
import umap
from langchain.schema import Document as LangChainDocument
from sklearn.mixture import GaussianMixture

RANDOM_SEED = 224
random.seed(RANDOM_SEED)

def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    ).fit_transform(embeddings)

def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric, random_state=RANDOM_SEED
    ).fit_transform(embeddings)

def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]

def GMM_cluster(
    embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED
) -> Tuple[List[np.ndarray], int]:
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters, gm  # Return gm model

def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> Tuple[List[np.ndarray], umap.UMAP, GaussianMixture]:
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))], None, None

    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters, gm_model = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    umap_models = []
    gm_models = []

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
            local_gm_model = None
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters, local_gm_model = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if local_gm_model:
            gm_models.append(local_gm_model)

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = []
            for emb in local_cluster_embeddings_:
                idx = np.where((embeddings == emb).all(axis=1))[0]
                if len(idx) > 0:
                    indices.append(idx[0])

            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    umap_model = umap.UMAP(
        n_neighbors=10, n_components=dim, metric="cosine", random_state=RANDOM_SEED
    ).fit(embeddings)  # Example UMAP model

    return all_local_clusters, umap_model, gm_model

def get_clusters(
    docs: List[LangChainDocument],
    embedding_map: Dict[str, np.ndarray],
    max_length_in_cluster: int = 10000,  # 10k tokens max per cluster
    tokenizer = tiktoken.get_encoding("cl100k_base"),  # default tokenizer
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    prev_total_length=None,
) -> (List[List[LangChainDocument]], Any, Any):
    # Build embeddings array in the same order as docs
    embeddings = []
    for doc in docs:
        doc_id = doc.metadata.get("id") or doc.id
        if not doc_id:
            raise ValueError("Document is missing an 'id' in its metadata.")
        emb = embedding_map.get(doc_id)
        if emb is None:
            raise ValueError(f"No embedding found for doc_id: {doc_id}")
        embeddings.append(np.array(emb))

    embeddings = np.array(embeddings)
    clusters, umap_model, gm_model = perform_clustering(embeddings, dim=reduction_dimension, threshold=threshold)

    node_clusters = []  # Changed from dict to list
    # Assuming 'clusters' is a list where each element corresponds to a document's cluster labels
    # For non-overlapping clusters, each document should belong to exactly one cluster
    # Adjust the logic based on how 'clusters' is structured

    unique_labels = np.unique(np.concatenate(clusters))
    for label in unique_labels:
        # Find indices of documents belonging to the current cluster
        indices = [i for i, cluster in enumerate(clusters) if label in cluster]
        cluster_docs = [docs[i] for i in indices]

        # If there's only one doc, no need to re-cluster
        if len(cluster_docs) == 1:
            node_clusters.append(cluster_docs)
            continue

        # Calculate token length
        total_length = sum(len(tokenizer.encode(doc.page_content)) for doc in cluster_docs)

        # If too large, recursively attempt to break down further
        if total_length > max_length_in_cluster and (
            prev_total_length is None or total_length < prev_total_length
        ):
            # Recursively get sub-clusters
            sub_clusters, _, _ = get_clusters(
                cluster_docs,
                embedding_map,
                max_length_in_cluster=max_length_in_cluster,
                tokenizer=tokenizer,
                reduction_dimension=reduction_dimension,
                threshold=threshold,
                prev_total_length=total_length,
            )
            node_clusters.extend(sub_clusters)  # Extend with the list of sub-clusters
        else:
            node_clusters.append(cluster_docs)  # Append the current cluster

    return node_clusters, umap_model, gm_model


def incremental_cluster_assign(
    umap_model: umap.UMAP,           # Fitted UMAP model from your previous run
    gm_model: GaussianMixture,      # Fitted GaussianMixture from your previous run
    new_embeddings: np.ndarray,
    threshold: float = 0.1,
) -> List[List[int]]:
    """
    Returns a list of cluster label-lists, one for each new embedding.
    Each embedding can belong to multiple clusters if its probability > threshold.
    """
    # 1. UMAP transform on new embeddings
    new_reduced = umap_model.transform(new_embeddings)

    # 2. GMM predict probabilities on the new data
    probs = gm_model.predict_proba(new_reduced)

    # 3. Assign each embedding to cluster(s) with probability above threshold
    new_labels = [np.where(prob > threshold)[0] for prob in probs]

    return new_labels

def insert_new_docs_incremental(
    old_docs: List[LangChainDocument],
    old_assignments: List[np.ndarray],  # old_docs[i] cluster assignment from your last run
    umap_model: umap.UMAP,             # fitted UMAP from last run
    gm_model: GaussianMixture,        # fitted GMM from last run
    new_docs: List[LangChainDocument],
    embedding_map: Dict[str, np.ndarray],
    threshold: float = 0.1
) -> Tuple[List[LangChainDocument], List[np.ndarray], umap.UMAP, GaussianMixture]:
    """
    Integrate new documents into existing clusters using incremental clustering.
    Returns updated docs, assignments, and updated models.
    """
    # 1. Convert new_docs to embeddings
    new_embs = []
    for doc in new_docs:
        emb = embedding_map.get(doc.id)
        if emb is None:
            raise ValueError(f"No embedding found for doc_id: {doc.id}")
        new_embs.append(emb)
    new_embs = np.array(new_embs)

    # 2. Get new cluster assignments via incremental logic
    new_labels = incremental_cluster_assign(umap_model, gm_model, new_embs, threshold)

    # 3. If cluster assignment is empty for a doc, create a new cluster ID
    existing_labels = np.unique(old_assignments) if len(old_assignments) > 0 else np.array([])
    next_cluster_id = int(existing_labels.max() + 1) if len(existing_labels) > 0 else 0

    final_new_assignments = []
    for label_list in new_labels:
        if len(label_list) == 0:
            # no cluster above threshold, create a new cluster
            label_list = np.array([next_cluster_id])
            next_cluster_id += 1
        final_new_assignments.append(label_list)

    # 4. Merge old docs + assignments with new
    all_docs = old_docs + new_docs
    all_assignments = list(old_assignments) + list(final_new_assignments)

    # 5. Optionally, update UMAP and GMM models with new data
    # For simplicity, we'll re-fit UMAP and GMM with all embeddings
    all_embeddings = np.array([embedding_map[doc.id] for doc in all_docs])
    new_umap_model = umap.UMAP(
        n_neighbors=10, n_components=10, metric="cosine", random_state=RANDOM_SEED
    ).fit(all_embeddings)
    new_reduced_embeddings = new_umap_model.transform(all_embeddings)
    new_gm_model = GaussianMixture(
        n_components=get_optimal_clusters(new_reduced_embeddings),
        random_state=RANDOM_SEED
    ).fit(new_reduced_embeddings)

    # 6. Assign clusters based on updated models
    new_assignments = []
    for emb in all_embeddings:
        reduced = new_umap_model.transform([emb])[0]
        probs = new_gm_model.predict_proba([reduced])[0]
        labels = np.where(probs > threshold)[0]
        if len(labels) == 0:
            labels = np.array([new_gm_model.means_.shape[0]])  # new cluster
        new_assignments.append(labels)

    # 7. Update docs' metadata with new cluster assignments
    for doc, labels in zip(all_docs, new_assignments):
        doc.metadata["incremental_labels"] = labels.tolist()

    return all_docs, new_assignments, new_umap_model, new_gm_model
