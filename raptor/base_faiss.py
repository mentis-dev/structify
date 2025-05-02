import asyncio
import uuid
import os
from enum import Enum
from typing import Any, List, Optional, Union

import numpy as np
import tiktoken
from langchain.schema import Document as LangChainDocument
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from raptor.clustering import get_clusters  # your GMM + UMAP clustering code
# Import the FAISS-based VectorStoreManager instead of Pinecone version
from app.vectorstore_manager import VectorStoreManager, Query  # Your custom manager
from tenacity import retry, stop_after_attempt, wait_fixed

DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following text, including as many key details as needed:\n\n{context}"
)


class QueryModes(str, Enum):
    """Query modes."""
    tree_traversal = "tree_traversal"
    collapsed = "collapsed"


class SummaryModule:
    def __init__(
            self,
            llm: Optional[OpenAI] = None,
            summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
            num_workers: int = 4,
            show_progress: bool = True,
    ) -> None:
        """
        A module responsible for generating summaries of documents/clusters.
        """
        self.llm = llm or ChatOpenAI(temperature=0)
        self.summary_prompt = summary_prompt
        self.num_workers = num_workers
        self.show_progress = show_progress

        prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.summary_prompt)]
        )
        self.llm_chain = create_stuff_documents_chain(self.llm, prompt_template)

    async def generate_summaries(
            self, documents_per_cluster: List[List[LangChainDocument]]
    ) -> List[str]:
        """
        Generate summaries for each cluster of documents asynchronously.
        """
        summaries = []
        semaphore = asyncio.Semaphore(self.num_workers)

        async def summarize_cluster(cluster_docs: List[LangChainDocument]) -> str:
            # The chain expects "context" as a list of Document objects
            return self.llm_chain.invoke({"context": cluster_docs})

        async def worker(cluster_docs):
            async with semaphore:
                return await summarize_cluster(cluster_docs)

        tasks = [worker(docs) for docs in documents_per_cluster]
        results = await asyncio.gather(*tasks)
        summaries.extend(results)
        return summaries


class RaptorRetriever:
    """
    RaptorRetriever that uses a standard hierarchical (level=0..tree_depth) approach.
    Leverages `VectorStoreManager` (FAISS) for document storage/retrieval.
    Clusters from scratch each time (no incremental approach).
    """

    def __init__(
            self,
            vectorstore_manager: VectorStoreManager,
            tree_depth: int = 3,
            similarity_top_k: int = 2,
            llm: Optional[OpenAI] = None,
            transformations: Optional[List[Any]] = None,
            summary_module: Optional[SummaryModule] = None,
            mode: QueryModes = QueryModes.collapsed,
            verbose: bool = True,
            **kwargs: Any,
    ) -> None:
        """
        :param vectorstore_manager: Manages FAISS (handles chunking, upserts, queries, etc.).
        :param tree_depth: Number of hierarchical clustering levels.
        :param similarity_top_k: K for collapsed retrieval or each level's top-k retrieval.
        :param llm: Summarization LLM (default=ChatOpenAI).
        :param transformations: Additional doc transformations (if any).
        :param summary_module: If None, create a default one with ChatOpenAI.
        :param mode: Collapsed or tree_traversal retrieval mode.
        :param verbose: Print debug info if True.
        """
        self.manager = vectorstore_manager
        self.tree_depth = tree_depth
        self.similarity_top_k = similarity_top_k
        self.mode = mode
        self._verbose = verbose
        self.transformations = transformations or []
        self.summary_module = summary_module or SummaryModule(llm=llm)

        # Use OpenAIEmbeddings by default
        self.embed_model = OpenAIEmbeddings()

        # A folder to potentially store state if needed
        self.model_folder = f"models/{self.manager.index_name}"
        os.makedirs(self.model_folder, exist_ok=True)
        if self._verbose:
            print(f"[RaptorRetriever] Using model folder: {self.model_folder}")

    def _run_transformations(
            self, documents: List[LangChainDocument]
    ) -> List[LangChainDocument]:
        """
        Apply transformations if any (e.g. text cleansing, chunk splitting).
        """
        return documents

    def wipe_all_docs(self, namespace: str = "") -> None:
        """
        Wipe all docs from FAISS in the given namespace.
        """
        if self._verbose:
            print(f"[RaptorRetriever] Wiping all docs in namespace '{namespace}'.")
        self.manager.delete_all_docs(namespace=namespace)

    def _ensure_embedding(self, doc: LangChainDocument) -> np.ndarray:
        """
        Ensure the document has an embedding in its metadata; if not, embed it.
        Returns an np.array of the embedding.
        """
        if "embedding" in doc.metadata and isinstance(doc.metadata["embedding"], list):
            return np.array(doc.metadata["embedding"])
        else:
            emb = self.embed_model.embed_query(doc.page_content)
            doc.metadata["embedding"] = emb
            return np.array(emb)

    @retry(
        stop=stop_after_attempt(3),  # Retry up to 3 times
        wait=wait_fixed(1)  # Wait 1 second between retries
    )
    def _get_documents_by_metadata(
            self, key: str, value: Union[str, int], namespace: str = ""
    ) -> List[LangChainDocument]:
        """
        Retrieve docs by metadata filter using manager.query_by_metadata.
        """
        metadata_filter = {key: value}
        docs = self.manager.query_by_metadata(metadata_filter=metadata_filter, namespace=namespace)
        if not docs:
            raise Exception(f"No docs found by metadata filter {key}={value}")

        return docs

    async def insert(
            self,
            documents: List[LangChainDocument],
            namespace: str = "",
            fresh_start: bool = False,
    ) -> None:
        """
        Insert documents into FAISS, then run standard hierarchical clustering from scratch.

        :param fresh_start: If True, wipe existing data in the namespace first.
        """
        if fresh_start:
            if self._verbose:
                print(f"[RaptorRetriever] Wiping old docs from namespace={namespace}.")
            self.wipe_all_docs(namespace)
        else:
            level_filter = {
                "level": {
                    "$gte": 1
                }
            }

            # Delete higher level documents
            self.manager.delete_with_metadata(level_filter, namespace=namespace)

        # Upsert documents into FAISS
        if self._verbose:
            print(f"[RaptorRetriever] Upserting {len(documents)} docs into namespace={namespace}.")
        self.manager.upsert_documents(
            documents,
            namespace=namespace,
            do_chunking=True,
            auto_delete=True,
            show_progress=self._verbose
        )

        for level in range(self.tree_depth):
            if self._verbose:
                print(f"[RaptorRetriever] Clustering + Summarizing level={level}")

            # 1) Retrieve docs at the current level
            level_docs = self._get_documents_by_metadata("level", level, namespace=namespace)
            if not level_docs:
                if self._verbose:
                    print(f"[RaptorRetriever] No docs found at level={level}, stopping.")
                break

            # 2) Build embedding map
            embedding_map = {}
            for d in level_docs:
                if not d.id:
                    d.id = str(uuid.uuid4())
                emb = self._ensure_embedding(d)
                embedding_map[d.id] = emb

            # 3) Clustering
            clusters, umap_model, gm_model = get_clusters(
                docs=level_docs,
                embedding_map=embedding_map,
                max_length_in_cluster=10000,
                tokenizer=tiktoken.get_encoding("cl100k_base"),
                reduction_dimension=10,
                threshold=0.1,
            )
            if self._verbose:
                print(f"[RaptorRetriever] Found {len(clusters)} clusters at level={level}")

            # 4) Summaries
            summaries = await self.summary_module.generate_summaries(clusters)

            # 5) Create summary docs for next level
            new_nodes = []
            for summary_text, cluster_docs in zip(summaries, clusters):
                summary_doc = LangChainDocument(
                    page_content=summary_text,
                    metadata={"level": level + 1, "document_id": str(uuid.uuid4())}
                )
                # Link child docs with parent_id
                for doc_ in cluster_docs:
                    doc_.metadata["parent_id"] = summary_doc.metadata["document_id"]

                new_nodes.append(summary_doc)

            # 6) Upsert new nodes into FAISS
            #    Also upsert level_docs if you want to refresh them with updated metadata
            all_insert_docs = level_docs + new_nodes
            self.manager.upsert_documents(
                all_insert_docs,
                namespace=namespace,
                do_chunking=False,
                auto_delete=False,
                level=level + 1,
                show_progress=self._verbose
            )

    def _retrieve_collapsed(
            self,
            query: Query,
            namespace: str = "",
            ignore_hierarchy: bool = False
    ) -> List[List[LangChainDocument]]:
        """
        Collapsed retrieval across all docs (limit = similarity_top_k),
        now accepting a Query object.
        """
        # If the user didn't specify top_k in the Query, fallback to self.similarity_top_k
        top_k = query.top_k or self.similarity_top_k

        # Base filter is whatever the user provided in query.metadata_filter
        base_filter = query.metadata_filter.copy() if query.metadata_filter else {}

        # If we are NOT ignoring hierarchy, we enforce "level=0"
        if not ignore_hierarchy:
            base_filter["level"] = 0

        # Create a new Query with merged filter and computed top_k
        final_query = Query(
            text=query.text,
            top_k=top_k,
            metadata_filter=base_filter
        )

        # Forward the final query to your manager
        return self.manager.query([final_query], namespace=namespace)

    def _retrieve_tree_traversal(self, query: Query, namespace: str = "") -> List[List[LangChainDocument]]:
        """
        Hierarchical retrieval: start from top-level nodes (level=tree_depth)
        down to level=0, using 'parent_id' to find children.

        This version takes a Query object (`query`) instead of a raw string.
        """
        level = self.tree_depth
        parent_ids = None
        nodes = []

        # We'll treat any user-provided filter as a "base" to which we add our custom keys.
        base_filter = query.metadata_filter.copy() if query.metadata_filter else {}

        # Use the user's top_k if given, otherwise fallback
        top_k = query.top_k or self.similarity_top_k

        while level >= 0:
            if parent_ids is None or len(parent_ids) == 0:
                # Retrieve top-level
                combined_filter = {**base_filter, "level": level}
                top_query = Query(
                    text=query.text,
                    top_k=top_k,
                    metadata_filter=combined_filter
                )
                top_nodes = self.manager.query([top_query], namespace=namespace)
                if self._verbose:
                    print(f"[tree_traversal] Found {len(top_nodes)} docs at level {level}")

                nodes = top_nodes
                # Collect parent_ids from top_nodes
                parent_ids = [doc.metadata.get("document_id") for doc in top_nodes[0]] if top_nodes and top_nodes[
                    0] else []

            else:
                # Retrieve children by parent_id
                child_nodes = []
                for pid in parent_ids:
                    if pid:
                        combined_filter = {**base_filter, "parent_id": pid}
                        child_query = Query(
                            text=query.text,
                            top_k=top_k,
                            metadata_filter=combined_filter
                        )
                        found_children = self.manager.query([child_query], namespace=namespace)
                        child_nodes.extend(found_children)

                if self._verbose:
                    print(f"[tree_traversal] Found {len(child_nodes)} child docs at level {level}")

                nodes = child_nodes
                # Flatten and get parent_ids
                parent_ids = []
                for node_list in child_nodes:
                    for doc in node_list:
                        parent_id = doc.metadata.get("document_id")
                        if parent_id:
                            parent_ids.append(parent_id)

            level -= 1

        return nodes

    def retrieve(self, query: Query, mode: Optional[QueryModes] = None, namespace: str = "default",
                 ignore_hoerarchy: bool = False) -> List[List[LangChainDocument]]:
        """
        Retrieve docs in 'collapsed' or 'tree_traversal' mode.
        """
        mode = mode or self.mode
        if mode == QueryModes.tree_traversal:
            return self._retrieve_tree_traversal(query, namespace=namespace)
        elif mode == QueryModes.collapsed:
            return self._retrieve_collapsed(query, namespace=namespace, ignore_hierarchy=ignore_hoerarchy)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def persist(self) -> None:
        """
        Explicitly save FAISS indexes to disk.
        This is more important in FAISS than with Pinecone since Pinecone is cloud-based.
        """
        # Our FAISS VectorStoreManager should auto-save after operations
        # But we can implement an explicit save here if needed
        if self._verbose:
            print("[RaptorRetriever] Ensuring FAISS indexes are persisted to disk.")

        # Save any additional state (cluster models, etc) if needed
        # For example, if you're keeping UMAP or GMM models:
        # umap_path = os.path.join(self.model_folder, "umap_model.pkl")
        # with open(umap_path, 'wb') as f:
        #     pickle.dump(self.umap_model, f)