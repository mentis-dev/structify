import asyncio
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import tiktoken
from langchain.schema import Document as LangChainDocument
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.vectorstores import VectorStore
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from raptor.clustering import get_clusters  # your GMM + UMAP clustering code

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
        """Module responsible for summarizing clusters of documents."""
        # Default to ChatOpenAI if none provided
        self.llm = llm or ChatOpenAI(temperature=0)
        self.summary_prompt = summary_prompt
        self.num_workers = num_workers
        self.show_progress = show_progress

        # Create a combine_documents chain with a chat prompt
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", self.summary_prompt)]
        )
        self.llm_chain = create_stuff_documents_chain(self.llm, prompt_template)

    async def generate_summaries(
        self, documents_per_cluster: List[List[LangChainDocument]]
    ) -> List[str]:
        """Generate summaries of documents per cluster asynchronously."""
        summaries = []
        semaphore = asyncio.Semaphore(self.num_workers)

        async def summarize_cluster(cluster_docs: List[LangChainDocument]) -> str:
            # The chain expects a list of Document objects under "context"
            return self.llm_chain.invoke({"context": cluster_docs})

        async def worker(cluster_docs):
            async with semaphore:
                return await summarize_cluster(cluster_docs)

        tasks = [worker(docs) for docs in documents_per_cluster]
        results = await asyncio.gather(*tasks)
        summaries.extend(results)
        return summaries


class RaptorRetriever:
    """Store-agnostic RaptorRetriever that requires a pre-initialized VectorStore."""

    def __init__(
        self,
        vectorstore: VectorStore,  # You must provide an existing vector store.
        tree_depth: int = 3,
        similarity_top_k: int = 2,
        llm: Optional[OpenAI] = None,
        transformations: Optional[List[Any]] = None,  # e.g., text splitters
        summary_module: Optional[SummaryModule] = None,
        mode: QueryModes = QueryModes.collapsed,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        :param vectorstore: A pre-initialized vector store (e.g. Chroma, Pinecone).
        :param tree_depth: The number of hierarchical levels to cluster/summarize.
        :param similarity_top_k: K for collapsed retrieval.
        :param llm: The LLM used for summarization (default = ChatOpenAI).
        :param transformations: If you want the retriever to apply transformations when
                                new docs are inserted, provide them here.
        :param summary_module: If None, a default SummaryModule with ChatOpenAI is used.
        :param mode: "collapsed" or "tree_traversal".
        :param verbose: Toggle debug prints.
        """
        self.vectorstore = vectorstore
        self.tree_depth = tree_depth
        self.similarity_top_k = similarity_top_k
        self.mode = mode
        self._verbose = verbose
        self.transformations = transformations or []

        # Create a default SummaryModule if needed
        self.summary_module = summary_module or SummaryModule(llm=llm)

        # For any new docs we add, we'll embed with OpenAIEmbeddings by default
        self.embed_model = OpenAIEmbeddings()

    def _run_transformations(
        self, documents: List[LangChainDocument]
    ) -> List[LangChainDocument]:
        """Apply any custom transformations to the documents (e.g. splitting)."""
        # If transformations are provided, apply them here:
        # For now, just a pass-through

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        transformed_docs = []
        for doc in documents:
            splits = text_splitter.split_text(doc.page_content)
            for chunk in splits:
                transformed_docs.append(
                    LangChainDocument(
                        page_content=chunk,
                        metadata=doc.metadata.copy()
                    )
                )
        return transformed_docs


    async def insert(self, documents: List[LangChainDocument]) -> None:
        """
        Ingest new documents and build hierarchical summaries for each level.
        - Assign metadata["level"] = 0 if missing
        - Assign metadata["id"] = str(uuid.uuid4()) if missing
        - Embed if no doc.metadata["embedding"] present
        - Upsert to the vector store
        - Perform clustering & summarization at each level
        """
        # 1. Transform docs if needed
        documents = self._run_transformations(documents)

        # 2. Ensure each doc has an id, a level=0, and an embedding
        for doc in documents:
            if "id" not in doc.metadata:
                doc.metadata["id"] = str(uuid.uuid4())
            if "level" not in doc.metadata:
                doc.metadata["level"] = 0
#            if "embedding" not in doc.metadata:
#                doc.metadata["embedding"] = self.embed_model.embed_query(doc.page_content)

        # 3. Add them to the vector store
        #    Many stores have an .add_documents() or .add_texts() method.
        #    We'll assume .add_documents() is available and can do re-embedding if needed
        #    but we already have embeddings, so pass them. This is store-specific:
        await self._add_documents_to_store(documents)

        # 4. Hierarchy build: from level=0 up to tree_depth
        for level in range(self.tree_depth):
            if self._verbose:
                print(f"Clustering and summarizing level {level}")

            level_docs = self._get_documents_by_metadata("level", level)
            if not level_docs:
                if self._verbose:
                    print(f"No documents at level {level}. Stopping.")
                break

            # Build an embedding map
            embedding_map = {}
            for doc in level_docs:
                doc_id = doc.metadata["id"]
                emb = doc.metadata.get("embedding")
                if emb is None:
                    # fallback embed if store didn't store it
                    emb = self.embed_model.embed_query(doc.page_content)
                    doc.metadata["embedding"] = emb
                embedding_map[doc_id] = emb

            # Cluster
            clusters = get_clusters(
                docs=level_docs,
                embedding_map=embedding_map,
                max_length_in_cluster=10000,
                tokenizer=tiktoken.get_encoding("cl100k_base"),
                reduction_dimension=10,
                threshold=0.1,
            )
            if self._verbose:
                print(f"Found {len(clusters)} clusters at level {level}.")

            # Summaries
            summaries = await self.summary_module.generate_summaries(clusters)
            new_nodes = []
            for summary_text, cluster_docs in zip(summaries, clusters):
                summary_doc = LangChainDocument(
                    page_content=summary_text,
                    metadata={
                        "id": str(uuid.uuid4()),
                        "level": level + 1,
                    },
                )
                new_nodes.append(summary_doc)

                parent_id = summary_doc.metadata["id"]
                for d in cluster_docs:
                    d.metadata["parent_id"] = parent_id

            # Upsert the new summary nodes
            for node in new_nodes:
                if "embedding" not in node.metadata:
                    node.metadata["embedding"] = self.embed_model.embed_query(node.page_content)
            await self._add_documents_to_store(level_docs + new_nodes)

    async def _add_documents_to_store(self, docs: List[LangChainDocument]) -> None:
        """
        A helper for adding/upserting docs into the vector store.
        This is store-specific logic, so you might have to adapt it.
        For example, Chroma uses add_documents(), Pinecone might need add_texts() etc.
        """
        # Example: Some stores have an async method. If not, remove 'async/await'.
        # We'll assume a typical interface: vectorstore.add_documents(...)
        # that either handles or re-embeds the documents.

        # If your store doesn't have an async method, do this:
        # self.vectorstore.add_documents(docs)

        if hasattr(self.vectorstore, "aadd_documents") and callable(self.vectorstore.aadd_documents):
            await self.vectorstore.aadd_documents(docs)
        else:
            # Fallback to sync version
            self.vectorstore.add_documents(docs)

    def _get_documents_by_metadata(self, key: str, value: Union[str, int]) -> List[LangChainDocument]:
        """
        Retrieve docs from the vector store by metadata.
        This depends on the store's ability to do metadata-based filtering.
        If your store doesn't support a filter param, you'll have to do an in-memory filter.
        """
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 10000, "filter": {key: value}, "include_values": True},

        )
        return retriever.get_relevant_documents("metadata query")

    def _retrieve_collapsed(self, query_str: str) -> List[LangChainDocument]:
        """Simple retrieval across all docs, limited by k."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.similarity_top_k}
        )
        return retriever.get_relevant_documents(query_str)

    def _retrieve_tree_traversal(self, query_str: str) -> List[LangChainDocument]:
        """
        Traverse from top-level nodes (level=tree_depth) down to level=0,
        filtering by parent_id at each stage.
        """
        level = self.tree_depth
        parent_ids = None
        nodes = []

        while level >= 0:
            if parent_ids is None:
                # Retrieve top-level docs
                top_nodes = self._get_nodes_by_query_and_filter(query_str, {"level": level})
                if self._verbose:
                    print(f"[tree_traversal] Found {len(top_nodes)} nodes at level {level}")
                nodes = top_nodes
                parent_ids = [doc.metadata.get("id") for doc in top_nodes]
            else:
                child_nodes = []
                for pid in parent_ids:
                    if pid:
                        child_nodes.extend(self._get_nodes_by_query_and_filter(query_str, {"parent_id": pid}))
                if self._verbose:
                    print(f"[tree_traversal] Found {len(child_nodes)} child nodes at level {level}")
                nodes = child_nodes
                parent_ids = [doc.metadata.get("id") for doc in child_nodes]

            level -= 1

        return nodes

    def _get_nodes_by_query_and_filter(
        self, query_str: str, filter_dict: Dict[str, Any]
    ) -> List[LangChainDocument]:
        """Helper method for retrieving with a metadata filter and a query."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.similarity_top_k, "filter": filter_dict}
        )
        return retriever.get_relevant_documents(query_str)

    def retrieve(self, query_str: str, mode: Optional[QueryModes] = None) -> List[LangChainDocument]:
        """
        Retrieve docs using either a collapsed approach or a hierarchical tree traversal.
        """
        mode = mode or self.mode
        if mode == QueryModes.tree_traversal:
            return self._retrieve_tree_traversal(query_str)
        elif mode == QueryModes.collapsed:
            return self._retrieve_collapsed(query_str)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def persist(self) -> None:
        """Persist the vector store, if it supports persistence."""
        if hasattr(self.vectorstore, "persist") and callable(self.vectorstore.persist):
            self.vectorstore.persist()
        else:
            if self._verbose:
                print("Vector store does not support persistence.")
