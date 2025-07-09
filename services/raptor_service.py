from typing import List, Dict, Any, Optional
from langchain.schema import Document
from app.vectorstore_manager import VectorStoreManager
from app.models.pydantic_models import Query
from big_raptor.base import OptimizedRaptorRetriever, QueryModes


class RaptorService:
    """Service layer for Raptor operations"""

    def __init__(
            self,
            index_name: str = "raptor_test",
            tree_depth: int = 3,
            similarity_top_k: int = 5,
            mode: QueryModes = QueryModes.collapsed,
            namespace: str = "default",
            dimension: int = 3072
    ):
        """Initialize Raptor service"""
        # Initialize the vector store manager
        self.manager = VectorStoreManager(
            index_name=index_name,
            embedding_dimension=dimension,
            index_folder="faiss_indexes"
        )

        # Initialize the retriever
        self.retriever = OptimizedRaptorRetriever(
            vectorstore_manager=self.manager,
            tree_depth=tree_depth,
            similarity_top_k=similarity_top_k,
            do_chunking=False,
            mode=mode,
            verbose=True
        )

        self.index_name = index_name
        self.namespace = namespace

    async def index_documents(
            self,
            documents: List[Document],
            namespace: str = None,
            fresh_start: bool = False
    ) -> int:
        """
        Index documents with Raptor

        Args:
            documents: List of LangChain documents
            namespace: Namespace to use (or default if None)
            fresh_start: Whether to clear existing index

        Returns:
            Number of indexed documents
        """
        # Use specified namespace or default
        ns = namespace or self.namespace

        # Index documents with Raptor
        await self.retriever.insert(
            documents=documents,
            namespace=ns,
            fresh_start=fresh_start
        )

        # Persist the index
        self.retriever.persist()

        return len(documents)

    def retrieve(
            self,
            query_text: str,
            mode: Optional[QueryModes] = None,
            top_k: Optional[int] = None,
            namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents with Raptor

        Args:
            query_text: Query text
            mode: Query mode (collapsed or tree_traversal)
            top_k: Maximum number of results to return
            namespace: Namespace to search (or default if None)

        Returns:
            Dict with query results and metadata
        """
        # Create query object
        query = Query(text=query_text, top_k=top_k)

        # Use specified mode or default
        query_mode = mode or self.retriever.mode

        # Use specified namespace or default
        ns = namespace or self.namespace

        # Perform retrieval
        results = self.retriever.retrieve(
            query=query,
            mode=query_mode,
            namespace=ns
        )

        # Format results for easier consumption
        formatted_results = []
        for i, doc_list in enumerate(results):
            for j, doc in enumerate(doc_list):
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "group": i,
                    "rank": j
                })

        return {
            "query": query_text,
            "mode": str(query_mode),
            "namespace": ns,
            "results": formatted_results,
            "count": len(formatted_results)
        }

    def get_available_indexes(self) -> List[str]:
        """Get list of available indexes"""
        from app.vectorstore_manager import list_indexes
        return list_indexes()