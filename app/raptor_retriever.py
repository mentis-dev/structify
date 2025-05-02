from typing import Dict, List, Any, Optional
from langchain.schema import Document as LangChainDocument
from langchain_core.runnables import RunnableConfig

from app.vectorstore_manager import VectorStoreManager
from app.models.pydantic_models import Query
from raptor_.base_faiss import RaptorRetriever, QueryModes


# Example graph component for Raptor integration
async def setup_raptor_retriever(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Set up the Raptor retriever with documents from the workspace.
    This component should be inserted into your graph workflow.
    """
    # Get the documents from the state
    documents = state.get("processed_documents", {})

    if not documents:
        return {**state, "error": "No documents found for Raptor indexing"}

    # Convert your document format to LangChain documents
    langchain_docs = []
    for doc_id, doc_data in documents.items():
        doc_content = doc_data.get("content", "")
        doc_metadata = {
            "document_id": doc_id,
            "title": doc_data.get("title", ""),
            "source": doc_data.get("source", ""),
            "workspace_id": state.get("workspace_id", ""),
            "brain_id": doc_data.get("brain_id", "")
        }
        langchain_docs.append(
            LangChainDocument(
                page_content=doc_content,
                metadata=doc_metadata
            )
        )

    # Create a namespace using the workspace ID
    namespace = f"workspace_{state.get('workspace_id', 'default')}"

    try:
        # Initialize the FAISS-based VectorStoreManager
        manager = VectorStoreManager(
            index_name=f"workspace_{state.get('workspace_id', 'default')}",
            embedding_model="text-embedding-3-large",
            embedding_dimension=3072,
            index_folder="faiss_indexes"
        )

        # Initialize the RaptorRetriever
        raptor = RaptorRetriever(
            vectorstore_manager=manager,
            tree_depth=3,  # Adjust based on your document structure
            similarity_top_k=5,
            mode=QueryModes.collapsed,
            verbose=True
        )

        # Insert documents into Raptor (this builds the hierarchical index)
        print(f"Indexing {len(langchain_docs)} documents with Raptor...")
        await raptor.insert(langchain_docs, namespace=namespace, fresh_start=True)

        # Update the state with the Raptor retriever
        return {
            **state,
            "raptor_retriever": raptor,
            "raptor_namespace": namespace
        }

    except Exception as e:
        return {**state, "error": f"Failed to set up Raptor retriever: {str(e)}"}


async def retrieve_with_raptor(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    Retrieve relevant documents using Raptor for analysis.
    This component should be used before analysis steps in your workflow.
    """
    # Get the Raptor retriever from the state
    raptor = state.get("raptor_retriever")
    namespace = state.get("raptor_namespace", "default")

    if not raptor:
        return {**state, "error": "Raptor retriever not initialized"}

    try:
        # Get the query from the state or use a default
        query_text = state.get("raptor_query", "Identify stakeholders, factors, and pain points")

        # Create a Query object
        query = Query(text=query_text)

        # Retrieve documents using the collapsed mode (or use tree_traversal if preferred)
        retrieved_docs = raptor.retrieve(
            query=query,
            namespace=namespace,
            mode=QueryModes.collapsed
        )

        # Extract the content from retrieved documents
        retrieved_content = ""
        retrieved_metadata = []

        if retrieved_docs:
            for i, doc_list in enumerate(retrieved_docs):
                if doc_list:
                    for doc in doc_list:
                        # Add document content to the context string
                        retrieved_content += f"\n--- Document Segment ---\n"
                        retrieved_content += doc.page_content
                        retrieved_content += "\n"

                        # Collect metadata for reference
                        retrieved_metadata.append({
                            "document_id": doc.metadata.get("document_id", ""),
                            "title": doc.metadata.get("title", ""),
                            "source": doc.metadata.get("source", ""),
                            "level": doc.metadata.get("level", 0),
                            "score": doc.metadata.get("score", 0.0)
                        })

        # Update the state with the retrieved content
        return {
            **state,
            "raptor_context": retrieved_content,
            "raptor_retrieved_docs": retrieved_metadata,
            "topic": retrieved_content  # Replace the 'topic' field with retrieved content
        }

    except Exception as e:
        return {**state, "error": f"Failed to retrieve with Raptor: {str(e)}"}


# Add these components to your graph workflow
def enhance_graph_with_raptor(graph):
    """
    Enhance your existing graph workflow with Raptor components.
    This is a conceptual example - you'll need to adapt it to your actual graph structure.
    """
    # This is a conceptual example - in practice, you'd need to update your actual graph
    # For example:

    # 1. First, get documents from the workspace
    # 2. Set up Raptor with those documents
    # 3. Use Raptor for retrieval before analysis
    # 4. Continue with analysis using retrieved context

    # Example integration points (pseudocode):
    # graph.add_node("setup_raptor", setup_raptor_retriever)
    # graph.add_edge("process_documents", "setup_raptor")
    # graph.add_node("retrieve_with_raptor", retrieve_with_raptor)
    # graph.add_edge("setup_raptor", "retrieve_with_raptor")
    # graph.add_edge("retrieve_with_raptor", "extract_info")

    return graph