import asyncio

from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

from app.models.pydantic_models import Query
# Import the FAISS-based VectorStoreManager instead of Pinecone version
from app.vectorstore_manager_ import VectorStoreManager

load_dotenv()
# Import your store-agnostic RaptorRetriever code
from raptor.base_faiss import RaptorRetriever, QueryModes

# 1. Load PDF documents
pdf_loader = PyPDFLoader("app/raptor_paper.pdf")
raw_docs = pdf_loader.load()
ids = [d.metadata.setdefault("document_id", str(i)) for (i, d) in enumerate(raw_docs)]

# 2. Initialize the FAISS-based VectorStoreManager
# Note: No need for Pinecone API key or environment
manager = VectorStoreManager(
    index_name="test",
    embedding_model="text-embedding-3-large",
    embedding_dimension=3072,
    index_folder="faiss_indexes"  # Optional: specify where to store FAISS index files
)

# 3. Initialize RaptorRetriever with FAISS as the vector store
raptor_retriever = RaptorRetriever(
    vectorstore_manager=manager,  # Pass in the FAISS-based store here
    tree_depth=3,                 # or however deep you want the hierarchy
    similarity_top_k=2,
    mode=QueryModes.collapsed,    # or QueryModes.tree_traversal
    verbose=True
)

# 4. Insert new documents (hierarchical clustering + summarization)

# Uncomment to insert documents
asyncio.run(raptor_retriever.insert(raw_docs[3:], namespace="raptor-ns", fresh_start=True))

# 5. Retrieve using collapsed mode
collapsed_docs = raptor_retriever.retrieve(Query(text="What is Raptor?"), namespace="raptor-ns")
print(f"Collapsed results: {len(collapsed_docs)} docs")
if collapsed_docs:
    print(collapsed_docs[0][0].page_content)

# 6. Retrieve using tree_traversal mode
tree_docs = raptor_retriever.retrieve(
    Query(text="What is Raptor compared to?"),
    mode=QueryModes.tree_traversal,
    namespace="raptor-ns"
)
print(f"Tree results: {len(tree_docs)} docs")
if tree_docs:
    print(tree_docs[0][0].page_content)

# 7. (Optional) If your RaptorRetriever has a persist method
# You may want to call it explicitly to ensure FAISS indexes are saved

# Uncomment to persist
# raptor_retriever.persist()

# 8. Add more documents incrementally
# asyncio.run(raptor_retriever.insert(raw_docs[:3], namespace="raptor-ns", fresh_start=False))

# 9. (Optional) Persist again after adding more documents
# raptor_retriever.persist()

# 10. Additional query examples (commented out)
# collapsed_docs = raptor_retriever.retrieve(Query(text="What is Raptor compared against?"), namespace="raptor-ns", ignore_hoerarchy=True)
# print(f"Collapsed results: {len(collapsed_docs)} docs")
# if collapsed_docs:
#     print(collapsed_docs[0].page_content)
#
# tree_docs = raptor_retriever.retrieve(
#     Query(text="What is Raptor compared against?"),
#     mode=QueryModes.tree_traversal,
#     namespace="raptor-ns"
# )
# print(f"Tree results: {len(tree_docs)} docs")
# if tree_docs:
#     print(tree_docs[0].page_content)