from typing import List, Dict, Any, Union
import os
from langchain.schema import Document


def prepare_documents(files: List[Dict], prefix: str = "doc") -> List[Document]:
    """
    Process uploaded files into LangChain documents

    Args:
        files: List of file objects from Streamlit
        prefix: Prefix for document IDs

    Returns:
        List of LangChain Document objects
    """
    from langchain.document_loaders import TextLoader, PDFLoader, CSVLoader

    documents = []

    for i, file_info in enumerate(files):
        # Save file to disk temporarily
        temp_path = f"temp_{file_info.name}"
        with open(temp_path, "wb") as f:
            f.write(file_info.getbuffer())

        # Determine loader based on file extension
        ext = os.path.splitext(file_info.name)[1].lower()

        try:
            if ext == ".pdf":
                loader = PDFLoader(temp_path)
            elif ext == ".csv":
                loader = CSVLoader(temp_path)
            else:
                # Default to text loader
                loader = TextLoader(temp_path)

            # Load document
            docs = loader.load()

            # Add metadata
            for j, doc in enumerate(docs):
                doc.metadata["document_id"] = f"{prefix}_{i}_{j}"
                doc.metadata["source"] = file_info.name
                doc.metadata["level"] = 0  # Base level for Raptor

            documents.extend(docs)

        except Exception as e:
            print(f"Error loading file {file_info.name}: {e}")
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return documents


def convert_supabase_to_langchain(
        document_infos: List[Dict],
        document_contents: Dict[str, Union[str, List[str]]],
        use_chunks: bool = True
) -> List[Document]:
    """
    Convert Supabase document data to LangChain Document objects

    Args:
        document_infos: List of document metadata from Supabase
        document_contents: Dict mapping document IDs to content (string or list of chunks)
        use_chunks: If True, treats document_contents as chunks and creates a Document for each chunk

    Returns:
        List of LangChain Document objects
    """
    documents = []

    for doc_info in document_infos:
        doc_id = doc_info.get("id")
        if doc_id not in document_contents:
            continue

        content = document_contents[doc_id]

        if use_chunks and isinstance(content, list):
            # Create a Document for each chunk
            for i, chunk in enumerate(content):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "document_id": f"{doc_id}_chunk_{i}",
                        "parent_document_id": doc_id,
                        "file_name": doc_info.get("file_name", ""),
                        "brain_id": doc_info.get("brain_id", ""),
                        "level": 0,  # Base level for Raptor
                        "source": doc_info.get("file_name", "Unknown"),
                        "chunk_id": i,
                        "total_chunks": len(content)
                    }
                )
                documents.append(doc)
        else:
            # Use combined content as a single Document
            if isinstance(content, list):
                content = "\n".join(content)

            doc = Document(
                page_content=content,
                metadata={
                    "document_id": doc_id,
                    "file_name": doc_info.get("file_name", ""),
                    "brain_id": doc_info.get("brain_id", ""),
                    "level": 0,  # Base level for Raptor
                    "source": doc_info.get("file_name", "Unknown")
                }
            )
            documents.append(doc)

    return documents


def chunk_document(document: Document, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split a document into chunks

    Args:
        document: LangChain Document
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of Document chunks
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    docs = splitter.split_documents([document])

    # Add chunk metadata
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = i
        doc.metadata["total_chunks"] = len(docs)

    return docs