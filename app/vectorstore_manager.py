import json
import os
import time
import uuid
import numpy as np
import faiss
import pickle
from functools import lru_cache
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv

load_dotenv()

from raptor.utils import get_value_by_key
from raptor.models import Query, ChunkSectionConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from pprint import pformat
import os.path

# Constants
DEFAULT_SECTION_CONFIG = {
    "fields": None,
    "combine_list": False,
    "metadata_keys": [],
    "content_exclude_keys": [],
    "section_name": None
}


class FaissIndexManager:
    """Simple class to manage FAISS indexes"""

    @staticmethod
    def save_index(index, index_path):
        """Save FAISS index to disk"""
        directory = os.path.dirname(index_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        faiss.write_index(index, index_path)

    @staticmethod
    def load_index(index_path):
        """Load FAISS index from disk"""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found at {index_path}")
        return faiss.read_index(index_path)

    @staticmethod
    def create_index(dimension):
        """Create a new FAISS index"""
        return faiss.IndexFlatL2(dimension)  # L2 distance


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


class VectorStoreManager:
    """
    VectorStoreManager for managing FAISS indexes using
    LangChain's chunking + OpenAI Embeddings for encoding.
    """

    def __init__(
            self,
            index_name: str,
            text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
            embeddings: Optional[OpenAIEmbeddings] = None,
            embedding_model: str = "text-embedding-3-large",
            embedding_dimension: int = 3072,
            default_top_k: int = 5,
            index_folder: str = "faiss_indexes"
    ):
        if default_top_k < 1:
            raise ValueError("default_top_k must be greater than 0")

        self.index_name = self._format_index_name(index_name)
        self.index = None
        self.index_folder = index_folder
        self.default_top_k = default_top_k
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter()
        self.embeddings = embeddings or OpenAIEmbeddings(model=embedding_model)
        self.dimension = embedding_dimension

        # Metadata storage - FAISS only stores vectors, not metadata
        self.metadata_path = os.path.join(self.index_folder, f"{self.index_name}_metadata.pkl")
        self.metadata_store = {}
        self.id_to_index = {}  # Maps document IDs to FAISS indices

        # Create folder if it doesn't exist
        if not os.path.exists(self.index_folder):
            os.makedirs(self.index_folder)

        # Load or create the index
        self.create_index(dimension=self.dimension)

    def _format_index_name(self, name: str) -> str:
        """Format index name to be compatible with filesystem"""
        return name.replace(" ", "_").lower()

    def _get_index_path(self):
        """Get the path to the FAISS index file"""
        return os.path.join(self.index_folder, f"{self.index_name}.index")

    def connect(self) -> None:
        """
        Connect to the FAISS index.
        """
        if not self.index:
            index_path = self._get_index_path()
            if not os.path.exists(index_path):
                raise RuntimeError(f"Index '{self.index_name}' does not exist. Create it first.")

            try:
                self.index = FaissIndexManager.load_index(index_path)
                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'rb') as f:
                        self.metadata_store, self.id_to_index = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to connect to index '{self.index_name}'.") from e

    def create_index(
            self,
            dimension: Optional[int] = 3072,
    ) -> None:
        """
        Create a new FAISS index with the specified configuration.
        """
        try:
            existing_indexes = list_indexes(self.index_folder)
            index_path = self._get_index_path()

            if self.index_name not in existing_indexes:
                print(f"Index '{self.index_name}' does not exist. Creating it now...")

                if not dimension:
                    raise RuntimeError("Embeddings must have a 'dimension' attribute.")

                # Create a new FAISS index
                self.index = FaissIndexManager.create_index(dimension)

                # Initialize empty metadata store
                self.metadata_store = {}
                self.id_to_index = {}

                # Save the index and metadata
                FaissIndexManager.save_index(self.index, index_path)
                self._save_metadata()

                print(f"Index '{self.index_name}' created successfully.")
            else:
                print(f"Index '{self.index_name}' already exists.")
                self.connect()
        except Exception as e:
            raise RuntimeError(f"Failed to create index: {e}") from e

    def _save_metadata(self):
        """Save metadata to disk"""
        with open(self.metadata_path, 'wb') as f:
            pickle.dump((self.metadata_store, self.id_to_index), f)

    def process_section(self, section_key: str, section_data: Any, config: Dict[str, Any]) -> List[Document]:
        """
        Process a section (dict or list) according to the given config.
        Returns a list of Document objects.
        """
        documents = []

        if isinstance(section_data, dict):
            fields = config.get("fields")
            content_lines = []
            if fields is not None:
                for key in fields:
                    if key in section_data:
                        value = section_data[key]
                        value_str = pformat(value, indent=2) if isinstance(value, (list, dict)) else str(value)
                        content_lines.append(f"{key}: {value_str}")
            else:
                for key, value in section_data.items():
                    value_str = pformat(value, indent=2) if isinstance(value, (list, dict)) else str(value)
                    content_lines.append(f"{key}: {value_str}")
            content = "\n".join(content_lines)
            metadata = {"section": config.get("section_name", section_key)}
            for mkey in config.get("metadata_keys", []):
                if mkey in section_data:
                    metadata[mkey] = section_data[mkey]
            documents.append(
                Document(page_content=f"{config.get('section_name', section_key)}:\n{content}", metadata=metadata))

        elif isinstance(section_data, list):
            if config.get("combine_list"):
                combined_lines = []
                for item in section_data:
                    if isinstance(item, dict):
                        item_lines = []
                        fields = config.get("fields")
                        if fields is not None:
                            for key in fields:
                                if key in item and key not in config.get("content_exclude_keys", []):
                                    value = item[key]
                                    value_str = pformat(value, indent=2) if isinstance(value, (list, dict)) else str(
                                        value)
                                    item_lines.append(f"{key}: {value_str}")
                        else:
                            for key, value in item.items():
                                if key in config.get("content_exclude_keys", []):
                                    continue
                                value_str = pformat(value, indent=2) if isinstance(value, (list, dict)) else str(value)
                                item_lines.append(f"{key}: {value_str}")
                        combined_lines.append("\n".join(item_lines))
                    else:
                        combined_lines.append(str(item))
                content = "\n\n".join(combined_lines)
                metadata = {"section": config.get("section_name", section_key)}
                documents.append(
                    Document(page_content=f"{config.get('section_name', section_key)}:\n{content}", metadata=metadata))
            else:
                for item in section_data:
                    if isinstance(item, dict):
                        metadata = {"section": config.get("section_name", section_key)}
                        for mkey in config.get("metadata_keys", []):
                            if mkey in item:
                                metadata[mkey] = item[mkey]
                        item_lines = []
                        fields = config.get("fields")
                        if fields is not None:
                            for key in fields:
                                if key in item and key not in config.get("content_exclude_keys", []):
                                    value = item[key]
                                    value_str = pformat(value, indent=2) if isinstance(value, (list, dict)) else str(
                                        value)
                                    item_lines.append(f"{key}: {value_str}")
                        else:
                            for key, value in item.items():
                                if key in config.get("content_exclude_keys", []):
                                    continue
                                value_str = pformat(value, indent=2) if isinstance(value, (list, dict)) else str(value)
                                item_lines.append(f"{key}: {value_str}")
                        content = "\n".join(item_lines)
                        documents.append(Document(page_content=f"{config.get('section_name', section_key)}:\n{content}",
                                                  metadata=metadata))
                    else:
                        documents.append(Document(page_content=str(item),
                                                  metadata={"section": config.get("section_name", section_key)}))
        return documents

    def chunk_json(self, data: Dict[str, Any], chunk_config: Optional[Dict[str, ChunkSectionConfig]]) -> List[Document]:
        """
        Loop over all top-level keys in the payload.
          - Keys that appear in chunk_config are processed using the provided options.
          - Keys with dict or list values not in the config are processed with the default settings.
          - Primitive keys are aggregated into a single "Top Level" Document.
        """
        all_docs: List[Document] = []
        primitive_data = {}
        if chunk_config is None:
            chunk_config = {}

        for key, value in data.items():
            if key in chunk_config:
                user_cfg = chunk_config[key]
                cfg = {**DEFAULT_SECTION_CONFIG, **user_cfg.dict(exclude_unset=True)}
                if cfg.get("section_name") is None:
                    cfg["section_name"] = key
                all_docs.extend(self.process_section(key, value, cfg))
            elif isinstance(value, (dict, list)):
                cfg = DEFAULT_SECTION_CONFIG.copy()
                cfg["section_name"] = key
                all_docs.extend(self.process_section(key, value, cfg))
            else:
                primitive_data[key] = value

        if primitive_data:
            content_lines = []
            for key, value in primitive_data.items():
                value_str = pformat(value, indent=2) if isinstance(value, (dict, list)) else str(value)
                content_lines.append(f"{key}: {value_str}")
            content = "\n".join(content_lines)
            all_docs.insert(0, Document(page_content=f"Top Level:\n{content}", metadata={"section": "Top Level"}))

        return all_docs

    def _split_documents(self, documents: List[Any], chunk_size: int = 500, chunk_overlap: int = 0,
                         semantic_chunking=False, chunk_config=None) -> List[Any]:
        """
        Split documents using a RecursiveCharacterTextSplitter, attach chunk metadata.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        split_docs = []
        for doc_index, doc in enumerate(documents):
            if semantic_chunking:
                split_chunks = self.chunk_json(json.loads(doc.page_content), chunk_config=chunk_config)
            else:
                split_chunks = text_splitter.split_documents([doc])
            for chunk_index, chunk in enumerate(split_chunks):
                chunk.metadata = doc.metadata.copy()
                chunk.metadata["document_id"] = chunk.metadata.get("document_id", doc_index)
                chunk.metadata["chunk_id"] = chunk_index
                chunk.metadata["total_chunks"] = len(split_chunks)
                split_docs.append(chunk)
        return split_docs

    def delete_all_docs(self, namespace: str = "") -> None:
        """
        Wipe the entire namespace, removing all vectors.
        In FAISS, we'll create a new empty index.
        """
        if self.index:
            # Create a new empty index
            self.index = FaissIndexManager.create_index(self.dimension)
            # Clear metadata
            self.metadata_store = {namespace: {}} if namespace else {}
            self.id_to_index = {}
            # Save to disk
            FaissIndexManager.save_index(self.index, self._get_index_path())
            self._save_metadata()
            print(f"Deleted all docs in namespace '{namespace}'.")

    def _batch_list(self, iterator, batch_size):
        """
        Generator that yields lists of up to batch_size items from an iterator.
        """
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Yield remaining items if any
        if batch:
            yield batch

    def upsert_documents(
            self,
            documents: List[Document],
            namespace: str = "",
            do_chunking: bool = True,
            semantic_chunking=False,
            chunk_config=None,
            batch_size: int = 100,
            auto_delete: bool = True,
            chunk_size: int = 500,
            level: int = 0,
            show_progress: bool = False,
    ) -> None:
        """
        Upsert documents into the FAISS index.
        If auto_delete=True, existing vectors for the same document_id are deleted first.
        """
        if not self.index:
            self.connect()

        # 1. Gather document IDs
        document_ids = [str(doc.metadata.setdefault("document_id", str(id(doc)))) for doc in documents]

        # 2. Optionally delete existing chunks for these doc IDs
        if auto_delete:
            self.delete_with_doc_ids(document_ids, namespace=namespace)

        # 3. Possibly chunk the documents
        if do_chunking:
            texts = self._split_documents(documents, chunk_size=chunk_size, semantic_chunking=semantic_chunking,
                                          chunk_config=chunk_config)
        else:
            texts = documents

        # 4. Embed
        all_texts = [chunk.page_content for chunk in texts]
        encoded_texts = self.embeddings.embed_documents(all_texts)

        # 5. Prepare and insert vectors
        vectors = np.array(encoded_texts).astype('float32')
        if len(vectors) > 0:
            # Get current index size
            current_index_size = self.index.ntotal

            # Add vectors to FAISS index
            self.index.add(vectors)

            # Store metadata
            if namespace not in self.metadata_store:
                self.metadata_store[namespace] = {}

            for i, (chunk, vector) in enumerate(zip(texts, encoded_texts)):
                metadata = chunk.metadata.copy()
                # Remove old embedding if any
                metadata.pop("embedding", None)
                metadata.setdefault("document_id", str(uuid.uuid4()))
                metadata.setdefault("source", "")
                metadata["text"] = chunk.page_content
                metadata.setdefault("level", level)
                chunk_number = int(metadata.get("chunk_id", 0))

                # Record ID format: "doc123#0"
                record_id = f"{metadata['document_id']}#{str(chunk_number)}"
                chunk.id = record_id

                # Map the record_id to its index in FAISS
                faiss_idx = current_index_size + i
                self.id_to_index[record_id] = faiss_idx

                # Store metadata
                self.metadata_store[namespace][record_id] = {
                    "id": record_id,
                    "metadata": metadata,
                }

            # Save index and metadata
            FaissIndexManager.save_index(self.index, self._get_index_path())
            self._save_metadata()

        return texts

    def delete_with_doc_ids(self, document_ids: List[str], namespace: str = "") -> None:
        """
        Delete documents (by doc_id) from FAISS index.
        Note: FAISS doesn't support direct deletion in IndexFlatL2.
        We'll mark items as deleted in metadata and rebuild the index if needed.
        """
        if not self.index:
            self.connect()

        if namespace not in self.metadata_store:
            return

        deleted_count = 0

        # For each document ID, find all chunks and mark them as deleted
        for doc_id in document_ids:
            # Find all record IDs that start with this document ID
            to_delete = [rid for rid in self.metadata_store[namespace].keys()
                         if rid.startswith(f"{doc_id}#")]

            # Remove them from metadata
            for rid in to_delete:
                if rid in self.metadata_store[namespace]:
                    del self.metadata_store[namespace][rid]
                    deleted_count += 1
                if rid in self.id_to_index:
                    del self.id_to_index[rid]

        if deleted_count > 0:
            print(f"Marked {deleted_count} records for deletion.")
            # Save metadata
            self._save_metadata()

            # If too many deletions, consider rebuilding the index
            # In a production system, you might want to use a more sophisticated
            # approach for handling deletions in FAISS

    def delete_with_metadata(self, metadata_filter: Dict, namespace: str = "") -> None:
        """
        Delete vectors from FAISS based on a metadata filter.
        Similar to delete_with_doc_ids, we mark records as deleted.
        """
        if not self.index:
            self.connect()

        if namespace not in self.metadata_store:
            return

        deleted_count = 0

        # For each record, check if it matches the filter
        to_delete = []
        for rid, record in self.metadata_store[namespace].items():
            # Simple metadata matching
            # In a real implementation, you'd want a more sophisticated filter logic
            matches = True
            for key, value in metadata_filter.items():
                if isinstance(value, dict):
                    # Handle operators like $in
                    if "$in" in value:
                        if key not in record["metadata"] or record["metadata"][key] not in value["$in"]:
                            matches = False
                            break
                else:
                    # Simple equality check
                    if key not in record["metadata"] or record["metadata"][key] != value:
                        matches = False
                        break

            if matches:
                to_delete.append(rid)

        # Remove matching records
        for rid in to_delete:
            if rid in self.metadata_store[namespace]:
                del self.metadata_store[namespace][rid]
                deleted_count += 1
            if rid in self.id_to_index:
                del self.id_to_index[rid]

        if deleted_count > 0:
            print(f"Marked {deleted_count} records for deletion.")
            # Save metadata
            self._save_metadata()

    def query(
            self,
            queries: List[Query],
            global_filter: Optional[Dict] = None,
            namespace: Optional[str] = "",
    ) -> List[list[Document]]:
        """
        Query the FAISS index to retrieve relevant doc chunks.
        """
        if not self.index:
            self.connect()

        if namespace not in self.metadata_store:
            self.metadata_store[namespace] = {}

        query_texts = [query.text for query in queries]
        query_vectors = self.embeddings.embed_documents(query_texts)

        retrieved_documents = []

        for query_obj, vector in zip(queries, query_vectors):
            top_k = query_obj.top_k or self.default_top_k

            # Convert query vector to numpy array
            query_vector = np.array([vector]).astype('float32')

            # Query FAISS index
            distances, indices = self.index.search(query_vector, top_k * 10)  # Get more results for filtering

            # Distances is a 2D array, get the first row
            distances = distances[0]
            indices = indices[0]

            # Convert distances to scores (lower distance = higher score)
            max_dist = np.max(distances) if len(distances) > 0 else 1.0
            scores = 1.0 - (distances / max_dist) if max_dist > 0 else np.ones_like(distances)

            # Filter results based on metadata
            filtered_results = []
            for i, idx in enumerate(indices):
                if idx < 0:  # FAISS returns -1 for padded results
                    continue

                # Find the record ID for this index
                record_id = None
                for rid, fidx in self.id_to_index.items():
                    if fidx == idx:
                        record_id = rid
                        break

                if not record_id or record_id not in self.metadata_store[namespace]:
                    continue

                record = self.metadata_store[namespace][record_id]

                # Apply metadata filter if provided
                if query_obj.metadata_filter or global_filter:
                    combined_filter = {}
                    if query_obj.metadata_filter:
                        combined_filter.update(query_obj.metadata_filter)
                    if global_filter:
                        combined_filter.update(global_filter)

                    # Simple filter logic - check if all filter criteria match
                    matches = True
                    for key, value in combined_filter.items():
                        if isinstance(value, dict):
                            # Handle operators like $in
                            if "$in" in value:
                                if key not in record["metadata"] or record["metadata"][key] not in value["$in"]:
                                    matches = False
                                    break
                        else:
                            # Simple equality check
                            if key not in record["metadata"] or record["metadata"][key] != value:
                                matches = False
                                break

                    if not matches:
                        continue

                # Add to filtered results
                filtered_results.append((record, scores[i]))

                # Break if we have enough results
                if len(filtered_results) >= top_k:
                    break

            # Convert to Document objects
            one_query_result = []
            for record, score in filtered_results:
                doc = Document(
                    page_content=record["metadata"].get("text", ""),
                    metadata={
                        "document_id": record["metadata"].get("document_id", ""),
                        "source": record["metadata"].get("source", ""),
                        "score": float(score),
                        "parent_id": record["metadata"].get("parent_id", "")
                    },
                )
                one_query_result.append(doc)

            retrieved_documents.append(one_query_result)

        return retrieved_documents

    def query_by_metadata(
            self,
            metadata_filter: Optional[Dict] = None,
            namespace: Optional[str] = "",
    ) -> List[Document]:
        """
        Query the FAISS index to retrieve doc chunks by metadata alone.
        """
        if not self.index:
            self.connect()

        if namespace not in self.metadata_store:
            return []

        retrieved_documents = []

        # Filter records by metadata
        for rid, record in self.metadata_store[namespace].items():
            # Skip if no filter provided
            if not metadata_filter:
                continue

            # Simple metadata matching
            matches = True
            for key, value in metadata_filter.items():
                if isinstance(value, dict):
                    # Handle operators like $in
                    if "$in" in value:
                        if key not in record["metadata"] or record["metadata"][key] not in value["$in"]:
                            matches = False
                            break
                else:
                    # Simple equality check
                    if key not in record["metadata"] or record["metadata"][key] != value:
                        matches = False
                        break

            if matches:
                # Get the vector for this record
                idx = self.id_to_index.get(rid)

                # Create Document object
                doc = Document(
                    page_content=record["metadata"].get("text", ""),
                    id=record["id"],
                    metadata={
                        "document_id": record["metadata"].get("document_id", ""),
                        "source": record["metadata"].get("source", ""),
                        "score": 1.0,  # Default score for metadata-only queries
                        **record["metadata"],
                    },
                )
                retrieved_documents.append(doc)

        return retrieved_documents

    @classmethod
    def from_config(cls, config: Dict[str, Any], index_name: Optional[str] = None) -> "VectorStoreManager":
        """
        Initialize VectorStoreManager from a configuration dictionary.
        """
        index_name = index_name or os.getenv("INDEX_NAME")
        if not index_name:
            raise ValueError("Index name must be provided.")

        if config.get("params", {}).get("index_name", index_name) != index_name:
            raise ValueError("Index name in config does not match the provided index name.")

        return cls(
            index_name=index_name,
        )

    # Placeholder async methods
    async def aquery(self, queries: List[Query], global_filter: Optional[Dict] = None,
                     namespace: Optional[str] = None) -> List[Document]:
        raise NotImplementedError("Asynchronous queries are not implemented yet.")

    async def aupsert(self, documents: List[Document], namespace: str = "") -> None:
        raise NotImplementedError("Asynchronous upsert is not implemented yet.")

    async def adelete(self, document_ids: List[str], namespace: str = "") -> None:
        raise NotImplementedError("Asynchronous delete is not implemented yet.")