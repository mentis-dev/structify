from typing import List, Dict, Any, Optional, Union
import os

# Import from your existing Supabase code
from core.supabase_db import *


class SupabaseService:
    """Service layer for Supabase operations"""

    def __init__(self, exclude_playgrounds: bool = True):
        """
        Initialize Supabase client

        Args:
            exclude_playgrounds: Whether to exclude workspaces named 'playground'
        """
        self.supabase = initialize_supabase()
        self.exclude_playgrounds = exclude_playgrounds

    def get_workspaces(self) -> List[Dict[str, Any]]:
        """Get all workspaces, optionally filtering out playgrounds"""
        workspaces = get_workspaces(self.supabase)

        # Format for consistent keys
        formatted_workspaces = []
        for workspace in workspaces:
            # Skip workspaces named 'playground' if exclude_playgrounds is True
            workspace_name = workspace.get("name", "").lower()
            if self.exclude_playgrounds and "playground" in workspace_name:
                continue

            formatted_workspace = {
                "workspace_id": workspace.get("workspace_id"),
                "name": workspace.get("name"),
                "created_at": workspace.get("created_at"),
                "updated_at": workspace.get("updated_at")
            }
            formatted_workspaces.append(formatted_workspace)

        return formatted_workspaces

    def get_brains_per_workspace(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get all brains for a workspace"""
        brains = get_brains_per_workspace(self.supabase, workspace_id)

        # Format for consistent keys
        formatted_brains = []
        for brain in brains:
            formatted_brain = {
                "brain_id": brain.get("brain_id") or brain.get("id"),
                "name": brain.get("name"),
                "description": brain.get("description"),
                "status": brain.get("status")
            }
            formatted_brains.append(formatted_brain)

        return formatted_brains

    def get_documents_per_brain(self, brain_id: str) -> List[Dict[str, Any]]:
        """Get all documents for a brain"""
        documents = get_documents_per_brain(self.supabase, brain_id)

        # Format for consistent keys
        formatted_documents = []
        for doc in documents:
            formatted_doc = {
                "id": doc.get("id"),
                "file_name": doc.get("file_name"),
                "brain_id": doc.get("brain_id"),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at")
            }
            formatted_documents.append(formatted_doc)

        return formatted_documents

    def get_document_content(self, document_id: str, return_chunks: bool = False) -> Union[str, List[str], None]:
        """
        Get content for a document

        Args:
            document_id: ID of the document
            return_chunks: If True, returns a list of chunks instead of combined text

        Returns:
            If return_chunks=False: Combined document content as string
            If return_chunks=True: List of document chunks
            None if there was an error
        """
        try:
            content = get_document_data(self.supabase, document_id, return_chunks=return_chunks)
            return content
        except Exception as e:
            print(f"Error getting document content: {e}")
            return None

    def get_document_contents(self, document_ids: List[str], return_chunks: bool = False) -> Dict[str, Any]:
        """
        Get content for multiple documents

        Args:
            document_ids: List of document IDs
            return_chunks: If True, returns chunks instead of combined text

        Returns:
            Dictionary mapping document IDs to their content (either string or list of chunks)
        """
        contents = {}
        for doc_id in document_ids:
            content = self.get_document_content(doc_id, return_chunks=return_chunks)
            if content:
                contents[doc_id] = content
        return contents

    def get_vectors_by_knowledge_ids(self, knowledge_ids: List[str]) -> Dict[str, str]:
        """Get vectors for knowledge IDs"""
        try:
            return get_vectors_by_knowledge_ids(self.supabase, knowledge_ids)
        except Exception as e:
            print(f"Error getting vectors: {e}")
            return {}