from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List



class Query(BaseModel):
    text: str
    top_k: Optional[int] = None
    metadata_filter: Optional[Dict[str, Any]] = None



class ChunkSectionConfig(BaseModel):
    section_name: Optional[str] = Field(
        None,
        description="Optional name to use for this section; if not provided, the key will be used."
    )
    fields: Optional[List[str]] = Field(
        None,
        description="List of keys to include in the content for this section. Default is all keys."
    )
    combine_list: Optional[bool] = Field(
        False,
        description="If True, combine list items into one chunk; otherwise, create separate chunks for each item."
    )
    metadata_keys: Optional[List[str]] = Field(
        [],
        description="List of keys to extract as metadata for this section."
    )
    content_exclude_keys: Optional[List[str]] = Field(
        [],
        description="List of keys to exclude from the content if they are already used in metadata."
    )
