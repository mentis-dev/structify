import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, List, Optional, Dict

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

@dataclass(kw_only=True)
class InputState:
    """Initial state passed in by the user."""
    workspace_id: str  # ID of the workspace to process
    max_documents: Optional[int] = None
    max_brains: int = 2
    output_dir: str = "output"
    extraction_schema: Optional[dict] = None  # Schema for stakeholder extraction

@dataclass(kw_only=True)
class State(InputState):
    """Internal state of the agent."""
    messages: Annotated[List[BaseMessage], add_messages] = field(default_factory=list)
    workspace_data: Optional[Dict[str, Any]] = None
    document_ids: List[str] = field(default_factory=list)
    current_document_id: Optional[str] = None
    current_document_content: Optional[str] = None
    processed_documents: Dict[str, Any] = field(default_factory=dict)
    aggregated_stakeholders: Optional[Dict[str, Any]] = None
    info: Optional[Dict[str, Any]] = None  # Extracted information
    loop_step: Annotated[int, operator.add] = field(default=0)
    error: Optional[str] = None

# This is defined as a standard dict to match the example usage
OutputState = Dict[str, Any]  # Will contain the final info
