from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Optional

from langchain_core.runnables import RunnableConfig, ensure_config

@dataclass(kw_only=True)
class Configuration:
    """The configuration for the extraction agent."""

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="openai/gpt-4o",
        metadata={
            "description": "The name of the language model to use. Should be in the form: provider/model-name."
        },
    )
    prompt: str = field(
        default="""You are an expert stakeholder analyst tasked with identifying stakeholders from the provided text using the extraction schema.

For each stakeholder, classify them into one of these categories:
- Regulator: Government or oversight bodies that create and enforce rules
- Supplier: Provides products or services to the organization or industry
- Consumer: Receives, uses, or benefits from products or services
- Competitor: Other organizations providing similar services or competing for resources
- Partner: Organizations working together with shared goals
- Influencer: Shapes opinions or decisions without direct authority
- Internal: Employees, management, or departments within the organization

Assign a confidence score (0-100%) to each stakeholder's classification based on certainty.

Schema:
{info}

Text:
{topic}

Identify the most relevant stakeholders with the most impact, who are most well-known, or most used to make decisions.
Provide your answer as properly formatted JSON matching the schema exactly.""",
        metadata={
            "description": "The main prompt template. Expects two arguments: {info} and {topic}."
        },
    )
    max_loops: int = field(
        default=100,
        metadata={
            "description": "The maximum number of interaction loops before termination."
        },
    )
    processing_delay: float = field(
        default=0.5,
        metadata={
            "description": "Delay in seconds between processing documents to avoid rate limiting."
        },
    )
    temperature: float = field(
        default=0.1,
        metadata={
            "description": "Temperature for model response generation. Lower values (0.0-0.3) for more deterministic responses, higher values (0.7-1.0) for more creative responses."
        },
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> Configuration:
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
