import json
import os
import datetime
from typing import Any, Dict, List, Optional, cast
from difflib import SequenceMatcher

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from openai import AsyncOpenAI

from enrichment.configuration import Configuration
from enrichment.state import InputState, OutputState, State
from enrichment.utils import init_model
from select_data import initialize_supabase, get_brains_per_workspace, get_documents_per_brain
from supabase_db import get_vectors_by_knowledge_ids

# Default stakeholder extraction schema
DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "stakeholders": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the stakeholder organization or individual."
                    },
                    "category": {
                        "type": "string",
                        "description": "The category of the stakeholder (Regulator, Supplier, Consumer, Competitor, Partner, Influencer, Internal)."
                    },
                    "role": {
                        "type": "string",
                        "description": "Brief description of the stakeholder's role or function."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score (0-100%) for the classification."
                    }
                },
                "required": ["name", "category", "role", "confidence"]
            },
            "description": "List of stakeholders identified in the text."
        },
        "factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Label": {
                        "type": "string",
                        "description": "The descriptive name for the factor."
                    },
                    "Type": {
                        "type": "string",
                        "description": "Classification type (e.g., Indirect Driver, Exogenous Driver, Outcome, Direct Driver)."
                    },
                    "Tags": {
                        "type": "string",
                        "description": "Any additional tags or keywords (optional)."
                    },
                    "Description": {
                        "type": "string",
                        "description": "A detailed description of the factor including stakeholder information."
                    }
                },
                "required": ["Label", "Type", "Description"]
            },
            "description": "List of factors identified in the text."
        },
        "pain_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "A unique identifier for the pain point (e.g., PP-001)."
                    },
                    "category": {
                        "type": "string",
                        "description": "The category of the pain point (e.g., Healthcare Access, Financial Barriers)."
                    },
                    "description": {
                        "type": "string",
                        "description": "A brief description of the pain point."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score (0-100%) for this pain point identification."
                    }
                },
                "required": ["id", "category", "description", "confidence"]
            },
            "description": "List of pain points identified in the text."
        }
    },
    "required": ["stakeholders", "factors", "pain_points"]
}

async def initialize_workspace(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Initialize the workspace and retrieve brains and documents.
    """
    try:
        # Create output directory
        os.makedirs(state.output_dir, exist_ok=True)
        
        # Initialize Supabase client
        supabase = initialize_supabase()
        
        # Get brains from workspace
        all_brains = get_brains_per_workspace(supabase, state.workspace_id)
        
        if not all_brains:
            return {
                "error": f"No brains found for workspace: {state.workspace_id}",
                "messages": [HumanMessage(content=f"No brains found for workspace: {state.workspace_id}")]
            }
        
        # Limit to max_brains if specified
        brains = all_brains
        if state.max_brains is not None:
            brains = all_brains[:state.max_brains]
        
        # Get all documents from each brain
        result = {}
        all_document_ids = []
        
        for brain in brains:
            brain_id = brain['brain_id']
            brain_name = brain.get('name', 'Unnamed Brain')
            
            documents = get_documents_per_brain(supabase, brain_id)
            
            if not documents:
                result[brain_id] = {"name": brain_name, "documents": []}
                continue
            
            # Extract document IDs
            document_ids = [doc['id'] for doc in documents]
            result[brain_id] = {
                'name': brain_name,
                'documents': documents
            }
            
            all_document_ids.extend(document_ids)
        
        # Limit number of documents if specified
        if state.max_documents is not None and state.max_documents < len(all_document_ids):
            all_document_ids = all_document_ids[:state.max_documents]
        
        workspace_data = {
            'workspace_id': state.workspace_id,
            'brains': result,
            'all_document_ids': all_document_ids
        }
        
        # Save workspace structure
        structure_file = os.path.join(state.output_dir, f"workspace_{state.workspace_id}_structure.json")
        with open(structure_file, 'w', encoding='utf-8') as f:
            json.dump(workspace_data, f, indent=2)
        
        # Set extraction schema if not provided
        extraction_schema = state.extraction_schema or DEFAULT_SCHEMA
        
        return {
            "workspace_data": workspace_data,
            "document_ids": all_document_ids,
            "extraction_schema": extraction_schema,
            "messages": [HumanMessage(content=f"Initialized workspace {state.workspace_id} with {len(all_document_ids)} documents from {len(brains)} brains.")]
        }
    
    except Exception as e:
        return {
            "error": f"Error initializing workspace: {str(e)}",
            "messages": [HumanMessage(content=f"Error initializing workspace: {str(e)}")]
        }

async def get_next_document(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Get the next document to process by retrieving its content.
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Increment loop counter
    loop_step = state.loop_step + 1
    
    # Check if we have more documents to process
    if not state.document_ids:
        return {
            "loop_step": loop_step,
            "messages": [HumanMessage(content="No more documents to process. Moving to aggregation.")]
        }
    
    try:
        # Get the next document ID
        doc_id = state.document_ids[0]
        remaining_docs = state.document_ids[1:]
        
        # Initialize Supabase client
        supabase = initialize_supabase()
        
        # Get vectors for the document
        vectors_dict = get_vectors_by_knowledge_ids(supabase, [doc_id])
        
        if not vectors_dict or doc_id not in vectors_dict:
            return {
                "document_ids": remaining_docs,
                "loop_step": loop_step,
                "messages": [HumanMessage(content=f"No content found for document {doc_id}. Skipping.")]
            }
        
        # Find document info
        doc_info = None
        brain_info = None
        
        for brain_id, brain_data in state.workspace_data['brains'].items():
            for doc in brain_data['documents']:
                if doc['id'] == doc_id:
                    doc_info = doc
                    brain_info = {
                        'brain_id': brain_id,
                        'brain_name': brain_data['name']
                    }
                    break
            if doc_info:
                break
        
        doc_name = doc_info.get('file_name', 'Unknown') if doc_info else 'Unknown'
        brain_name = brain_info.get('brain_name', 'Unknown') if brain_info else 'Unknown'
        
        return {
            "current_document_id": doc_id,
            "current_document_content": vectors_dict[doc_id],
            "document_ids": remaining_docs,
            "loop_step": loop_step,
            "messages": [HumanMessage(content=f"Processing document {doc_name} from brain {brain_name}.")]
        }
    
    except Exception as e:
        return {
            "error": f"Error getting next document: {str(e)}",
            "loop_step": loop_step,
            "messages": [HumanMessage(content=f"Error getting next document: {str(e)}")]
        }

async def extract_stakeholders(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Extract and classify stakeholders, factors, and pain points from the current document using the LLM.
    """
    if not state.current_document_id or not state.current_document_content:
        return {
            "messages": [HumanMessage(content="No current document to process.")]
        }
    
    try:
        configuration = Configuration.from_runnable_config(config)
        
        # Format the prompt
        prompt_text = configuration.prompt.format(
            info=json.dumps(state.extraction_schema, indent=2),
            topic=state.current_document_content
        )
        
        # Initialize and call the model
        model = init_model(config)
        messages = [HumanMessage(content=prompt_text)]
        response = cast(AIMessage, await model.ainvoke(messages))
        
        # Try to parse the response as JSON
        try:
            # First attempt to extract JSON if it's wrapped in markdown code blocks
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            result_json = json.loads(content)
            
            # Find document info
            doc_info = None
            brain_info = None
            
            for brain_id, brain_data in state.workspace_data['brains'].items():
                for doc in brain_data['documents']:
                    if doc['id'] == state.current_document_id:
                        doc_info = doc
                        brain_info = {
                            'brain_id': brain_id,
                            'brain_name': brain_data['name']
                        }
                        break
                if doc_info:
                    break
            
            # Prepare document result
            doc_result = {
                'brain_info': brain_info,
                'document_info': {
                    'id': state.current_document_id,
                    'file_name': doc_info.get('file_name', 'Unknown') if doc_info else 'Unknown',
                    'content_length': len(state.current_document_content)
                },
                'extracted_stakeholders': result_json.get('stakeholders', [])
            }
            
            # Store factors separately
            factors = result_json.get('factors', [])
            if factors:
                # Save factors to the document result
                doc_result['extracted_factors'] = factors
            
            # Store pain points separately
            pain_points = result_json.get('pain_points', [])
            if pain_points:
                # Save pain points to the document result
                doc_result['extracted_pain_points'] = pain_points
            
            # Save individual result
            doc_file = os.path.join(state.output_dir, f"doc_{state.current_document_id}_extraction.json")
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc_result, f, indent=2)
            
            # Update processed documents
            processed_documents = state.processed_documents.copy()
            processed_documents[state.current_document_id] = doc_result
            
            stakeholder_count = len(result_json.get('stakeholders', []))
            factor_count = len(factors)
            pain_point_count = len(pain_points)
            doc_name = doc_info.get('file_name', state.current_document_id) if doc_info else state.current_document_id
            
            return {
                "processed_documents": processed_documents,
                "current_document_id": None,
                "current_document_content": None,
                "info": result_json,  # Store the most recent extraction result
                "messages": [AIMessage(content=f"Extracted {stakeholder_count} stakeholders, {factor_count} factors, and {pain_point_count} pain points from document {doc_name}.")]
            }
            
        except json.JSONDecodeError:
            return {
                "error": f"Error parsing JSON response for document {state.current_document_id}",
                "current_document_id": None,
                "current_document_content": None,
                "messages": [HumanMessage(content=f"Error parsing response for document {state.current_document_id}.")]
            }
        
    except Exception as e:
        return {
            "error": f"Error extracting information: {str(e)}",
            "current_document_id": None,
            "current_document_content": None,
            "messages": [HumanMessage(content=f"Error extracting information: {str(e)}")]
        }


async def aggregate_stakeholders(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Aggregate stakeholder classifications, factors, and pain points across all processed documents.
    """
    if not state.processed_documents:
        return {
            "error": "No processed documents available for aggregation.",
            "messages": [HumanMessage(content="No processed documents available for aggregation.")]
        }
    
    try:
        # Create structure to store all stakeholders
        stakeholder_data = {}
        
        # Collect all stakeholders from all documents
        for doc_id, doc_result in state.processed_documents.items():
            # Get the extracted stakeholders
            extracted = doc_result.get('extracted_stakeholders', [])
            if not extracted or not isinstance(extracted, list):
                continue
            
            doc_name = doc_result.get('document_info', {}).get('file_name', 'Unknown')
            brain_info = doc_result.get('brain_info', {})
            brain_name = brain_info.get('brain_name', 'Unknown') if brain_info else 'Unknown'
            
            for stakeholder in extracted:
                if not isinstance(stakeholder, dict):
                    continue
                    
                name = stakeholder.get('name')
                category = stakeholder.get('category')
                role = stakeholder.get('role')
                confidence = stakeholder.get('confidence')
                hierarchy_level = stakeholder.get('hierarchy_level', 'Meso')  # Default to Meso if not provided
                
                if not name:
                    continue
                    
                if name not in stakeholder_data:
                    stakeholder_data[name] = {
                        'categories': {},
                        'roles': {},
                        'hierarchy_levels': {},
                        'documents': [],
                        'total_mentions': 0
                    }
                
                # Add document reference
                stakeholder_data[name]['documents'].append({
                    'id': doc_id,
                    'name': doc_name,
                    'brain': brain_name
                })
                
                # Increment total mentions
                stakeholder_data[name]['total_mentions'] += 1
                
                # Add category with confidence
                if category:
                    if category not in stakeholder_data[name]['categories']:
                        stakeholder_data[name]['categories'][category] = {
                            'count': 0,
                            'confidence_sum': 0
                        }
                    stakeholder_data[name]['categories'][category]['count'] += 1
                    stakeholder_data[name]['categories'][category]['confidence_sum'] += confidence if confidence else 0
                
                # Add role
                if role:
                    if role not in stakeholder_data[name]['roles']:
                        stakeholder_data[name]['roles'][role] = 0
                    stakeholder_data[name]['roles'][role] += 1
                
                # Add hierarchy level
                if hierarchy_level:
                    if hierarchy_level not in stakeholder_data[name]['hierarchy_levels']:
                        stakeholder_data[name]['hierarchy_levels'][hierarchy_level] = 0
                    stakeholder_data[name]['hierarchy_levels'][hierarchy_level] += 1
        
        # Determine the most common category and role for each stakeholder
        aggregated_stakeholders = []
        
        for name, data in stakeholder_data.items():
            # Find most common category
            most_common_category = None
            highest_count = 0
            avg_confidence = 0
            
            for category, cat_data in data['categories'].items():
                if cat_data['count'] > highest_count:
                    highest_count = cat_data['count']
                    most_common_category = category
                    avg_confidence = cat_data['confidence_sum'] / cat_data['count'] if cat_data['count'] > 0 else 0
            
            # Find most common role
            most_common_role = None
            highest_role_count = 0
            
            for role, count in data['roles'].items():
                if count > highest_role_count:
                    highest_role_count = count
                    most_common_role = role
            
            # Find most common hierarchy level
            most_common_hierarchy = "Meso"  # Default
            highest_hierarchy_count = 0
            
            for level, count in data['hierarchy_levels'].items():
                if count > highest_hierarchy_count:
                    highest_hierarchy_count = count
                    most_common_hierarchy = level
            
            # Add aggregated stakeholder entry
            aggregated_stakeholders.append({
                'name': name,
                'category': most_common_category,
                'role': most_common_role,
                'hierarchy_level': most_common_hierarchy,
                'confidence': round(avg_confidence),
                'mentions': data['total_mentions'],
                'documents': data['documents']
            })
        
        # Sort by mentions (descending)
        aggregated_stakeholders.sort(key=lambda x: x.get('mentions', 0), reverse=True)
        
        # Create final aggregated result for stakeholders
        aggregated_result = {
            'workspace_id': state.workspace_id,
            'total_documents': len(state.processed_documents),
            'stakeholders': aggregated_stakeholders
        }
        
        # Save aggregated stakeholders
        aggregated_file = os.path.join(state.output_dir, f"workspace_{state.workspace_id}_aggregated_stakeholders.json")
        with open(aggregated_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated_result, f, indent=2)
        
        # Aggregate factors across all documents
        all_factors = []
        for doc_id, doc_result in state.processed_documents.items():
            if 'extracted_factors' in doc_result:
                factors = doc_result['extracted_factors']
                for factor in factors:
                    # Check if this is a new unique factor or already exists
                    is_new = True
                    for existing in all_factors:
                        if existing.get('Label') == factor.get('Label'):
                            is_new = False
                            break
                    
                    if is_new:
                        # Add document reference
                        doc_info = doc_result.get('document_info', {})
                        brain_info = doc_result.get('brain_info', {})
                        
                        factor['source_document'] = {
                            'id': doc_id,
                            'name': doc_info.get('file_name', 'Unknown'),
                            'brain': brain_info.get('brain_name', 'Unknown') if brain_info else 'Unknown'
                        }
                        
                        all_factors.append(factor)
        
        # Aggregate pain points across all documents
        all_pain_points = []
        pain_point_ids = set()  # Keep track of existing IDs
        id_counter = 1
        
        for doc_id, doc_result in state.processed_documents.items():
            if 'extracted_pain_points' in doc_result:
                pain_points = doc_result['extracted_pain_points']
                for pain_point in pain_points:
                    # Ensure pain point has a valid ID or generate one
                    if not pain_point.get('id') or pain_point.get('id') in pain_point_ids:
                        new_id = f"PP-{id_counter:03d}"
                        while new_id in pain_point_ids:
                            id_counter += 1
                            new_id = f"PP-{id_counter:03d}"
                        pain_point['id'] = new_id
                        id_counter += 1
                    
                    pain_point_ids.add(pain_point['id'])
                    
                    # Add document reference
                    doc_info = doc_result.get('document_info', {})
                    brain_info = doc_result.get('brain_info', {})
                    
                    pain_point['source_document'] = {
                        'id': doc_id,
                        'name': doc_info.get('file_name', 'Unknown'),
                        'brain': brain_info.get('brain_name', 'Unknown') if brain_info else 'Unknown'
                    }
                    
                    # Assign hierarchy level based on content
                    if not pain_point.get('hierarchy_level'):
                        description = pain_point.get('description', '').lower()
                        if any(term in description for term in ['system', 'policy', 'government', 'budget', 'national']):
                            pain_point['hierarchy_level'] = 'Macro'
                        elif any(term in description for term in ['local', 'individual', 'resident', 'community']):
                            pain_point['hierarchy_level'] = 'Micro'
                        else:
                            pain_point['hierarchy_level'] = 'Meso'
                    
                    all_pain_points.append(pain_point)
        
        # Include all data in the final info
        info = {
            'stakeholders': aggregated_stakeholders,
            'factors': all_factors,
            'pain_points': all_pain_points
        }
        
        # Save combined results
        combined_result = {
            'workspace_id': state.workspace_id,
            'total_documents': len(state.processed_documents),
            'stakeholders': aggregated_stakeholders,
            'factors': all_factors,
            'pain_points': all_pain_points
        }
        
        # Save the combined results
        combined_file = os.path.join(state.output_dir, f"workspace_{state.workspace_id}_analysis.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(combined_result, f, indent=2)

        # Add relationships between hierarchy levels
        hierarchy_relationships = []
        relationship_id = 1

        # Create relationships between macro and meso levels
        for macro in [s for s in aggregated_stakeholders if s.get('hierarchy_level') == 'Macro']:
            for meso in [s for s in aggregated_stakeholders if s.get('hierarchy_level') == 'Meso']:
                # Analyze text to determine relationship type
                relationship_type = "Regulation" if macro.get('category') == "Regulator" else "Funding"
                
                hierarchy_relationships.append({
                    "id": f"R{relationship_id}",
                    "source": macro.get('name'),
                    "source_level": "macro",
                    "target": meso.get('name'),
                    "target_level": "meso",
                    "type": relationship_type,
                    "strength": 0.85  # Default strength
                })
                relationship_id += 1

        # Create relationships between meso and micro levels
        for meso in [s for s in aggregated_stakeholders if s.get('hierarchy_level') == 'Meso']:
            for micro in [s for s in aggregated_stakeholders if s.get('hierarchy_level') == 'Micro']:
                # Determine relationship type based on categories
                relationship_type = "Service" if meso.get('category') == "Supplier" else "Support"
                
                hierarchy_relationships.append({
                    "id": f"R{relationship_id}",
                    "source": meso.get('name'),
                    "source_level": "meso",
                    "target": micro.get('name'),
                    "target_level": "micro", 
                    "type": relationship_type,
                    "strength": 0.75  # Default strength
                })
                relationship_id += 1

        # Add to the combined result
        combined_result["relationships"] = hierarchy_relationships

        # Create multi-level ecosystem representation
        multi_level_ecosystem = {
            "ecosystemMap": {
                "name": f"Ecosystem Analysis for Workspace {state.workspace_id}",
                "levels": {
                    "macro": {
                        "nodes": [s for s in aggregated_stakeholders if s.get('hierarchy_level') == 'Macro'],
                        "pain_points": [p for p in all_pain_points if p.get('hierarchy_level') == 'Macro']
                    },
                    "meso": {
                        "nodes": [s for s in aggregated_stakeholders if s.get('hierarchy_level') == 'Meso'],
                        "pain_points": [p for p in all_pain_points if p.get('hierarchy_level') == 'Meso']
                    },
                    "micro": {
                        "nodes": [s for s in aggregated_stakeholders if s.get('hierarchy_level') == 'Micro'],
                        "pain_points": [p for p in all_pain_points if p.get('hierarchy_level') == 'Micro']
                    }
                },
                "relationships": hierarchy_relationships
            }
        }

        # Save the multi-level ecosystem representation
        ecosystem_file = os.path.join(state.output_dir, f"workspace_{state.workspace_id}_ecosystem_map.json")
        with open(ecosystem_file, 'w', encoding='utf-8') as f:
            json.dump(multi_level_ecosystem, f, indent=2)

        # Add to the final info
        info["multi_level_ecosystem"] = multi_level_ecosystem
        
        return {
            "aggregated_stakeholders": aggregated_result,
            "factors": all_factors,
            "pain_points": all_pain_points,
            "info": info,  # Final output that will be in OutputState
            "messages": [AIMessage(content=f"Aggregated {len(aggregated_stakeholders)} stakeholders, {len(all_factors)} factors, and {len(all_pain_points)} pain points from {len(state.processed_documents)} documents. Created multi-level ecosystem map.")]
        }
    
    except Exception as e:
        return {
            "error": f"Error aggregating results: {str(e)}",
            "messages": [HumanMessage(content=f"Error aggregating results: {str(e)}")]
        }
        

async def validate_stakeholders(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Validate stakeholders using LLM but with enhanced error handling.
    Uses structured, simple interactions with robust cleaning of LLM responses.
    """
    if state.error:
        return {
            "error": state.error,
            "messages": [HumanMessage(content=f"Error in previous step: {state.error}")]
        }
        
    if not state.aggregated_stakeholders:
        return {
            "error": "No aggregated stakeholders available for validation",
            "messages": [HumanMessage(content="No stakeholders to validate. Skipping validation step.")]
        }
    
    try:
        from difflib import SequenceMatcher
        import re
        import json
        
        # Helper function to calculate string similarity
        def similarity_score(a, b):
            """Calculate string similarity score between 0 and 1"""
            return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
        
        print("Starting stakeholder validation with LLM assistance...")
        
        # Extract stakeholders from aggregated results
        stakeholders = state.aggregated_stakeholders.get('stakeholders', [])
        total_stakeholders = len(stakeholders)
        
        if not stakeholders:
            return {
                "messages": [HumanMessage(content="No stakeholders found to validate.")]
            }
        
        # Step 1: Find potential duplicates based on name similarity
        print("Finding potential duplicate stakeholders...")
        duplicate_groups = []
        processed = set()
        
        # First pass: Group stakeholders by exact name match (case-insensitive)
        name_groups = {}
        
        for i, stakeholder in enumerate(stakeholders):
            name = stakeholder.get('name', '').lower().strip()
            if not name:
                continue
                
            if name not in name_groups:
                name_groups[name] = []
            name_groups[name].append(i)
        
        # Collect exact duplicates
        for name, indices in name_groups.items():
            if len(indices) > 1:
                # These are exact matches (case-insensitive)
                duplicate_groups.append([stakeholders[i] for i in indices])
                processed.update(indices)
        
        # Second pass: Find similar names
        for i, stakeholder1 in enumerate(stakeholders):
            if i in processed:
                continue
                
            name1 = stakeholder1.get('name', '').lower().strip()
            if not name1:
                continue
                
            similar_stakeholders = [i]
            
            for j, stakeholder2 in enumerate(stakeholders):
                if i == j or j in processed:
                    continue
                    
                name2 = stakeholder2.get('name', '').lower().strip()
                if not name2:
                    continue
                
                # Calculate similarity or check containment
                similarity = similarity_score(name1, name2)
                contained = name1 in name2 or name2 in name1
                
                if similarity > 0.8 or contained:
                    similar_stakeholders.append(j)
            
            if len(similar_stakeholders) > 1:
                duplicate_groups.append([stakeholders[idx] for idx in similar_stakeholders])
                processed.update(similar_stakeholders)
        
        print(f"Found {len(duplicate_groups)} potential duplicate groups")
        
        # Step 2: Process duplicates with LLM in a robust way
        model = init_model(config)
        resolved_stakeholders = []
        handled_duplicates = set()
        duplicates_merged = 0
        
        for group_idx, group in enumerate(duplicate_groups):
            try:
                # Display group for debugging
                group_str = "\n".join([f"- {s.get('name', '')} ({s.get('category', 'Unknown')}) - {s.get('role', 'Unknown role')}" for s in group])
                print(f"\nDuplicate group {group_idx+1}:\n{group_str}")
                
                # Use a simple yes/no prompt format that's much more robust
                prompt = f"""As a stakeholder analyst, I need to decide if these are duplicate stakeholders that refer to the same entity:

{group_str}

Based ONLY on the names, categories, and roles, are these the same entity or different entities?
Answer with ONLY 'YES' if they are the same entity or 'NO' if they are different entities.
"""
                
                # Call the LLM with simple yes/no prompt
                messages = [HumanMessage(content=prompt)]
                response = await model.ainvoke(messages)
                
                # Strip and sanitize LLM output before processing (added as requested)
                response_content = response.content.strip().upper()
                response_content = response_content.replace('"', '').replace("'", "").strip()
                
                # First try to parse as JSON if it looks like JSON (added as requested)
                are_duplicates = False
                if '{' in response_content and '}' in response_content:
                    try:
                        # Try to extract JSON
                        json_str = response_content[response_content.find('{'):response_content.rfind('}')+1]
                        parsed = json.loads(json_str)
                        are_duplicates_value = str(parsed.get("are_duplicates", "")).upper()
                        are_duplicates = are_duplicates_value == "YES" or are_duplicates_value == "TRUE"
                    except Exception:
                        # Fall back to simple text matching
                        are_duplicates = 'YES' in response_content
                else:
                    # Simple text matching
                    are_duplicates = 'YES' in response_content
                
                if are_duplicates:
                    # If they are duplicates, we now need to choose the best representative
                    # Let's use a simple scoring approach instead of asking the LLM again
                    best_stakeholder = None
                    highest_score = -1
                    
                    for stakeholder in group:
                        score = 0
                        
                        # More mentions = higher score
                        score += stakeholder.get('mentions', 0) * 2
                        
                        # Higher confidence = higher score
                        score += stakeholder.get('confidence', 0) / 10
                        
                        # Prefer longer names (usually more specific)
                        name = stakeholder.get('name', '')
                        score += len(name) / 10
                        
                        # Prefer proper capitalization
                        if name and name[0].isupper():
                            score += 2
                        
                        # Avoid acronyms in parentheses
                        if re.search(r'\([A-Z]+\)', name):
                            score -= 5
                        
                        if score > highest_score:
                            highest_score = score
                            best_stakeholder = stakeholder
                    
                    if not best_stakeholder:
                        # Fallback to first stakeholder if scoring failed
                        best_stakeholder = group[0]
                    
                    # Create consolidated stakeholder
                    consolidated = best_stakeholder.copy()
                    
                    # Track the names that were consolidated
                    consolidated_names = [s.get('name', '') for s in group]
                    print(f"Consolidated: {', '.join(consolidated_names)} → {consolidated.get('name', '')}")
                    
                    # Sum mentions
                    consolidated['mentions'] = sum(s.get('mentions', 0) for s in group)
                    
                    # Take max confidence
                    consolidated['confidence'] = max(s.get('confidence', 0) for s in group)
                    
                    # Combine documents
                    docs = []
                    for s in group:
                        stakeholder_docs = s.get('documents', [])
                        if stakeholder_docs:
                            docs.extend(stakeholder_docs)
                    consolidated['documents'] = docs
                    
                    # Add metadata about the consolidation
                    consolidated['original_names'] = consolidated_names
                    
                    # Add to results
                    resolved_stakeholders.append(consolidated)
                    
                    # Add names to handled set
                    handled_duplicates.update([s.get('name', '').lower().strip() for s in group])
                    
                    # Count merged duplicates
                    duplicates_merged += len(group) - 1
                else:
                    # If not duplicates, keep all original stakeholders
                    print(f"Not duplicates, keeping separate")
                    for stakeholder in group:
                        name = stakeholder.get('name', '').lower().strip()
                        if name and name not in handled_duplicates:
                            resolved_stakeholders.append(stakeholder)
                            handled_duplicates.add(name)
            
            except Exception as e:
                # Log the raw response content on errors (added as requested)
                print(f"Error processing duplicate group: {str(e)}\nRaw response: {response.content}")
                # Keep originals if error
                for stakeholder in group:
                    name = stakeholder.get('name', '').lower().strip()
                    if name and name not in handled_duplicates:
                        resolved_stakeholders.append(stakeholder)
                        handled_duplicates.add(name)
        
        # Add stakeholders that weren't in any duplicate group
        for stakeholder in stakeholders:
            name = stakeholder.get('name', '').lower().strip()
            if name and name not in handled_duplicates:
                resolved_stakeholders.append(stakeholder)
                handled_duplicates.add(name)
        
        # Step 3: Validate categories with LLM
        valid_categories = [
            "Regulator", "Supplier", "Consumer", "Competitor", 
            "Partner", "Influencer", "Internal"
        ]
        
        # Only validate a sample of stakeholders to reduce API calls
        validation_sample = resolved_stakeholders
        if len(resolved_stakeholders) > 20:
            # Process top 20 by mentions
            validation_sample = sorted(resolved_stakeholders, key=lambda x: x.get('mentions', 0), reverse=True)[:20]
        
        validated_stakeholders = []
        category_changes = 0
        
        for stakeholder in validation_sample:
            try:
                name = stakeholder.get('name', '')
                category = stakeholder.get('category', '')
                role = stakeholder.get('role', '')
                
                # Skip if already has valid category
                if category in valid_categories:
                    validated_stakeholders.append(stakeholder)
                    continue
                
                # Create simple prompt for category validation
                prompt = f"""As a stakeholder analyst, I need to assign the correct category to this stakeholder:

Name: {name}
Current Category: {category}
Role: {role}

The valid categories are:
- Regulator: Government or oversight bodies that create and enforce rules
- Supplier: Provides products or services to the organization or industry
- Consumer: Receives or benefits from products or services
- Competitor: Other organizations providing similar services or competing for resources
- Partner: Organizations working together with shared goals
- Influencer: Shapes opinions or decisions without direct authority
- Internal: Employees, management, or departments within the organization

Which ONE category best describes this stakeholder?
Answer with ONLY the category name from the list above.
"""
                
                # Call the LLM with simple category prompt
                messages = [HumanMessage(content=prompt)]
                response = await model.ainvoke(messages)
                
                # Strip and sanitize LLM output before processing (added as requested)
                response_content = response.content.strip()
                response_content = response_content.replace('"', '').replace("'", "").strip()
                
                # Extract category (just take the first word that matches a valid category)
                found_category = None
                for valid_cat in valid_categories:
                    if valid_cat.lower() in response_content.lower():
                        found_category = valid_cat
                        break
                
                # If we found a valid category and it's different from current
                if found_category and found_category != category:
                    # Create a copy to modify
                    updated = stakeholder.copy()
                    updated['category'] = found_category
                    updated['category_change_reason'] = f"Changed from '{category}' based on name/role analysis"
                    
                    print(f"Category change: {name} - {category} → {found_category}")
                    
                    validated_stakeholders.append(updated)
                    category_changes += 1
                else:
                    # Keep original
                    validated_stakeholders.append(stakeholder)
            
            except Exception as e:
                # Log the raw response content on errors (added as requested)
                print(f"Error validating category for {stakeholder.get('name')}: {str(e)}\nRaw response: {response.content}")
                # Keep original if error
                validated_stakeholders.append(stakeholder)
        
        # Add remaining stakeholders without validation (if we limited sample size)
        if len(resolved_stakeholders) > 20:
            validated_names = [v.get('name', '').lower().strip() for v in validated_stakeholders]
            for stakeholder in resolved_stakeholders:
                name = stakeholder.get('name', '').lower().strip()
                if name and name not in validated_names:
                    validated_stakeholders.append(stakeholder)
        
        # Create final output
        validation_results = {
            'workspace_id': state.workspace_id,
            'original_stakeholders': total_stakeholders,
            'validated_stakeholders': len(validated_stakeholders),
            'duplicates_merged': duplicates_merged,
            'category_changes': category_changes,
            'stakeholders': validated_stakeholders
        }
        
        # Save the validated stakeholders
        validated_file = os.path.join(state.output_dir, f"workspace_{state.workspace_id}_validated_stakeholders.json")
        with open(validated_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        
        # Update the aggregated stakeholders
        aggregated_stakeholders = state.aggregated_stakeholders.copy()
        aggregated_stakeholders['stakeholders'] = validated_stakeholders
        aggregated_stakeholders['stakeholder_validation'] = {
            'original_count': total_stakeholders,
            'validated_count': len(validated_stakeholders),
            'duplicates_merged': duplicates_merged,
            'category_changes': category_changes
        }
        
        # Update info for final output
        info = None
        if state.info:
            info = state.info.copy()
            info['stakeholders'] = validated_stakeholders
            
            # Update the combined results file
            combined_file = os.path.join(state.output_dir, f"workspace_{state.workspace_id}_analysis.json")
            if os.path.exists(combined_file):
                try:
                    with open(combined_file, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    analysis_data['stakeholders'] = validated_stakeholders
                    analysis_data['stakeholder_validation'] = {
                        'original_count': total_stakeholders,
                        'validated_count': len(validated_stakeholders),
                        'duplicates_merged': duplicates_merged,
                        'category_changes': category_changes
                    }
                    
                    with open(combined_file, 'w', encoding='utf-8') as f:
                        json.dump(analysis_data, f, indent=2)
                except Exception as e:
                    print(f"Error updating analysis file: {str(e)}")
        
        print(f"\nValidation Complete: Merged {duplicates_merged} duplicates and corrected {category_changes} categories")
        
        return {
            "validated_stakeholders": validation_results,
            "aggregated_stakeholders": aggregated_stakeholders,
            "info": info if state.info else None,
            "messages": [AIMessage(content=f"Validated {len(validated_stakeholders)} stakeholders with LLM assistance. Merged {duplicates_merged} duplicates and corrected {category_changes} category assignments.")]
        }
    
    except Exception as e:
        print(f"Error in stakeholder validation: {str(e)}")
        return {
            "error": f"Error in stakeholder validation: {str(e)}",
            "messages": [HumanMessage(content=f"Error in stakeholder validation: {str(e)}")]
        }

def route_after_initialization(state: State) -> str:
    """Route after workspace initialization."""
    if state.error:
        return "__end__"
    if not state.document_ids:
        return "__end__"  # No documents to process
    return "get_next_document"


def route_after_get_document(state: State) -> str:
    """Route after getting a document."""
    if state.error:
        return "__end__"
    if state.current_document_id and state.current_document_content:
        return "extract_stakeholders"
    if state.document_ids:
        return "get_next_document"
    return "aggregate_stakeholders"


def route_after_extraction(state: State) -> str:
    """Route after stakeholder extraction."""
    if state.error:
        return "__end__"
    if state.document_ids:
        return "get_next_document"
    return "aggregate_stakeholders"


def route_after_aggregation(state: State) -> str:
    """Route after stakeholder aggregation."""
    if state.error:
        return "__end__"
    return "validate_stakeholders"


def should_continue(state: State, config: RunnableConfig) -> str:
    """Decide whether to continue or end based on state."""
    configuration = Configuration.from_runnable_config(config)
    
    if state.loop_step >= configuration.max_loops:
        return "__end__"
    
    if state.error:
        return "__end__"
        
    return "aggregate_stakeholders"


# Create the workflow graph
workflow = StateGraph(State)
workflow.add_node("initialize_workspace", initialize_workspace)
workflow.add_node("get_next_document", get_next_document)
workflow.add_node("extract_stakeholders", extract_stakeholders)
workflow.add_node("aggregate_stakeholders", aggregate_stakeholders)
workflow.add_node("validate_stakeholders", validate_stakeholders)

# Add edges
workflow.add_edge("__start__", "initialize_workspace")
workflow.add_conditional_edges("initialize_workspace", route_after_initialization)
workflow.add_conditional_edges("get_next_document", route_after_get_document)
workflow.add_conditional_edges("extract_stakeholders", route_after_extraction)
workflow.add_conditional_edges("aggregate_stakeholders", route_after_aggregation)
workflow.add_edge("validate_stakeholders", "__end__")

# Compile the graph
graph = workflow.compile()
