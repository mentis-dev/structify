import json
import os
from typing import Any, Dict, Optional, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from core.configuration import Configuration
from core.state import State
from core.utils import init_model
from core.supabase_db.select_data import initialize_supabase, get_brains_per_workspace, get_documents_per_brain
from core.supabase_db.supabase_db import get_vectors_by_knowledge_ids
from core.stakeholder_validator import validate_stakeholders  # Import the validator

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
        
        # Store original stakeholders for comparison
        original_stakeholders = aggregated_stakeholders.copy()
        
        return {
            "aggregated_stakeholders": aggregated_result,
            "factors": all_factors,
            "pain_points": all_pain_points,
            "info": info,  # Final output that will be in OutputState
            "original_stakeholders": original_stakeholders,  # Store for validation comparison
            "messages": [AIMessage(content=f"Aggregated {len(aggregated_stakeholders)} stakeholders, {len(all_factors)} factors, and {len(all_pain_points)} pain points from {len(state.processed_documents)} documents. Created multi-level ecosystem map.")]
        }
    
    except Exception as e:
        return {
            "error": f"Error aggregating results: {str(e)}",
            "messages": [HumanMessage(content=f"Error aggregating results: {str(e)}")]
        }

async def validate_stakeholders_node(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Validate stakeholders by finding duplicates and normalizing categories.
    """
    if state.error:
        return {
            "error": state.error,
            "messages": [HumanMessage(content=f"Error in previous step: {state.error}")]
        }
        
    if not state.aggregated_stakeholders or "stakeholders" not in state.aggregated_stakeholders:
        return {
            "error": "No aggregated stakeholders available for validation",
            "messages": [HumanMessage(content="No stakeholders to validate. Skipping validation step.")]
        }
    
    try:
        configuration = Configuration.from_runnable_config(config)
        model_name = configuration.model
        
        # Get stakeholders from aggregated results
        stakeholders = state.aggregated_stakeholders.get("stakeholders", [])
        
        if not stakeholders:
            return {
                "messages": [HumanMessage(content="No stakeholders found to validate.")]
            }
        
        # Run validation
        validation_results = await validate_stakeholders(
            stakeholders=stakeholders,
            model=model_name,
            output_dir=state.output_dir,
            verbose=True
        )
        
        # Extract the validated stakeholders
        validated_stakeholders = validation_results.get("stakeholders", [])
        
        # Update the aggregated stakeholders
        aggregated_stakeholders = state.aggregated_stakeholders.copy()
        aggregated_stakeholders["stakeholders"] = validated_stakeholders
        aggregated_stakeholders["stakeholder_validation"] = {
            "original_count": validation_results.get("original_stakeholders", 0),
            "validated_count": validation_results.get("validated_stakeholders", 0),
            "duplicates_merged": validation_results.get("duplicates_merged", 0),
            "category_changes": validation_results.get("category_changes", 0)
        }
        
        # Update combined results file
        combined_file = os.path.join(state.output_dir, f"workspace_{state.workspace_id}_analysis.json")
        
        if os.path.exists(combined_file):
            try:
                with open(combined_file, 'r', encoding='utf-8') as f:
                    combined_result = json.load(f)
                
                # Update stakeholders in combined results
                combined_result["stakeholders"] = validated_stakeholders
                combined_result["stakeholder_validation"] = aggregated_stakeholders["stakeholder_validation"]
                
                # Save updated combined results
                with open(combined_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_result, f, indent=2)
            except Exception as e:
                print(f"Error updating combined results file: {str(e)}")
        
        # Save validated stakeholders separately
        validated_file = os.path.join(state.output_dir, f"workspace_{state.workspace_id}_validated_stakeholders.json")
        
        with open(validated_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        
        # Update info for final output
        info = state.info.copy() if state.info else None
        
        if info and "stakeholders" in info:
            info["stakeholders"] = validated_stakeholders
            info["stakeholder_validation"] = aggregated_stakeholders["stakeholder_validation"]
        
        duplicates_merged = validation_results.get("duplicates_merged", 0)
        category_changes = validation_results.get("category_changes", 0)
        
        return {
            "aggregated_stakeholders": aggregated_stakeholders,
            "validated_stakeholders": validation_results,
            "info": info,
            "messages": [AIMessage(content=f"Validated stakeholders: merged {duplicates_merged} duplicates and corrected {category_changes} category assignments.")]
        }
    
    except Exception as e:
        return {
            "error": f"Error validating stakeholders: {str(e)}",
            "messages": [HumanMessage(content=f"Error validating stakeholders: {str(e)}")]
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
    return "validate_stakeholders"  # Always go to validation after aggregation

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
workflow.add_node("validate_stakeholders", validate_stakeholders_node)  # Add the validation node

# Add edges
workflow.add_edge("__start__", "initialize_workspace")
workflow.add_conditional_edges("initialize_workspace", route_after_initialization)
workflow.add_conditional_edges("get_next_document", route_after_get_document)
workflow.add_conditional_edges("extract_stakeholders", route_after_extraction)
workflow.add_conditional_edges("aggregate_stakeholders", route_after_aggregation)  # Add conditional edge after aggregation
workflow.add_edge("validate_stakeholders", "__end__")  # Add edge from validation to end

# Compile the graph
graph = workflow.compile()
