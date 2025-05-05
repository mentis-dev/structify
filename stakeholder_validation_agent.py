#!/usr/bin/env python3
import os
import json
import asyncio
import argparse
import time
from dotenv import load_dotenv
from difflib import SequenceMatcher

from enrichment.state import InputState
from enrichment.configuration import Configuration
from enrichment.utils import init_model
from langchain_core.messages import HumanMessage, AIMessage

def print_banner(text):
    """Print a banner with text centered"""
    width = 80
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")


def similarity_score(a, b):
    """Calculate string similarity score between 0 and 1"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()



async def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Validate stakeholders from a workspace analysis")
    parser.add_argument("workspace_id", help="ID of the workspace to process")
    parser.add_argument("--output-dir", default="output", help="Directory for output files")
    parser.add_argument("--model", default="openai/gpt-4o", help="The LLM model to use")
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Run the validation
    results = await validate_stakeholders(
        workspace_id=args.workspace_id,
        output_dir=args.output_dir,
        model=args.model
    )
    
    if results.get("error"):
        print(f"Validation completed with error: {results.get('error')}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

async def validate_stakeholders(
    state: State, *, config: Optional[RunnableConfig] = None
) -> Dict[str, Any]:
    """
    Validate stakeholders using LLM but with robust error handling.
    Uses structured, simple interactions that are less prone to parsing errors.
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
                response_content = response.content.strip().upper()
                
                # Check if response contains YES
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
                print(f"Error processing duplicate group: {str(e)}")
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
                response_content = response.content.strip()
                
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
                print(f"Error validating category for {stakeholder.get('name')}: {str(e)}")
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
    
if __name__ == "__main__":
    asyncio.run(main())
