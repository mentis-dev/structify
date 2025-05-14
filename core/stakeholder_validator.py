import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.messages import HumanMessage, AIMessage
from core.utils import init_model


class StakeholderValidator:
    """Class for validating stakeholders by finding duplicates and correcting categories"""
    
    def __init__(
        self,
        model: str = "openai/gpt-4o",
        output_dir: str = "output"
    ):
        self.model = model
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    async def validate_stakeholders(
        self, 
        stakeholders: List[Dict[str, Any]],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate stakeholders by finding duplicates and normalizing categories.
        
        Args:
            stakeholders: List of stakeholder dictionaries to validate
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with validation results
        """
        if verbose:
            print(f"Starting stakeholder validation for {len(stakeholders)} stakeholders")
        
        # Initialize LLM 
        model = init_model({"configurable": {"model": self.model}})
        
        # Step 1: Find potential duplicates based on name similarity
        duplicate_groups = self._find_duplicate_groups(stakeholders)
        
        if verbose:
            print(f"Found {len(duplicate_groups)} potential duplicate groups")
        
        # Step 2: Process duplicates with LLM
        resolved_stakeholders = []
        handled_duplicates = set()
        duplicates_merged = 0
        
        for group_idx, group in enumerate(duplicate_groups):
            try:
                if verbose:
                    group_str = "\n".join([f"- {s.get('name', '')} ({s.get('category', 'Unknown')}) - {s.get('role', 'Unknown role')}" for s in group])
                    print(f"\nDuplicate group {group_idx+1}:\n{group_str}")
                
                # Use simple yes/no prompt for robustness
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
                    # Choose the best representative using scoring
                    consolidated = self._get_best_representative(group)
                    resolved_stakeholders.append(consolidated)
                    
                    # Add names to handled set
                    handled_duplicates.update([s.get('name', '').lower().strip() for s in group])
                    
                    # Count merged duplicates
                    duplicates_merged += len(group) - 1
                    
                    if verbose:
                        consolidated_names = [s.get('name', '') for s in group]
                        print(f"Consolidated: {', '.join(consolidated_names)} → {consolidated.get('name', '')}")
                else:
                    # If not duplicates, keep all original stakeholders
                    if verbose:
                        print(f"Not duplicates, keeping separate")
                    for stakeholder in group:
                        name = stakeholder.get('name', '').lower().strip()
                        if name and name not in handled_duplicates:
                            resolved_stakeholders.append(stakeholder)
                            handled_duplicates.add(name)
            
            except Exception as e:
                if verbose:
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
        
        # Only validate a sample if there are many stakeholders
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
                
                # Extract category
                found_category = None
                for valid_cat in valid_categories:
                    if valid_cat.lower() in response_content.lower():
                        found_category = valid_cat
                        break
                
                # If found valid category and it's different
                if found_category and found_category != category:
                    # Create a copy to modify
                    updated = stakeholder.copy()
                    updated['category'] = found_category
                    updated['category_change_reason'] = f"Changed from '{category}' based on name/role analysis"
                    
                    if verbose:
                        print(f"Category change: {name} - {category} → {found_category}")
                    
                    validated_stakeholders.append(updated)
                    category_changes += 1
                else:
                    # Keep original
                    validated_stakeholders.append(stakeholder)
            
            except Exception as e:
                if verbose:
                    print(f"Error validating category for {stakeholder.get('name')}: {str(e)}")
                # Keep original if error
                validated_stakeholders.append(stakeholder)
        
        # Add remaining stakeholders without validation
        if len(resolved_stakeholders) > 20:
            validated_names = [v.get('name', '').lower().strip() for v in validated_stakeholders]
            for stakeholder in resolved_stakeholders:
                name = stakeholder.get('name', '').lower().strip()
                if name and name not in validated_names:
                    validated_stakeholders.append(stakeholder)
        
        # Create final output
        validation_results = {
            'original_stakeholders': len(stakeholders),
            'validated_stakeholders': len(validated_stakeholders),
            'duplicates_merged': duplicates_merged,
            'category_changes': category_changes,
            'stakeholders': validated_stakeholders
        }
        
        # Save the validated stakeholders
        output_file = os.path.join(self.output_dir, f"validated_stakeholders.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        
        if verbose:
            print(f"\nValidation Complete: Merged {duplicates_merged} duplicates and corrected {category_changes} categories")
            print(f"Results saved to {output_file}")
        
        return validation_results
    
    def _find_duplicate_groups(self, stakeholders: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find potential duplicate groups based on name similarity"""
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
                similarity = self._similarity_score(name1, name2)
                contained = name1 in name2 or name2 in name1
                
                if similarity > 0.8 or contained:
                    similar_stakeholders.append(j)
            
            if len(similar_stakeholders) > 1:
                duplicate_groups.append([stakeholders[idx] for idx in similar_stakeholders])
                processed.update(similar_stakeholders)
        
        return duplicate_groups
    
    def _similarity_score(self, a, b):
        """Calculate string similarity score between 0 and 1"""
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
    
    def _get_best_representative(self, group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Choose the best representative stakeholder from a group of duplicates"""
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
        
        return consolidated


# Convenience function
async def validate_stakeholders(
    stakeholders: List[Dict[str, Any]],
    model: str = "openai/gpt-4o",
    output_dir: str = "output",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate a list of stakeholders by finding duplicates and normalizing categories.
    
    Args:
        stakeholders: List of stakeholder dictionaries to validate
        model: LLM model to use
        output_dir: Directory to save results
        verbose: Whether to print progress
        
    Returns:
        Dictionary with validation results
    """
    validator = StakeholderValidator(model=model, output_dir=output_dir)
    return await validator.validate_stakeholders(stakeholders=stakeholders, verbose=verbose)
