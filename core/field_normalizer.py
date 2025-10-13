"""
Field Name Normalizer for Causality Analysis
Handles robust field name normalization across different LLM response formats
"""

from typing import Dict, List, Any, Optional


class FieldNameNormalizer:
    """Robust field name normalization for causality analysis"""
    
    def __init__(self):
        # Define standard field mappings
        self.field_mappings = {
            'cause_fields': [
                'cause_entity_id', 'cause_pain_point_id', 'cause_entity', 
                'cause', 'source', 'from', 'cause_id', 'from_entity',
                'causing_entity_id', 'causal_entity_id', 'from_id'
            ],
            'effect_fields': [
                'effect_entity_id', 'effect_pain_point_id', 'effect_entity',
                'effect', 'target', 'to', 'effect_id', 'to_entity',
                'affected_entity_id', 'resulting_entity_id', 'to_id'
            ],
            'confidence_fields': [
                'confidence', 'strength', 'score', 'probability', 'certainty',
                'confidence_score', 'strength_score'
            ],
            'sequence_fields': [
                'entity_sequence', 'chain', 'sequence', 'path', 'entities', 
                'nodes', 'chain_entities', 'entity_chain'
            ],
            'loop_entity_fields': [
                'participating_entities', 'loop_entities', 'entities', 
                'nodes', 'loop', 'participants', 'involved_entities',
                'loop_participants', 'cycle_entities'
            ],
            'mechanism_fields': [
                'mechanism', 'description', 'explanation', 'how', 'process'
            ],
            'evidence_fields': [
                'evidence', 'reasoning', 'justification', 'support', 'basis'
            ]
        }
    
    def normalize_relationship(self, relationship: Dict) -> Dict:
        """Normalize a single relationship to standard field names"""
        if not isinstance(relationship, dict):
            return relationship
            
        normalized = relationship.copy()
        
        # Normalize cause field
        cause_value = self._get_first_valid_field(relationship, self.field_mappings['cause_fields'])
        if cause_value:
            normalized['cause_entity_id'] = str(cause_value).strip()
        
        # Normalize effect field  
        effect_value = self._get_first_valid_field(relationship, self.field_mappings['effect_fields'])
        if effect_value:
            normalized['effect_entity_id'] = str(effect_value).strip()
            
        # Normalize confidence field
        confidence_value = self._get_first_valid_field(relationship, self.field_mappings['confidence_fields'])
        if confidence_value is not None:
            try:
                normalized['confidence'] = float(confidence_value)
            except (ValueError, TypeError):
                normalized['confidence'] = 70  # Default fallback
        
        # Normalize mechanism field
        mechanism_value = self._get_first_valid_field(relationship, self.field_mappings['mechanism_fields'])
        if mechanism_value:
            normalized['mechanism'] = str(mechanism_value).strip()
        
        # Normalize evidence field
        evidence_value = self._get_first_valid_field(relationship, self.field_mappings['evidence_fields'])
        if evidence_value:
            normalized['evidence'] = str(evidence_value).strip()
            
        return normalized
    
    def normalize_chain(self, chain: Dict) -> Dict:
        """Normalize a chain to standard field names"""
        if not isinstance(chain, dict):
            return chain
            
        normalized = chain.copy()
        
        # Normalize sequence field
        sequence_value = self._get_first_valid_field(chain, self.field_mappings['sequence_fields'])
        if sequence_value and isinstance(sequence_value, list):
            # Clean up entity IDs in sequence
            cleaned_sequence = [str(entity_id).strip() for entity_id in sequence_value if entity_id]
            normalized['entity_sequence'] = cleaned_sequence
        elif sequence_value:
            # Handle case where sequence is not a list
            normalized['entity_sequence'] = [str(sequence_value).strip()]
            
        # Ensure confidence exists
        if 'confidence' not in normalized:
            confidence_value = self._get_first_valid_field(chain, self.field_mappings['confidence_fields'])
            if confidence_value is not None:
                try:
                    normalized['confidence'] = float(confidence_value)
                except (ValueError, TypeError):
                    normalized['confidence'] = 70
            else:
                normalized['confidence'] = 70
                
        return normalized
    
    def normalize_loop(self, loop: Dict) -> Dict:
        """Normalize a loop to standard field names"""
        if not isinstance(loop, dict):
            return loop
            
        normalized = loop.copy()
        
        # Normalize participating entities field
        entities_value = self._get_first_valid_field(loop, self.field_mappings['loop_entity_fields'])
        if entities_value and isinstance(entities_value, list):
            # Clean up entity IDs in loop
            cleaned_entities = [str(entity_id).strip() for entity_id in entities_value if entity_id]
            normalized['participating_entities'] = cleaned_entities
        elif entities_value:
            # Handle case where entities is not a list
            normalized['participating_entities'] = [str(entities_value).strip()]
        
        # Normalize loop strength
        strength_value = self._get_first_valid_field(loop, ['loop_strength', 'strength', 'confidence', 'score'])
        if strength_value is not None:
            try:
                normalized['loop_strength'] = float(strength_value)
            except (ValueError, TypeError):
                normalized['loop_strength'] = 70
        elif 'loop_strength' not in normalized:
            normalized['loop_strength'] = 70
            
        return normalized
    
    def _get_first_valid_field(self, data: Dict, field_names: List[str]) -> Any:
        """Get the first valid field value from a list of possible field names"""
        for field_name in field_names:
            if field_name in data and data[field_name] is not None:
                value = data[field_name]
                # Handle empty strings
                if isinstance(value, str):
                    stripped_value = value.strip()
                    if stripped_value:  # Non-empty string
                        return stripped_value
                elif value:  # Non-string, non-None, truthy value
                    return value
        return None
    
    def normalize_relationships_batch(self, relationships: List[Dict]) -> List[Dict]:
        """Normalize a batch of relationships"""
        if not isinstance(relationships, list):
            return []
        return [self.normalize_relationship(rel) for rel in relationships if isinstance(rel, dict)]
    
    def normalize_chains_batch(self, chains: List[Dict]) -> List[Dict]:
        """Normalize a batch of chains"""
        if not isinstance(chains, list):
            return []
        return [self.normalize_chain(chain) for chain in chains if isinstance(chain, dict)]
    
    def normalize_loops_batch(self, loops: List[Dict]) -> List[Dict]:
        """Normalize a batch of loops"""
        if not isinstance(loops, list):
            return []
        return [self.normalize_loop(loop) for loop in loops if isinstance(loop, dict)]
    
    def validate_normalized_relationship(self, relationship: Dict) -> bool:
        """Validate that a normalized relationship has required fields"""
        required_fields = ['cause_entity_id', 'effect_entity_id']
        return all(
            field in relationship and 
            relationship[field] and 
            str(relationship[field]).strip() 
            for field in required_fields
        )
    
    def validate_normalized_chain(self, chain: Dict) -> bool:
        """Validate that a normalized chain has required fields"""
        return (
            'entity_sequence' in chain and 
            isinstance(chain['entity_sequence'], list) and 
            len(chain['entity_sequence']) >= 2
        )
    
    def validate_normalized_loop(self, loop: Dict) -> bool:
        """Validate that a normalized loop has required fields"""
        return (
            'participating_entities' in loop and 
            isinstance(loop['participating_entities'], list) and 
            len(loop['participating_entities']) >= 2
        )
