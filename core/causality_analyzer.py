import json
import os
import re
import asyncio
from typing import Dict, List, Tuple, Any, Optional, Set, cast
from datetime import datetime
from collections import defaultdict, Counter
import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

# Import your existing model initialization function
from core.utils import init_model
from core.configuration import Configuration

# Import the new field normalizer
from core.field_normalizer import FieldNameNormalizer


class UniversalCausalityValidator:
    """Enhanced validator supporting any entity type and domain"""
    
    def __init__(self, config_path: str, entities: List[Dict] = None, domain_config: Dict = None):
        self.config = self._load_config(config_path)
        self.entities = entities or []
        self.domain_config = domain_config or {}
        self.entity_ids = {entity.get('id', entity.get('entity_id', '')) for entity in self.entities}
        self.entity_prefix = self.domain_config.get('entity_prefix', 'PP')
        self.strong_themes = self._auto_detect_themes()
        self.normalizer = FieldNameNormalizer()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration with fallback to defaults"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Default configuration if file not found"""
        return {
            "analysis_parameters": {
                "minimum_confidence_threshold": 50,
                "chain_confidence_threshold": 45,
                "loop_confidence_threshold": 50,
                "enable_confidence_boosting": True
            },
            "causal_patterns": {
                "explicit_causal_keywords": ["leads to", "causes", "results in", "triggers"],
                "resource_constraint_patterns": ["shortage", "lack of", "insufficient"],
                "quality_impact_patterns": ["poor quality", "delays", "inefficient"],
                "adoption_barrier_patterns": ["high cost", "complex", "barriers"]
            },
            "confidence_boosting": {
                "explicit_language_boost": 15,
                "pattern_match_boost": 10,
                "domain_knowledge_boost": 8
            }
        }
    
    def _auto_detect_themes(self) -> Set[str]:
        """Auto-detect strong themes from entity data"""
        themes = set()
        all_text = " ".join([
            entity.get('description', '') + " " + entity.get('title', '') + " " + 
            entity.get('name', '') + " " + str(entity.get('themes', []))
            for entity in self.entities
        ]).lower()
        
        # Universal theme patterns that work across domains
        theme_patterns = {
            'cost': ['cost', 'expensive', 'price', 'afford', 'budget', 'financial'],
            'technology': ['technology', 'digital', 'tech', 'system', 'software', 'automation'],
            'workforce': ['staff', 'worker', 'employee', 'human resources', 'personnel', 'team'],
            'quality': ['quality', 'standard', 'performance', 'service', 'reliability'],
            'accessibility': ['access', 'reach', 'available', 'barrier', 'obstacle'],
            'awareness': ['awareness', 'knowledge', 'understand', 'information', 'education'],
            'capacity': ['capacity', 'throughput', 'volume', 'bandwidth', 'limit'],
            'efficiency': ['efficiency', 'productivity', 'optimization', 'streamline'],
            'compliance': ['compliance', 'regulation', 'policy', 'standard', 'requirement'],
            'security': ['security', 'risk', 'vulnerability', 'threat', 'safety']
        }
        
        for theme, keywords in theme_patterns.items():
            if sum(1 for keyword in keywords if keyword in all_text) >= 2:
                themes.add(theme)
        
        return themes
    
    def validate_causal_relationship(self, relationship: Dict) -> Tuple[bool, float]:
        """Enhanced validation for individual causal relationships with normalization"""
        
        # Normalize the relationship first
        normalized_rel = self.normalizer.normalize_relationship(relationship)
        
        # Check if normalization produced valid required fields
        if not self.normalizer.validate_normalized_relationship(normalized_rel):
            return False, 0
        
        # Extract normalized values
        base_confidence = normalized_rel.get('confidence', 70)
        mechanism = normalized_rel.get('mechanism', '').lower()
        cause_id = normalized_rel.get('cause_entity_id', '')
        effect_id = normalized_rel.get('effect_entity_id', '')
        
        # Basic validation
        min_threshold = self.config['analysis_parameters'].get('minimum_confidence_threshold', 50)
        if base_confidence < min_threshold:
            return False, base_confidence
        
        # Validate entity IDs exist
        if cause_id not in self.entity_ids or effect_id not in self.entity_ids:
            return False, base_confidence
        
        # Apply confidence boosting if enabled
        if self.config['analysis_parameters'].get('enable_confidence_boosting', False):
            adjusted_confidence = self._apply_confidence_boosting(
                base_confidence, mechanism, normalized_rel
            )
        else:
            adjusted_confidence = base_confidence
        
        # Final threshold check
        return adjusted_confidence >= min_threshold, adjusted_confidence
    
    def _apply_confidence_boosting(self, base_confidence: float, mechanism: str, relationship: Dict) -> float:
        """Apply confidence boosting based on evidence strength"""
        boost = 0
        boosting_config = self.config.get('confidence_boosting', {})
        
        # Check for explicit causal language
        explicit_keywords = self.config['causal_patterns']['explicit_causal_keywords']
        if any(keyword in mechanism for keyword in explicit_keywords):
            boost += boosting_config.get('explicit_language_boost', 15)
        
        # Check for pattern matches
        pattern_categories = ['resource_constraint_patterns', 'quality_impact_patterns', 'adoption_barrier_patterns']
        for category in pattern_categories:
            patterns = self.config['causal_patterns'].get(category, [])
            if any(pattern in mechanism for pattern in patterns):
                boost += boosting_config.get('pattern_match_boost', 10)
                break
        
        # Check theme alignment
        if any(theme in mechanism for theme in self.strong_themes):
            boost += boosting_config.get('domain_knowledge_boost', 8)
        
        return min(base_confidence + boost, 100)  # Cap at 100%


class UniversalCausalityAnalyzer:
    """Universal causality analyzer working with any entity type"""
    
    def __init__(self, 
                 model: str = "gpt-4",
                 output_dir: str = "output",
                 temperature: float = 0.1,
                 prompts_dir: str = "../prompts"):
        load_dotenv()
        self.model = model
        self.output_dir = output_dir
        self.temperature = temperature
        self.prompts_dir = prompts_dir
        self.normalizer = FieldNameNormalizer()
    
    async def analyze_causality(self, 
                              entities: List[Dict],
                              domain_config: Dict = None,
                              prompt_filename: str = 'causality_prompt.txt',
                              config_filename: str = 'causality_config.json',
                              schema_filename: str = 'causality_schema.json',
                              verbose: bool = False) -> Dict:
        """Main causality analysis with universal domain support"""
        
        try:
            # Set up domain configuration
            domain_config = domain_config or self._infer_domain_config(entities)
            
            # Load prompt and schema from prompts directory
            prompt_path = os.path.join(self.prompts_dir, prompt_filename)
            config_path = os.path.join(self.prompts_dir, config_filename)
            schema_path = os.path.join(self.prompts_dir, schema_filename)
            
            prompt = self._load_prompt(prompt_path)
            schema = self._load_schema(schema_path)
            
            # Prepare analysis request
            analysis_request = self._prepare_analysis_request(entities, domain_config, prompt, schema)
            
            if verbose:
                print(f"Analyzing {len(entities)} entities for causality...")
                print(f"Domain: {domain_config.get('domain_name', 'Unknown')}")
                print(f"Entity type: {domain_config.get('entity_type', 'Unknown')}")
                print(f"Using model: {self.model}")
            
            # Get LLM analysis
            raw_result = await self._get_llm_analysis(analysis_request, verbose)
            
            # Parse and validate results
            parsed_results = self._parse_llm_response(raw_result, verbose)
            
            # Enhanced validation with normalization
            validator = UniversalCausalityValidator(config_path, entities, domain_config)
            validated_results = self._apply_enhanced_validation(parsed_results, validator, verbose)
            
            # Auto-generate chains from relationships
            if validated_results.get('causal_relationships'):
                auto_chains = self._build_chains_from_relationships(
                    validated_results['causal_relationships'], domain_config, verbose
                )
                existing_chains = validated_results.get('causal_chains', [])
                validated_results['causal_chains'] = self._merge_chain_results(existing_chains, auto_chains)
            
            # Auto-generate loops from relationships
            if validated_results.get('causal_relationships'):
                auto_loops = self._detect_loops_from_relationships(
                    validated_results['causal_relationships'], domain_config, verbose
                )
                existing_loops = validated_results.get('causal_loops', [])
                validated_results['causal_loops'] = self._merge_loop_results(existing_loops, auto_loops)
            
            # Add domain context
            validated_results['domain_context'] = domain_config
            
            # Network analysis
            network_analysis = self._perform_network_analysis(validated_results, entities, verbose)
            validated_results.update(network_analysis)
            
            # Add analysis metadata
            validated_results['analysis_metadata'] = self._generate_analysis_metadata(validated_results)
            
            if verbose:
                self._print_analysis_summary(validated_results)
            
            return validated_results
            
        except Exception as e:
            if verbose:
                print(f"Error in causality analysis: {str(e)}")
            return self._get_empty_result(domain_config)
    
    def _apply_enhanced_validation(self, results: Dict, validator: UniversalCausalityValidator, verbose: bool = False) -> Dict:
        """Apply validation with early field normalization"""
        validated = self._get_empty_result()
        
        # Normalize and validate relationships
        relationships = results.get('causal_relationships', [])
        if verbose:
            print(f"Processing {len(relationships)} relationships for validation...")
        
        valid_relationships = []
        for rel in relationships:
            is_valid, confidence = validator.validate_causal_relationship(rel)
            if is_valid:
                # Get the normalized version
                normalized_rel = validator.normalizer.normalize_relationship(rel)
                normalized_rel['confidence'] = confidence
                valid_relationships.append(normalized_rel)
        
        validated['causal_relationships'] = valid_relationships
        
        # Normalize chains and loops
        chains = results.get('causal_chains', [])
        normalized_chains = self.normalizer.normalize_chains_batch(chains)
        # Filter valid chains
        self._current_relationships = valid_relationships
        validated_chains = self._validate_chains_against_relationships(normalized_chains, valid_relationships)
        validated['causal_chains'] = validated_chains

        loops = results.get('causal_loops', [])
        normalized_loops = self.normalizer.normalize_loops_batch(loops)
        validated_loops = self._validate_loops_against_relationships(normalized_loops, valid_relationships)
        validated['causal_loops'] = validated_loops
  
        # Copy other components
        validated['root_causes'] = results.get('root_causes', [])
        validated['ultimate_effects'] = results.get('ultimate_effects', [])
        validated['leverage_points'] = results.get('leverage_points', [])
        
        if verbose:
            print(f"Validation results:")
            print(f"  Relationships: {len(validated['causal_relationships'])}/{len(relationships)}")
            print(f"  Chains: {len(validated['causal_chains'])}/{len(chains)}")
            print(f"  Loops: {len(validated['causal_loops'])}/{len(loops)}")
        
        return validated

    def _validate_chains_against_relationships(self, chains: List[Dict], relationships: List[Dict]) -> List[Dict]:
        """Validate that all edges in chains exist in relationships"""
        relationship_edges = set()
        for rel in relationships:
            cause = rel.get('cause_entity_id', '')
            effect = rel.get('effect_entity_id', '')
            if cause and effect:
                relationship_edges.add((cause, effect))
        
        validated_chains = []
        for chain in chains:
            if not self.normalizer.validate_normalized_chain(chain):
                continue
                
            sequence = chain.get('entity_sequence', [])
            if len(sequence) < 3:
                continue
            
            valid_chain = True
            for i in range(len(sequence) - 1):
                from_node = sequence[i]
                to_node = sequence[i + 1]
                if (from_node, to_node) not in relationship_edges:
                    valid_chain = False
                    break
            
            if valid_chain:
                validated_chains.append(chain)
        
        return validated_chains

    def _validate_loops_against_relationships(self, loops: List[Dict], relationships: List[Dict]) -> List[Dict]:
        """Validate that all edges in loops exist in relationships and require minimum 3 nodes"""
        relationship_edges = set()
        for rel in relationships:
            cause = rel.get('cause_entity_id', '')
            effect = rel.get('effect_entity_id', '')
            if cause and effect:
                relationship_edges.add((cause, effect))
        
        validated_loops = []
        for loop in loops:
            if not self.normalizer.validate_normalized_loop(loop):
                continue
                
            participants = loop.get('participating_entities', [])
            # CRITICAL FIX: Require minimum 3 unique nodes (reject 2-node cycles)
            if len(participants) < 3:
                continue
            
            valid_loop = True
            for i in range(len(participants)):
                from_node = participants[i]
                to_node = participants[(i + 1) % len(participants)]
                if (from_node, to_node) not in relationship_edges:
                    valid_loop = False
                    break
            
            if valid_loop:
                validated_loops.append(loop)
        
        return validated_loops
    
    
    def _build_chains_from_relationships(self, relationships: List[Dict], domain_config: Dict, verbose: bool = False) -> List[Dict]:
        """Universal chain detection with normalized field handling"""
        
        if verbose:
            print(f"Building causal chains from {len(relationships)} relationships...")
        
        # Build directed graph using normalized relationships
        graph = defaultdict(list)
        confidence_map = {}
        
        for rel in relationships:
            # Relationships should already be normalized by validation
            cause = rel.get('cause_entity_id', '')
            effect = rel.get('effect_entity_id', '')
            confidence = rel.get('confidence', 50)
            
            if cause and effect:
                graph[cause].append(effect)
                confidence_map[(cause, effect)] = confidence
        
        if verbose:
            print(f"Built graph with {len(graph)} nodes and {len(confidence_map)} edges")
        
        # Find all paths of length 3+
        chains = []
        chain_id = 1
        
        def dfs_chains(current_node, visited_path, max_depth=6):
            nonlocal chain_id
            
            if len(visited_path) > max_depth:
                return
            
            if len(visited_path) >= 3:  # Minimum chain length
                path_confidences = []
                for i in range(len(visited_path)-1):
                    conf = confidence_map.get((visited_path[i], visited_path[i+1]), 50)
                    path_confidences.append(conf)
                
                if path_confidences:
                    avg_confidence = sum(path_confidences) / len(path_confidences)
                    overall_strength = min(avg_confidence * (0.95 ** (len(visited_path) - 3)), 100)
                    
                    chains.append({
                        "chain_id": f"CHAIN-{chain_id:03d}",
                        "entity_sequence": visited_path.copy(),
                        "chain_description": self._generate_chain_description(visited_path, domain_config),
                        "overall_strength": round(overall_strength, 1),
                        "chain_type": "linear",
                        "confidence": round(avg_confidence),
                        "intervention_priority": "upstream" if len(visited_path) <= 4 else "midstream"
                    })
                    chain_id += 1
            
            # Continue exploring
            for next_node in graph.get(current_node, []):
                if next_node not in visited_path:  # Avoid cycles in linear chains
                    dfs_chains(next_node, visited_path + [next_node], max_depth)
        
        # Start DFS from all nodes
        explored_starts = set()
        for start_node in graph.keys():
            if start_node not in explored_starts:
                dfs_chains(start_node, [start_node])
                explored_starts.add(start_node)
        
        # Remove duplicates and return top chains
        unique_chains = self._deduplicate_chains(chains)
        sorted_chains = sorted(unique_chains, key=lambda x: x.get('confidence', 0), reverse=True)[:15]
        
        if verbose:
            print(f"Generated {len(sorted_chains)} unique causal chains")
        
        return sorted_chains


    def _detect_loops_from_relationships(self, relationships: List[Dict], domain_config: Dict, verbose: bool = False) -> List[Dict]:
        """FIXED: Detect genuine feedback loops with correct cycle detection logic"""
        
        if verbose:
            print(f"Detecting causal loops from {len(relationships)} relationships...")
        
        # Build directed graph using normalized relationships
        graph = defaultdict(list)
        confidence_map = {}
        
        for rel in relationships:
            # Relationships should already be normalized by validation
            cause = rel.get('cause_entity_id', '')
            effect = rel.get('effect_entity_id', '')
            confidence = rel.get('confidence', 50)
            
            if cause and effect:
                graph[cause].append(effect)
                confidence_map[(cause, effect)] = confidence
        
        if verbose:
            print(f"Built graph with {len(graph)} nodes")
            for node, targets in graph.items():
                print(f"  {node} -> {targets}")
        
        loops = []
        loop_id = 1
        
        # Use proper cycle detection algorithm
        def find_cycles_dfs(current_path, visited_in_path):
            """DFS-based cycle detection that only finds genuine cycles"""
            nonlocal loop_id
            
            current_node = current_path[-1]
            
            for neighbor in graph.get(current_node, []):
                if neighbor in visited_in_path:
                    # Found a genuine cycle - check if it's back to a node in our current path
                    try:
                        cycle_start_idx = current_path.index(neighbor)
                        cycle_nodes = current_path[cycle_start_idx:] + [neighbor]
                        
                        # Only count cycles of length 3+ (at least 3 unique nodes)
                        if len(cycle_nodes) >= 4:  # 3 unique nodes + 1 duplicate
                            unique_cycle = cycle_nodes[:-1]  # Remove duplicate end node
                            
                            # Verify this is a genuine loop by checking all edges exist
                            valid_loop = True
                            loop_confidences = []
                            for i in range(len(unique_cycle)):
                                from_node = unique_cycle[i]
                                to_node = unique_cycle[(i + 1) % len(unique_cycle)]
                                if (from_node, to_node) not in confidence_map:
                                    valid_loop = False
                                    break
                                loop_confidences.append(confidence_map[(from_node, to_node)])
                            
                            if valid_loop and loop_confidences:
                                avg_confidence = sum(loop_confidences) / len(loop_confidences)
                                
                                loops.append({
                                    "loop_id": f"LOOP-{loop_id:03d}",
                                    "participating_entities": unique_cycle,
                                    "loop_type": "reinforcing",
                                    "loop_description": self._generate_loop_description(cycle_nodes, domain_config),
                                    "loop_strength": round(avg_confidence, 1),
                                    "break_points": self._identify_break_points(cycle_nodes)
                                })
                                loop_id += 1
                                
                                if verbose:
                                    print(f"Found genuine loop: {' -> '.join(unique_cycle)} -> {unique_cycle[0]}")
                    except ValueError:
                        # neighbor not in current path, continue
                        pass
                
                elif len(current_path) < 8:  # Limit depth to prevent infinite recursion
                    # Continue exploring
                    find_cycles_dfs(current_path + [neighbor], visited_in_path | {neighbor})
        
        # Start cycle detection from each node that has outgoing edges
        for start_node in graph.keys():
            find_cycles_dfs([start_node], {start_node})
        
        # Remove duplicates (same cycle detected from different starting points)
        unique_loops = []
        seen_cycles = set()
        
        for loop in loops:
            participants = loop.get('participating_entities', [])
            if participants:
                # Normalize cycle representation (start from lexicographically smallest)
                min_idx = participants.index(min(participants))
                normalized_cycle = participants[min_idx:] + participants[:min_idx]
                cycle_signature = tuple(normalized_cycle)
                
                if cycle_signature not in seen_cycles:
                    seen_cycles.add(cycle_signature)
                    unique_loops.append(loop)
        
        sorted_loops = sorted(unique_loops, key=lambda x: x.get('loop_strength', 0), reverse=True)[:10]
        
        if verbose:
            print(f"Generated {len(sorted_loops)} unique causal loops")
            if len(sorted_loops) == 0:
                print("No genuine feedback loops found - this is often correct")
        
        return sorted_loops
    
    def _deduplicate_chains(self, chains: List[Dict]) -> List[Dict]:
        """Remove duplicate chains using normalized field names"""
        seen_sequences = set()
        unique_chains = []
        
        for chain in chains:
            sequence = chain.get('entity_sequence', [])
            if sequence:
                sequence_tuple = tuple(sequence)
                if sequence_tuple not in seen_sequences:
                    seen_sequences.add(sequence_tuple)
                    unique_chains.append(chain)
        
        return unique_chains
    
    def _deduplicate_loops(self, loops: List[Dict]) -> List[Dict]:
        """Remove duplicate loops using normalized field names"""
        seen_loops = set()
        unique_loops = []
        
        for loop in loops:
            participants = loop.get('participating_entities', [])
            if participants:
                participants_tuple = tuple(sorted(participants))
                if participants_tuple not in seen_loops:
                    seen_loops.add(participants_tuple)
                    unique_loops.append(loop)
        
        return unique_loops
    
    def _merge_chain_results(self, llm_chains: List[Dict], auto_chains: List[Dict]) -> List[Dict]:
        """Merge chains from different sources"""
        # Normalize LLM chains first
        normalized_llm_chains = self.normalizer.normalize_chains_batch(llm_chains)
        
        # Combine and deduplicate
        all_chains = normalized_llm_chains + auto_chains
        return self._deduplicate_chains(all_chains)

    def _merge_loop_results(self, llm_loops: List[Dict], auto_loops: List[Dict]) -> List[Dict]:
        """Merge loops from different sources with robust validation"""
        # Normalize LLM loops first
        normalized_llm_loops = self.normalizer.normalize_loops_batch(llm_loops)
    
        # CRITICAL: Validate LLM loops against actual relationships
        validated_llm_loops = []
    
        if normalized_llm_loops and hasattr(self, '_current_relationships'):
            # Build relationship lookup for validation
            relationship_edges = set()
            for rel in self._current_relationships:
                cause = rel.get('cause_entity_id', '')
                effect = rel.get('effect_entity_id', '')
                if cause and effect:
                    relationship_edges.add((cause, effect))
        
            # Validate each LLM loop
            for loop in normalized_llm_loops:
                participants = loop.get('participating_entities', [])
                if len(participants) >= 3:  # Minimum 3 nodes
                    # Check if all edges in the loop actually exist
                    valid_loop = True
                    for i in range(len(participants)):
                        from_node = participants[i]
                        to_node = participants[(i + 1) % len(participants)]
                        if (from_node, to_node) not in relationship_edges:
                            valid_loop = False
                            break
                
                    if valid_loop:
                        validated_llm_loops.append(loop)
    
        # Combine validated LLM loops with auto-generated loops
        all_loops = validated_llm_loops + auto_loops
        return self._deduplicate_loops(all_loops)
    
    
    def _infer_domain_config(self, entities: List[Dict]) -> Dict:
        """Infer domain configuration from entities"""
        if not entities:
            return self._get_default_domain_config()
        
        # Try to detect entity type from IDs
        entity_ids = [entity.get('id', entity.get('entity_id', '')) for entity in entities if entity.get('id') or entity.get('entity_id')]
        
        if entity_ids:
            first_id = entity_ids[0]
            if '-' in first_id:
                prefix = first_id.split('-')[0]
            else:
                prefix = 'EN'
        else:
            prefix = 'EN'
        
        # Map prefixes to domain types
        prefix_mapping = {
            'PP': {'domain_name': 'Problem Analysis', 'entity_type': 'pain_points'},
            'RF': {'domain_name': 'Risk Analysis', 'entity_type': 'risk_factors'},
            'BT': {'domain_name': 'Process Analysis', 'entity_type': 'bottlenecks'},
            'IS': {'domain_name': 'Issue Analysis', 'entity_type': 'issues'},
            'CH': {'domain_name': 'Challenge Analysis', 'entity_type': 'challenges'},
            'BR': {'domain_name': 'Barrier Analysis', 'entity_type': 'barriers'},
            'EN': {'domain_name': 'Entity Analysis', 'entity_type': 'entities'}
        }
        
        domain_info = prefix_mapping.get(prefix, {'domain_name': 'System Analysis', 'entity_type': 'entities'})
        
        return {
            'domain_name': domain_info['domain_name'],
            'entity_type': domain_info['entity_type'],
            'entity_prefix': prefix,
            'scope': 'System-wide analysis',
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _get_default_domain_config(self) -> Dict:
        """Default domain configuration"""
        return {
            'domain_name': 'General System Analysis',
            'entity_type': 'entities',
            'entity_prefix': 'EN',
            'scope': 'General analysis',
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _generate_chain_description(self, sequence: List[str], domain_config: Dict) -> str:
        """Generate domain-aware chain description"""
        entity_type = domain_config.get('entity_type', 'entities')
        if len(sequence) < 3:
            return f"Short causal chain in {entity_type}"
        
        return f"Causal chain in {entity_type}: {sequence[0]} cascades through {len(sequence)-2} intermediate {entity_type} to ultimately result in {sequence[-1]}"
    
    def _generate_loop_description(self, cycle: List[str], domain_config: Dict) -> str:
        """Generate domain-aware loop description"""
        entity_type = domain_config.get('entity_type', 'entities')
        unique_nodes = cycle[:-1]  # Remove duplicate end node
        return f"Reinforcing feedback loop in {entity_type} where {' → '.join(unique_nodes[:3])} creates a self-perpetuating cycle"
    
    def _identify_break_points(self, cycle: List[str]) -> List[str]:
        """Identify potential intervention points"""
        unique_nodes = cycle[:-1]
        return unique_nodes[:min(3, len(unique_nodes))]
    
    def _load_prompt(self, prompt_path: str) -> str:
        """Load analysis prompt"""
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return self._get_default_prompt()
    
    def _load_schema(self, schema_path: str) -> Dict:
        """Load JSON schema"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _get_default_prompt(self) -> str:
        """Fallback prompt"""
        return """
        You are a causality analysis expert. Analyze the provided entities to identify:
        1. Direct causal relationships between entities
        2. Causal chains where problems connect sequentially 
        3. Causal loops where problems create reinforcing cycles
        4. Root causes and leverage points for intervention
        
        Return comprehensive JSON with causal_relationships, causal_chains, causal_loops, 
        root_causes, leverage_points, and network_metrics.
        """
    
    def _prepare_analysis_request(self, entities: List[Dict], domain_config: Dict, prompt: str, schema: Dict) -> str:
        """Prepare the complete analysis request"""
        entity_type = domain_config.get('entity_type', 'entities')
        entities_text = "\n".join([
            f"- {entity.get('id', 'UNKNOWN')}: {entity.get('title', entity.get('name', 'No title'))} - {entity.get('description', 'No description')}"
            for entity in entities
        ])
        
        schema_text = json.dumps(schema, indent=2) if schema else ""
        
        domain_context = f"""
Domain: {domain_config.get('domain_name', 'Unknown')}
Entity Type: {entity_type}
Entity Prefix: {domain_config.get('entity_prefix', 'EN')}
Scope: {domain_config.get('scope', 'General analysis')}
"""
        
        return f"""
{prompt}

## Domain Context:
{domain_context}

## {entity_type.title()} to Analyze:
{entities_text}

## Expected JSON Schema:
{schema_text}

Analyze these {entity_type} and return the complete causality network in JSON format.
Focus on identifying strong causal relationships, chains, and loops within this {domain_config.get('domain_name', 'system')}.
"""
    
    async def _get_llm_analysis(self, analysis_request: str, verbose: bool = False) -> str:
        """Get analysis from LLM using your existing framework"""
        if verbose:
            print(f"=== PROMPT BEING SENT TO MODEL ===")
            print(f"Prompt length: {len(analysis_request)} characters")
            print(f"First 500 characters:")
            print(analysis_request[:500])
            print(f"Last 500 characters:")
            print(analysis_request[-500:])
            print("=" * 50)
        
        try:
            # Use your existing framework pattern
            config = Configuration(
                model=self.model,
                prompt="",  # We'll send the prompt directly in messages
                max_loops=100,
                temperature=self.temperature
            ).__dict__
            
            model = init_model({"configurable": config})
            messages = [HumanMessage(content=analysis_request)]
            
            response = cast(AIMessage, await model.ainvoke(messages))
            result = response.content
            
            if verbose:
                print(f"Model: {self.model}")
                print(f"LLM response length: {len(result)} characters")
            
            return result
        
        except Exception as e:
            if verbose:
                print(f"Error calling LLM: {str(e)}")
            raise
    
    def _parse_llm_response(self, raw_result: str, verbose: bool = False) -> Dict:
        """Parse LLM response with fallback strategies"""
        
        if verbose:
            print(f"=== RAW LLM RESPONSE ===")
            print(f"Response length: {len(raw_result)} characters")
            print(f"First 200 characters:")
            print(raw_result[:200])
            print("=" * 50)
        
        # Strategy 1: Direct JSON parsing
        try:
            cleaned = raw_result.strip()
            if cleaned.startswith('```json'):
                cleaned = re.sub(r'^```json\s*', '', cleaned)
            if cleaned.endswith('```'):
                cleaned = re.sub(r'\s*```$', '', cleaned)
            
            result = json.loads(cleaned)
            if verbose:
                print("✓ Successfully parsed JSON directly")
            return result
        except json.JSONDecodeError as e:
            if verbose:
                print(f"Direct JSON parsing failed: {e}")
        
        # Strategy 2: Extract JSON block
        try:
            json_match = re.search(r'\{.*\}', raw_result, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if verbose:
                    print("✓ Successfully extracted and parsed JSON block")
                return result
        except json.JSONDecodeError:
            if verbose:
                print("JSON block extraction failed")
        
        # Strategy 3: Return empty structure
        if verbose:
            print("Using fallback empty structure")
        return self._get_empty_result()
    
    def _perform_network_analysis(self, results: Dict, entities: List[Dict], verbose: bool = False) -> Dict:
        """Perform network analysis"""
        
        # Build network graph
        G = nx.DiGraph()
        
        # Add nodes
        for entity in entities:
            entity_id = entity.get('id', entity.get('entity_id', ''))
            if entity_id:
                G.add_node(entity_id)
        
        # Add edges from relationships (using normalized field names)
        for rel in results.get('causal_relationships', []):
            cause = rel.get('cause_entity_id', '')
            effect = rel.get('effect_entity_id', '')
            if cause and effect:
                G.add_edge(cause, effect, weight=rel.get('confidence', 0))
        
        # Calculate network metrics
        network_metrics = {
            'total_entities_analyzed': len(entities),
            'total_relationships': len(results.get('causal_relationships', [])),
            'total_chains': len(results.get('causal_chains', [])),
            'total_loops': len(results.get('causal_loops', [])),
            'network_density': nx.density(G) if G.number_of_nodes() > 1 else 0,
            'average_confidence': 0,
            'longest_chain': 0,
            'strongest_loop': 0
        }
        
        if results.get('causal_relationships'):
            avg_confidence = sum(rel.get('confidence', 0) for rel in results['causal_relationships']) / len(results['causal_relationships'])
            network_metrics['average_confidence'] = round(avg_confidence, 1)
        
        if results.get('causal_chains'):
            longest_chain = max((len(chain.get('entity_sequence', [])) for chain in results['causal_chains']), default=0)
            network_metrics['longest_chain'] = longest_chain
        
        if results.get('causal_loops'):
            strongest_loop = max((loop.get('loop_strength', 0) for loop in results['causal_loops']), default=0)
            network_metrics['strongest_loop'] = strongest_loop
        
        if verbose:
            print(f"Network analysis: {network_metrics}")
        
        return {'network_metrics': network_metrics}
    
    def _generate_analysis_metadata(self, results: Dict) -> Dict:
        """Generate analysis metadata"""
        return {
            'analysis_approach': ['explicit_causality', 'semantic_causality', 'domain_knowledge', 'temporal_logic'],
            'data_sources': ['Entity descriptions', 'Domain expertise', 'Logical inference'],
            'limitations': [
                'Analysis based on description inference rather than empirical data',
                'Causal strength estimates are qualitative assessments',
                'Time delays are approximate based on typical system dynamics',
                'Limited to entities identified in source data'
            ],
            'validation_methods': ['Cross-reference checking', 'Logical consistency', 'Domain pattern matching', 'Field normalization']
        }
    
    def _get_empty_result(self, domain_config: Dict = None) -> Dict:
        """Return empty result structure"""
        return {
            'domain_context': domain_config or self._get_default_domain_config(),
            'causal_relationships': [],
            'causal_chains': [],
            'causal_loops': [],
            'root_causes': [],
            'ultimate_effects': [],
            'leverage_points': [],
            'network_metrics': {
                'total_entities_analyzed': 0,
                'total_relationships': 0,
                'total_chains': 0,
                'total_loops': 0,
                'network_density': 0,
                'average_confidence': 0,
                'longest_chain': 0,
                'strongest_loop': 0
            },
            'analysis_metadata': {
                'analysis_approach': ['explicit_causality', 'semantic_causality'],
                'data_sources': ['Entity descriptions'],
                'limitations': ['No data provided for analysis'],
                'validation_methods': ['Basic validation']
            }
        }
    
    def _print_analysis_summary(self, results: Dict):
        """Print analysis summary"""
        print("\n=== CAUSALITY ANALYSIS SUMMARY ===")
        domain = results.get('domain_context', {})
        print(f"Domain: {domain.get('domain_name', 'Unknown')}")
        print(f"Entity Type: {domain.get('entity_type', 'Unknown')}")
        print(f"Scope: {domain.get('scope', 'Unknown')}")
        print(f"\nResults:")
        print(f"  Causal Relationships: {len(results.get('causal_relationships', []))}")
        print(f"  Causal Chains: {len(results.get('causal_chains', []))}")
        print(f"  Causal Loops: {len(results.get('causal_loops', []))}")
        print(f"  Root Causes: {len(results.get('root_causes', []))}")
        print(f"  Leverage Points: {len(results.get('leverage_points', []))}")
        
        metrics = results.get('network_metrics', {})
        print(f"\nNetwork Metrics:")
        print(f"  Entities Analyzed: {metrics.get('total_entities_analyzed', 0)}")
        print(f"  Network Density: {metrics.get('network_density', 0):.3f}")
        print(f"  Average Confidence: {metrics.get('average_confidence', 0)}%")
        print(f"  Longest Chain: {metrics.get('longest_chain', 0)} entities")
        print(f"  Strongest Loop: {metrics.get('strongest_loop', 0)}%")


# Main interface functions for backward compatibility and easy integration
async def analyze_entity_causality(entities: List[Dict] = None,
                                 extraction_results: Dict = None,
                                 domain_config: Dict = None,
                                 model: str = None,
                                 output_dir: str = None,
                                 prompts_dir: str = "../prompts",
                                 prompt_filename: str = 'causality_prompt.txt',
                                 config_filename: str = 'causality_config.json',
                                 schema_filename: str = 'causality_schema.json',
                                 temperature: float = 0.1,
                                 verbose: bool = True,
                                 **kwargs) -> Dict:
    """
    Universal causality analysis function
    Works with any entity type: pain points, risk factors, bottlenecks, etc.
    """
    
    # Handle different input formats for backward compatibility
    if extraction_results is not None and entities is None:
        if isinstance(extraction_results, dict):
            entities = (extraction_results.get('pain_points', []) or 
                       extraction_results.get('entities', []) or
                       extraction_results.get('results', []))
        else:
            entities = []
    
    if not entities:
        if verbose:
            print("No entities provided for causality analysis")
        return {"causal_relationships": [], "causal_chains": [], "causal_loops": []}
    
    # Use provided model or default
    if model is None:
        model = "gpt-4"
    
    if verbose:
        print(f"Initializing universal causality analyzer with model: {model}")
        print(f"Prompts directory: {prompts_dir}")
    
    # Initialize analyzer with prompts directory
    analyzer = UniversalCausalityAnalyzer(
        model=model,
        output_dir=output_dir or "causality_analysis",
        temperature=temperature,
        prompts_dir=prompts_dir
    )
    
    # Perform analysis
    results = await analyzer.analyze_causality(
        entities=entities,
        domain_config=domain_config,
        prompt_filename=prompt_filename,
        config_filename=config_filename,
        schema_filename=schema_filename,
        verbose=verbose
    )
    
    # Save results if output directory specified
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'causality_analysis.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"Results saved to: {output_file}")
        except Exception as e:
            if verbose:
                print(f"Failed to save results: {e}")
    
    return results


# Backward compatibility aliases
async def analyze_pain_point_causality(*args, **kwargs) -> Dict:
    """Backward compatibility alias for pain point analysis"""
    return await analyze_entity_causality(*args, **kwargs)


async def analyze_risk_factor_causality(risk_factors: List[Dict] = None, **kwargs) -> Dict:
    """Analyze causality for risk factors"""
    domain_config = {
        'domain_name': 'Risk Analysis',
        'entity_type': 'risk_factors',
        'entity_prefix': 'RF',
        'scope': 'Risk assessment analysis',
        'analysis_date': datetime.now().strftime('%Y-%m-%d')
    }
    return await analyze_entity_causality(entities=risk_factors, domain_config=domain_config, **kwargs)


async def analyze_bottleneck_causality(bottlenecks: List[Dict] = None, **kwargs) -> Dict:
    """Analyze causality for process bottlenecks"""
    domain_config = {
        'domain_name': 'Process Analysis',
        'entity_type': 'bottlenecks',
        'entity_prefix': 'BT',
        'scope': 'Process bottleneck analysis',
        'analysis_date': datetime.now().strftime('%Y-%m-%d')
    }
    return await analyze_entity_causality(entities=bottlenecks, domain_config=domain_config, **kwargs)


# Synchronous wrappers for non-async environments
def analyze_entity_causality_sync(*args, **kwargs) -> Dict:
    """Synchronous wrapper for entity causality analysis"""
    return asyncio.run(analyze_entity_causality(*args, **kwargs))

def analyze_pain_point_causality_sync(*args, **kwargs) -> Dict:
    """Synchronous wrapper for pain point causality analysis"""
    return asyncio.run(analyze_pain_point_causality(*args, **kwargs))

def analyze_risk_factor_causality_sync(*args, **kwargs) -> Dict:
    """Synchronous wrapper for risk factor causality analysis"""
    return asyncio.run(analyze_risk_factor_causality(*args, **kwargs))

def analyze_bottleneck_causality_sync(*args, **kwargs) -> Dict:
    """Synchronous wrapper for bottleneck causality analysis"""
    return asyncio.run(analyze_bottleneck_causality(*args, **kwargs))


# Example usage and testing
if __name__ == "__main__":
    # Example entities for testing
    sample_entities = [
        {
            "id": "PP-001",
            "title": "High Technology Costs",
            "description": "Advanced healthcare technology is expensive and creates affordability barriers for widespread adoption",
            "themes": ["cost", "technology", "accessibility"]
        },
        {
            "id": "PP-002", 
            "title": "Staff Training Gaps",
            "description": "Healthcare workers lack adequate training on new technologies, leading to underutilization",
            "themes": ["training", "workforce", "technology"]
        },
        {
            "id": "PP-003",
            "title": "Patient Resistance",
            "description": "Patients are reluctant to adopt new technologies due to complexity and unfamiliarity",
            "themes": ["adoption", "complexity", "awareness"]
        }
    ]
    
    # Test the analyzer
    async def test_analysis():
        print("Testing Universal Causality Analyzer...")
        
        results = await analyze_entity_causality(
            entities=sample_entities,
            verbose=True
        )
        
        print("\nTest completed!")
        print(f"Generated {len(results.get('causal_relationships', []))} relationships")
        print(f"Generated {len(results.get('causal_chains', []))} chains")
        print(f"Generated {len(results.get('causal_loops', []))} loops")
    
    # Uncomment to run test
    # asyncio.run(test_analysis())
