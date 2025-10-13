#!/usr/bin/env python3
import asyncio
import json
import os
import time
from collections import defaultdict
from typing import Dict, Any, List, Optional, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from core.configuration import Configuration
from core.utils import init_model


class StakeholderAnalyzer:
    """Class for analyzing stakeholders, factors, pain points, and relationships from text"""

    def __init__(
            self,
            model: str = "openai/gpt-4.1",
            output_dir: str = "output",
            temperature: float = 0.1,
            prompts_dir: str = "prompts" # New parameter for prompts directory
    ):
        load_dotenv()
        self.model = model
        self.output_dir = output_dir
        self.temperature = temperature
        self.prompts_dir = prompts_dir # Store prompts directory
        
        self.extraction_schema = self._load_extraction_schema()
        self.prompt_template = self._load_prompt_template()

        os.makedirs(output_dir, exist_ok=True)
        # Ensure prompts directory exists, or at least the files are accessible
        if not os.path.exists(self.prompts_dir):
            print(f"Warning: Prompts directory '{self.prompts_dir}' not found. Ensure prompt and schema files are accessible.")


    def _load_extraction_schema(self) -> Dict[str, Any]:
        """Loads the extraction schema from a JSON file."""
        schema_path = os.path.join(self.prompts_dir, 'extraction_schema.json')
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Extraction schema file not found at: {schema_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding extraction schema JSON from {schema_path}: {e}")

    def _load_prompt_template(self) -> str:
        """Loads the prompt template from a text file."""
        prompt_path = os.path.join(self.prompts_dir, 'extraction_prompt.txt')
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template file not found at: {prompt_path}")

    def verify_files_loaded(self):
        """Debug method to verify the loaded files contain expected content"""
        print("\n=== FILE VERIFICATION ===")
        
        # Check schema file
        schema_path = os.path.join(self.prompts_dir, 'extraction_schema.json')
        print(f"Schema file path: {schema_path}")
        print(f"Schema file exists: {os.path.exists(schema_path)}")
        
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Schema file size: {len(content)} characters")
            print(f"Contains 'challenge_relevance_capacity': {'challenge_relevance_capacity' in content}")
            print(f"Contains 'thematic_analysis': {'thematic_analysis' in content}")
            print(f"Contains 'challenge_readiness': {'challenge_readiness' in content}")
        
        # Check prompt file
        prompt_path = os.path.join(self.prompts_dir, 'extraction_prompt.txt')
        print(f"\nPrompt file path: {prompt_path}")
        print(f"Prompt file exists: {os.path.exists(prompt_path)}")
        
        if os.path.exists(prompt_path):
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Prompt file size: {len(content)} characters")
            print(f"Contains 'challenge_relevance_capacity': {'challenge_relevance_capacity' in content}")
            print(f"Contains 'thematic_analysis': {'thematic_analysis' in content}")
            
        # Check loaded schema structure
        print(f"\n=== LOADED SCHEMA STRUCTURE ===")
        if 'properties' in self.extraction_schema and 'stakeholders' in self.extraction_schema['properties']:
            stakeholder_props = self.extraction_schema['properties']['stakeholders'].get('items', {}).get('properties', {})
            print(f"Stakeholder properties: {list(stakeholder_props.keys())}")
        
        top_level_keys = list(self.extraction_schema.get('properties', {}).keys())
        print(f"Top-level schema properties: {top_level_keys}")
        print("="*50)

    def print_schema_sample(self, max_chars=500):
        """Print a sample of the loaded schema"""
        schema_str = json.dumps(self.extraction_schema, indent=2)
        print(f"\n=== SCHEMA SAMPLE (first {max_chars} chars) ===")
        print(schema_str[:max_chars])
        if len(schema_str) > max_chars:
            print("...")
        print("="*50)

    def print_prompt_sample(self, max_chars=500):
        """Print a sample of the loaded prompt"""
        print(f"\n=== PROMPT SAMPLE (first {max_chars} chars) ===")
        print(self.prompt_template[:max_chars])
        if len(self.prompt_template) > max_chars:
            print("...")
        print("="*50)

    def _create_config(self, max_loops: int = 100) -> Dict[str, Any]:
        """Create the configuration for the model invocation."""
        return Configuration(
            model=self.model,
            prompt=self.prompt_template, # Use loaded template
            max_loops=max_loops,
            temperature=self.temperature
        ).__dict__

    async def analyze_text(
            self,
            text: str,
            doc_id: Optional[str] = None,
            doc_name: Optional[str] = None,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyzes a single text document to extract stakeholder, factor, pain point,
        and relationship information.
        """
        start_time = time.time()
        doc_id = doc_id or f"doc_{int(time.time())}"
        doc_name = doc_name or f"Document {doc_id}"

        if verbose:
            print(f"\nAnalyzing document: {doc_name}\nText length: {len(text)} characters")

        config = self._create_config()

        try:
            extraction_result = await self._extract_information(text, doc_id, doc_name, config)
            if verbose:
                self._print_extraction_summary(extraction_result, doc_name)

            output_file = os.path.join(self.output_dir, f"{doc_id}_analysis.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2)

            if verbose:
                print(f"Analysis saved to {output_file}")
                print(f"Processing time: {time.time() - start_time:.2f} seconds")
            return extraction_result

        except Exception as e:
            error_msg = f"Error analyzing document '{doc_name}': {e}"
            if verbose:
                print(error_msg)
            return {"document_info": {"id": doc_id, "name": doc_name}, "error": error_msg}

    async def analyze_multiple_texts(
            self,
            texts: List[Dict[str, str]],
            verbose: bool = True,
            aggregate: bool = True
    ) -> Dict[str, Any]:
        """
        Analyzes multiple text documents concurrently and optionally aggregates their results.
        """
        if verbose:
            print(f"Analyzing {len(texts)} documents")

        start_time = time.time()
        
        tasks = []
        for i, text_item in enumerate(texts):
            text = text_item.get('text', '')
            doc_id = text_item.get('id', f"doc_{i + 1}_{int(time.time())}")
            doc_name = text_item.get('name', f"Document {i + 1}")
            tasks.append(self.analyze_text(text=text, doc_id=doc_id, doc_name=doc_name, verbose=False)) # Set verbose to False for individual runs to avoid excessive output

        individual_results_list = await asyncio.gather(*tasks)
        
        results_map = {}
        for result in individual_results_list:
            if 'document_info' in result and result['document_info'].get('id'):
                results_map[result['document_info']['id']] = result
            elif result.get('error') and 'document_info' in result: # Handle error results with doc info
                 results_map[result['document_info'].get('id', f"error_doc_{int(time.time())}")] = result
            else: # Fallback for completely malformed results
                results_map[f"unknown_doc_{int(time.time())}"] = result


        if aggregate and results_map:
            if verbose:
                print("\nAggregating results across all documents...")
            aggregated_results = self._aggregate_results(results_map)
            output_file = os.path.join(self.output_dir, f"aggregated_analysis_{int(time.time())}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(aggregated_results, f, indent=2)

            if verbose:
                print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
                print(f"Aggregated results saved to {output_file}")
                self._print_aggregation_summary(aggregated_results)
            return {"individual_results": results_map, "aggregated_results": aggregated_results}

        if verbose:
            print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
        return {"individual_results": results_map}


    async def _extract_information(
            self,
            text: str,
            doc_id: str,
            doc_name: str,
            config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extracts information from a single text using the configured LLM.
        """
        response_content: str = ""
        try:
            # Format the prompt with the schema and topic
            prompt_text = config['prompt'].format(
                info=json.dumps(self.extraction_schema, indent=2),
                topic=text
            )
            model = init_model({"configurable": config})
            messages = [HumanMessage(content=prompt_text)]
            
            response = cast(AIMessage, await model.ainvoke(messages))
            response_content = response.content

            # Robust JSON extraction
            if "```json" in response_content:
                json_string = response_content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in response_content:
                json_string = response_content.split("```", 1)[1].split("```", 1)[0].strip()
            else:
                json_string = response_content.strip() # Assume raw JSON if no fences

            result_json = json.loads(json_string)

            # UPDATED: Include the new sections
            return {
                'document_info': {'id': doc_id, 'name': doc_name, 'content_length': len(text)},
                'stakeholders': result_json.get('stakeholders', []),
                'factors': result_json.get('factors', []),
                'pain_points': result_json.get('pain_points', []),
                'relationships': result_json.get('relationships', []),
                'thematic_analysis': result_json.get('thematic_analysis', {}),  # NEW
                'challenge_readiness': result_json.get('challenge_readiness', {})  # NEW
            }

        except json.JSONDecodeError as e:
            return {
                'document_info': {'id': doc_id, 'name': doc_name, 'content_length': len(text)},
                'error': f"Error parsing JSON response: {e}",
                'raw_response_content': response_content
            }
        except Exception as e:
            return {
                'document_info': {'id': doc_id, 'name': doc_name, 'content_length': len(text)},
                'error': f"Error extracting information: {e}",
                'raw_response_content': response_content # Include raw content for general errors too
            }

    def _aggregate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates extracted information from multiple documents.
        """
        stakeholder_data = defaultdict(lambda: {
            'categories': defaultdict(lambda: {'count': 0, 'confidence_sum': 0}),
            'roles': defaultdict(int),
            'hierarchy_levels': defaultdict(int),
            'documents': [],
            'total_mentions': 0,
            # NEW: Aggregate challenge-ready fields
            'challenge_relevance_capacity': defaultdict(int),
            'cross_theme_connections': defaultdict(int),
            'decision_authority': defaultdict(int),
            'resource_control': defaultdict(int)
        })
        all_factors = []
        all_pain_points = []
        all_relationships = []
        pain_point_ids = set()
        relationship_ids = set()
        id_counter_pp = 1
        id_counter_rel = 1

        # NEW: Aggregate thematic analysis and challenge readiness
        all_themes = []  # Changed to list to handle dict themes
        all_cross_cutting_themes = []  # Changed to list to handle dict themes
        all_suggested_challenges = []
        aggregated_challenge_readiness = {}

        for doc_id, result in results.items():
            if result.get('error'):
                continue

            doc_info = result.get('document_info', {})
            doc_name = doc_info.get('name', 'Unknown')

            # Aggregate Stakeholders
            for stakeholder in result.get('stakeholders', []):
                name = stakeholder.get('name')
                if not name:
                    continue

                sd = stakeholder_data[name]
                if {'id': doc_id, 'name': doc_name} not in sd['documents']:
                    sd['documents'].append({'id': doc_id, 'name': doc_name})
                sd['total_mentions'] += 1

                category = stakeholder.get('category')
                confidence = stakeholder.get('confidence', 0)
                if category:
                    sd['categories'][category]['count'] += 1
                    sd['categories'][category]['confidence_sum'] += confidence

                role = stakeholder.get('role')
                if role:
                    sd['roles'][role] += 1

                hierarchy_level = stakeholder.get('hierarchy_level', 'Meso')
                if hierarchy_level:
                    sd['hierarchy_levels'][hierarchy_level] += 1

                # NEW: Aggregate challenge-ready fields
                for field in ['challenge_relevance_capacity', 'cross_theme_connections', 'decision_authority', 'resource_control']:
                    value = stakeholder.get(field)
                    if value:
                        sd[field][value] += 1

            # Aggregate Factors (ensure uniqueness by Label)
            for factor in result.get('factors', []):
                if factor.get('Label') and not any(f.get('Label') == factor['Label'] for f in all_factors):
                    factor['source_document'] = {'id': doc_id, 'name': doc_name}
                    all_factors.append(factor)

            # Aggregate Pain Points (assign unique IDs if not present or duplicate)
            for pain_point in result.get('pain_points', []):
                original_id = pain_point.get('id')
                if not original_id or original_id in pain_point_ids:
                    # Generate a new unique ID
                    while f"PP-{id_counter_pp:03d}" in pain_point_ids:
                        id_counter_pp += 1
                    pain_point['id'] = f"PP-{id_counter_pp:03d}"
                    id_counter_pp += 1
                pain_point_ids.add(pain_point['id']) # Add the (potentially new) ID

                pain_point['source_document'] = {'id': doc_id, 'name': doc_name}
                pain_point.setdefault('hierarchy_level', self._determine_hierarchy_from_description(pain_point.get('description', '')))
                pain_point.setdefault('affected_stakeholders', [])
                pain_point.setdefault('causing_stakeholders', [])
                pain_point.setdefault('stakeholder_impact', {})
                pain_point.setdefault('severity', 'Medium')
                all_pain_points.append(pain_point)

            # Aggregate Relationships (assign unique IDs if not present or duplicate)
            for relationship in result.get('relationships', []):
                original_id = relationship.get('id')
                if not original_id or original_id in relationship_ids:
                    # Generate a new unique ID
                    while f"REL-{id_counter_rel:03d}" in relationship_ids:
                        id_counter_rel += 1
                    relationship['id'] = f"REL-{id_counter_rel:03d}"
                    id_counter_rel += 1
                relationship_ids.add(relationship['id']) # Add the (potentially new) ID

                relationship['source_document'] = {'id': doc_id, 'name': doc_name}
                all_relationships.append(relationship)

            # NEW: Aggregate thematic analysis
            thematic_analysis = result.get('thematic_analysis', {})
            if thematic_analysis:
                # Handle identified_themes (list of dicts)
                identified_themes = thematic_analysis.get('identified_themes', [])
                all_themes.extend(identified_themes)
                
                # Handle cross_cutting_themes (list of dicts)
                cross_cutting_themes = thematic_analysis.get('cross_cutting_themes', [])
                all_cross_cutting_themes.extend(cross_cutting_themes)

            # NEW: Aggregate challenge readiness
            challenge_readiness = result.get('challenge_readiness', {})
            if challenge_readiness:
                suggested_challenges = challenge_readiness.get('suggested_challenges', [])
                all_suggested_challenges.extend(suggested_challenges)

        # Finalize aggregated stakeholder data
        aggregated_stakeholders = []
        for name, data in stakeholder_data.items():
            most_common_category = max(data['categories'], key=lambda k: data['categories'][k]['count']) if data['categories'] else None
            avg_confidence = (data['categories'][most_common_category]['confidence_sum'] / data['categories'][most_common_category]['count']) if most_common_category and data['categories'][most_common_category]['count'] > 0 else 0

            most_common_role = max(data['roles'], key=data['roles'].get) if data['roles'] else None
            most_common_hierarchy = max(data['hierarchy_levels'], key=data['hierarchy_levels'].get) if data['hierarchy_levels'] else "Meso"

            # NEW: Get most common values for challenge-ready fields
            challenge_relevance_capacity = max(data['challenge_relevance_capacity'], key=data['challenge_relevance_capacity'].get) if data['challenge_relevance_capacity'] else None
            cross_theme_connections = max(data['cross_theme_connections'], key=data['cross_theme_connections'].get) if data['cross_theme_connections'] else None
            decision_authority = max(data['decision_authority'], key=data['decision_authority'].get) if data['decision_authority'] else None
            resource_control = max(data['resource_control'], key=data['resource_control'].get) if data['resource_control'] else None

            stakeholder_dict = {
                'name': name,
                'category': most_common_category,
                'role': most_common_role,
                'hierarchy_level': most_common_hierarchy,
                'confidence': round(avg_confidence),
                'mentions': data['total_mentions'],
                'documents': data['documents']
            }

            # NEW: Add challenge-ready fields if present
            if challenge_relevance_capacity:
                stakeholder_dict['challenge_relevance_capacity'] = challenge_relevance_capacity
            if cross_theme_connections:
                stakeholder_dict['cross_theme_connections'] = cross_theme_connections
            if decision_authority:
                stakeholder_dict['decision_authority'] = decision_authority
            if resource_control:
                stakeholder_dict['resource_control'] = resource_control

            aggregated_stakeholders.append(stakeholder_dict)

        aggregated_stakeholders.sort(key=lambda x: x.get('mentions', 0), reverse=True)

        # Build ecosystem map
        multi_level_ecosystem = {
            "ecosystemMap": {
                "name": "Multi-Level Ecosystem Analysis",
                "levels": {level: {"nodes": [], "pain_points": []} for level in ["macro", "meso", "micro"]},
                "relationships": all_relationships
            }
        }
        for s in aggregated_stakeholders:
            level = s.get('hierarchy_level', 'meso').lower()
            if level in multi_level_ecosystem["ecosystemMap"]["levels"]:
                multi_level_ecosystem["ecosystemMap"]["levels"][level]["nodes"].append(s)
        for p in all_pain_points:
            level = p.get('hierarchy_level', 'meso').lower()
            if level in multi_level_ecosystem["ecosystemMap"]["levels"]:
                multi_level_ecosystem["ecosystemMap"]["levels"][level]["pain_points"].append(p)

        # NEW: Build aggregated thematic analysis
        aggregated_thematic_analysis = {}
        if all_themes:
            # Extract unique themes by theme_name, keeping the most detailed version
            unique_themes = {}
            for theme in all_themes:
                theme_name = theme.get('theme_name', 'unknown')
                if theme_name not in unique_themes or len(str(theme)) > len(str(unique_themes[theme_name])):
                    unique_themes[theme_name] = theme
            
            aggregated_thematic_analysis['identified_themes'] = list(unique_themes.values())[:8]
        
        if all_cross_cutting_themes:
            # Extract unique cross-cutting themes by theme_name
            unique_cross_themes = {}
            for theme in all_cross_cutting_themes:
                theme_name = theme.get('theme_name', 'unknown')
                if theme_name not in unique_cross_themes:
                    unique_cross_themes[theme_name] = theme
            
            aggregated_thematic_analysis['cross_cutting_themes'] = list(unique_cross_themes.values())[:5]

        # NEW: Build aggregated challenge readiness
        if all_suggested_challenges:
            # Remove duplicates and limit
            unique_challenges = []
            seen_challenges = set()
            for challenge in all_suggested_challenges:
                challenge_id = challenge.get('title', challenge.get('description', ''))
                if challenge_id and challenge_id not in seen_challenges:
                    unique_challenges.append(challenge)
                    seen_challenges.add(challenge_id)
            
            aggregated_challenge_readiness['suggested_challenges'] = unique_challenges[:6]  # Limit to 6 unique challenges

        # Build final result
        result_dict = {
            'total_documents': len(results),
            'stakeholders': aggregated_stakeholders,
            'factors': all_factors,
            'pain_points': all_pain_points,
            'relationships': all_relationships,
            'ecosystem_map': multi_level_ecosystem,
            'stakeholder_validation': {
                'original_count': len(aggregated_stakeholders),
                'validated_count': len(aggregated_stakeholders),
                'duplicates_merged': 0,
                'category_changes': 0
            }
        }

        # NEW: Add thematic analysis and challenge readiness if present
        if aggregated_thematic_analysis:
            result_dict['thematic_analysis'] = aggregated_thematic_analysis
        if aggregated_challenge_readiness:
            result_dict['challenge_readiness'] = aggregated_challenge_readiness

        return result_dict

    def _determine_hierarchy_from_description(self, description: str) -> str:
        """Helper to determine hierarchy level from pain point description."""
        description_lower = description.lower()
        if any(term in description_lower for term in ['system', 'policy', 'government', 'national', 'macroeconomic', 'global']):
            return 'Macro'
        elif any(term in description_lower for term in ['individual', 'resident', 'customer', 'employee', 'user', 'personal', 'team', 'local']):
            return 'Micro'
        else:
            return 'Meso' # Default or organizational/sector level

    def _print_section_summary(self, title: str, items: List[Dict[str, Any]], item_printer_func, limit: int = 3) -> None:
        """Generic helper to print summary sections."""
        if items:
            print(f"\n{title}:")
            for i, item in enumerate(items[:limit]):
                item_printer_func(i, item)

    def _print_stakeholder_item(self, i: int, s: Dict[str, Any]) -> None:
        """Prints a single stakeholder's summary."""
        print(f"{i + 1}. {s.get('name')} ({s.get('category')}) - {s.get('confidence')}% confidence")

    def _print_factor_item(self, i: int, f: Dict[str, Any]) -> None:
        """Prints a single factor's summary."""
        print(f"{i + 1}. {f.get('Label')} ({f.get('Type')})")
        desc = f.get('Description', '')
        if desc:
            print(f"   Description: {desc[:100]}{'...' if len(desc) > 100 else ''}")

    def _print_pain_point_item(self, i: int, p: Dict[str, Any]) -> None:
        """Prints a single pain point's summary."""
        affected = p.get('affected_stakeholders', [])
        causing = p.get('causing_stakeholders', [])
        print(f"{i + 1}. {p.get('id')} - {p.get('category')} ({p.get('confidence')}%) - Severity: {p.get('severity', 'N/A')}")
        desc = p.get('description', '')
        if desc:
            print(f"   Description: {desc[:100]}{'...' if len(desc) > 100 else ''}")
        if affected:
            print(f"   Affected stakeholders: {', '.join(affected)}")
        if causing:
            print(f"   Causing stakeholders: {', '.join(causing)}")

    def _print_relationship_item(self, i: int, r: Dict[str, Any]) -> None:
        """Prints a single relationship's summary."""
        print(f"{i + 1}. {r.get('id')}: {r.get('source_stakeholder')} -> {r.get('target_stakeholder')}")
        print(f"   Type: {r.get('relationship_type')} - {r.get('relationship_subtype')} ({r.get('confidence')}%)")
        just = r.get('justification', '')
        if just:
            print(f"   Justification: {just[:100]}{'...' if len(just) > 100 else ''}")

    def _print_extraction_summary(self, result: Dict[str, Any], doc_name: str) -> None:
        """Prints a summary of extracted data for a single document."""
        if result.get('error'):
            print(f"Error processing {doc_name}: {result.get('error')}")
            if result.get('raw_response_content'):
                print(f"Raw Response Content (partial): {result['raw_response_content'][:500]}...")
            return

        stakeholders = result.get('stakeholders', [])
        factors = result.get('factors', [])
        pain_points = result.get('pain_points', [])
        relationships = result.get('relationships', [])
        thematic_analysis = result.get('thematic_analysis', {})
        challenge_readiness = result.get('challenge_readiness', {})

        print(f"\n--- Extraction Summary for '{doc_name}' ---")
        print(f"- {len(stakeholders)} stakeholders extracted")
        print(f"- {len(factors)} factors extracted")
        print(f"- {len(pain_points)} pain points extracted")
        print(f"- {len(relationships)} relationships extracted")
        
        # NEW: Print thematic analysis and challenge readiness info
        if thematic_analysis:
            main_themes = thematic_analysis.get('main_themes', [])
            print(f"- Thematic analysis: {len(main_themes)} main themes")
        
        if challenge_readiness:
            suggested_challenges = challenge_readiness.get('suggested_challenges', [])
            print(f"- Challenge readiness: {len(suggested_challenges)} suggested challenges")

        self._print_section_summary("Top Stakeholders", stakeholders, self._print_stakeholder_item)
        self._print_section_summary("Top Factors", factors, self._print_factor_item)
        self._print_section_summary("Top Pain Points", pain_points, self._print_pain_point_item)
        self._print_section_summary("Top Relationships", relationships, self._print_relationship_item)
        
        # NEW: Print thematic analysis summary
        if thematic_analysis:
            print(f"\nThematic Analysis:")
            main_themes = thematic_analysis.get('main_themes', [])
            if main_themes:
                # Handle both string and dict formats for themes
                theme_strings = []
                for theme in main_themes[:5]:
                    if isinstance(theme, dict):
                        theme_strings.append(theme.get('name', theme.get('theme', str(theme))))
                    else:
                        theme_strings.append(str(theme))
                print(f"  Main themes: {', '.join(theme_strings)}")
            cross_themes = thematic_analysis.get('cross_cutting_themes', [])
            if cross_themes:
                # Handle both string and dict formats for themes
                cross_theme_strings = []
                for theme in cross_themes[:3]:
                    if isinstance(theme, dict):
                        cross_theme_strings.append(theme.get('name', theme.get('theme', str(theme))))
                    else:
                        cross_theme_strings.append(str(theme))
                print(f"  Cross-cutting themes: {', '.join(cross_theme_strings)}")
        
        # NEW: Print challenge readiness summary
        if challenge_readiness:
            print(f"\nChallenge Readiness:")
            suggested_challenges = challenge_readiness.get('suggested_challenges', [])
            if suggested_challenges:
                for i, challenge in enumerate(suggested_challenges[:2]):
                    title = challenge.get('title', challenge.get('description', ''))
                    print(f"  {i+1}. {title}")
        
        print("-" * 40)


    def _print_aggregation_summary(self, result: Dict[str, Any]) -> None:
        """Prints a summary of aggregated data across multiple documents."""
        stakeholders = result.get('stakeholders', [])
        factors = result.get('factors', [])
        pain_points = result.get('pain_points', [])
        relationships = result.get('relationships', [])
        thematic_analysis = result.get('thematic_analysis', {})
        challenge_readiness = result.get('challenge_readiness', {})

        print("\n--- Aggregated Results Summary ---")
        print(f"Total documents analyzed: {result.get('total_documents', 0)}")
        print(f"- {len(stakeholders)} unique stakeholders")
        print(f"- {len(factors)} unique factors")
        print(f"- {len(pain_points)} unique pain points")
        print(f"- {len(relationships)} unique relationships")
        
        # NEW: Print aggregated thematic analysis info
        if thematic_analysis:
            main_themes = thematic_analysis.get('identified_themes', [])
            print(f"- {len(main_themes)} aggregated identified themes")
        
        if challenge_readiness:
            suggested_challenges = challenge_readiness.get('suggested_challenges', [])
            print(f"- {len(suggested_challenges)} unique suggested challenges")

        # Stakeholder Hierarchy Breakdown
        print("\nStakeholder Hierarchy Breakdown:")
        macro_stakeholders = [s for s in stakeholders if s.get('hierarchy_level') == 'Macro']
        meso_stakeholders = [s for s in stakeholders if s.get('hierarchy_level') == 'Meso']
        micro_stakeholders = [s for s in stakeholders if s.get('hierarchy_level') == 'Micro']
        print(f"- Macro level: {len(macro_stakeholders)} stakeholders")
        print(f"- Meso level: {len(meso_stakeholders)} stakeholders")
        print(f"- Micro level: {len(micro_stakeholders)} stakeholders")

        # Pain Point Severity Breakdown
        if pain_points:
            severity_counts = defaultdict(int)
            for pp in pain_points:
                severity_counts[pp.get('severity', 'Unknown')] += 1
            print("\nPain Point Severity Breakdown:")
            for severity, count in sorted(severity_counts.items()):
                print(f"- {severity}: {count} pain points")

        # Stakeholders Most Affected by Pain Points
        if pain_points and stakeholders:
            stakeholder_pain_count = defaultdict(int)
            for pp in pain_points:
                for stakeholder_name in pp.get('affected_stakeholders', []):
                    stakeholder_pain_count[stakeholder_name] += 1
            print("\nStakeholders Most Affected by Pain Points:")
            for i, (stakeholder, count) in enumerate(sorted(stakeholder_pain_count.items(), key=lambda x: x[1], reverse=True)[:5]):
                print(f"{i + 1}. {stakeholder}: {count} pain points")

        # Relationship Type Breakdown
        if relationships:
            relationship_types = defaultdict(int)
            for rel in relationships:
                relationship_types[rel.get('relationship_type', 'Unknown')] += 1
            print("\nRelationship Type Breakdown:")
            for rel_type, count in sorted(relationship_types.items(), key=lambda x: x[1], reverse=True):
                print(f"- {rel_type}: {count} relationships")

        self._print_section_summary("Top Stakeholders by Mentions", stakeholders, lambda i, s: print(f"{i + 1}. {s.get('name')} ({s.get('category')}) - {s.get('mentions')} mentions"), limit=5)

        critical_pain_points = [pp for pp in pain_points if pp.get('severity') == 'Critical']
        if critical_pain_points:
            print(f"\nCritical Pain Points ({len(critical_pain_points)}):")
            for i, pp in enumerate(critical_pain_points[:3]):
                affected = pp.get('affected_stakeholders', [])
                print(f"{i + 1}. {pp.get('id')} - {pp.get('category')}")
                print(f"   Affected: {', '.join(affected)}")
                desc = pp.get('description', '')
                if desc:
                    print(f"   Description: {desc[:100]}{'...' if len(desc) > 100 else ''}")

        if relationships:
            sorted_relationships = sorted(relationships, key=lambda x: x.get('strength', 0), reverse=True)
            self._print_section_summary("Top Relationships by Strength", sorted_relationships, self._print_relationship_item, limit=5)
        
        # NEW: Print aggregated thematic analysis and challenges
        if thematic_analysis:
            print(f"\nAggregated Thematic Analysis:")
            identified_themes = thematic_analysis.get('identified_themes', [])
            if identified_themes:
                # Extract theme names for display
                theme_names = []
                for theme in identified_themes:
                    if isinstance(theme, dict):
                        theme_names.append(theme.get('theme_name', str(theme)))
                    else:
                        theme_names.append(str(theme))
                print(f"  Identified themes: {', '.join(theme_names)}")
            
            cross_themes = thematic_analysis.get('cross_cutting_themes', [])
            if cross_themes:
                # Extract cross-cutting theme names for display
                cross_theme_names = []
                for theme in cross_themes:
                    if isinstance(theme, dict):
                        cross_theme_names.append(theme.get('theme_name', str(theme)))
                    else:
                        cross_theme_names.append(str(theme))
                print(f"  Cross-cutting themes: {', '.join(cross_theme_names)}")
        
        if challenge_readiness:
            print(f"\nSuggested Challenges for Visualization:")
            suggested_challenges = challenge_readiness.get('suggested_challenges', [])
            for i, challenge in enumerate(suggested_challenges[:3]):
                title = challenge.get('title', challenge.get('description', ''))
                print(f"  {i+1}. {title}")
        
        print("-" * 40)


# Example Usage (assuming core.configuration and core.utils are set up)
async def main():
    # Make sure to create a 'prompts' directory in the same location as stakeholder_analyzer.py
    # and put extraction_prompt.txt and extraction_schema.json inside it.
    analyzer = StakeholderAnalyzer(prompts_dir="./prompts") 

    # --- Example 1: Analyze a single text ---
    print("\n--- Running Single Text Analysis Example ---")
    sample_text_1 = """
    The city council is debating a new zoning law. Local businesses, especially small retailers, are concerned
    about the potential increase in operational costs. Residents in the downtown area are split;
    some support the new law for its environmental benefits, while others fear it will reduce parking.
    A key pain point for small retailers is the proposed tax increase on commercial properties,
    which is causing significant financial strain. The mayor's office is trying to mediate between the
    council and the business community. There's a strong relationship of dependency between local businesses
    and the city's tax revenue.
    """
    single_analysis_result = await analyzer.analyze_text(
        text=sample_text_1,
        doc_name="Zoning Law Debate",
        verbose=True
    )
    # print(json.dumps(single_analysis_result, indent=2)) # Uncomment to see full JSON output


    # --- Example 2: Analyze multiple texts and aggregate ---
    print("\n--- Running Multiple Texts Analysis & Aggregation Example ---")
    sample_texts_for_multi = [
        {
            "id": "doc_A",
            "name": "Community Meeting Minutes",
            "text": """
            During the last community meeting, residents expressed concerns about the lack of public transportation options.
            Elderly citizens are particularly affected by this, as it limits their access to healthcare facilities.
            The local transportation department acknowledged the issue but cited budget constraints as a major factor.
            A new advocacy group formed by affected residents is pressuring the city.
            """
        },
        {
            "id": "doc_B",
            "name": "City Budget Report",
            "text": """
            The city's latest budget report shows a deficit, primarily due to decreased state funding (a macro factor).
            This impacts the transportation department's ability to fund new initiatives, including expanding bus routes.
            The mayor emphasized the need for new revenue streams. Business owners might be willing to invest if it improves
            access for their customers.
            """
        },
        {
            "id": "doc_C",
            "name": "Local News Article",
            "text": """
            A recent article highlighted the growing divide between the city council and advocacy groups over public transport.
            The advocacy group emphasizes that the current transport system negatively impacts the quality of life for low-income families.
            The city council states they are exploring public-private partnerships.
            """
        }
    ]

    multi_analysis_result = await analyzer.analyze_multiple_texts(
        texts=sample_texts_for_multi,
        verbose=True,
        aggregate=True
    )
    # print(json.dumps(multi_analysis_result, indent=2)) # Uncomment to see full JSON output

if __name__ == "__main__":
    asyncio.run(main())
