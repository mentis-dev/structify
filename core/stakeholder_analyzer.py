#!/usr/bin/env python3
import asyncio
import json
import os
import time
from typing import Dict, Any, List, Optional, Union, cast

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from core.configuration import Configuration
from core.utils import init_model


class StakeholderAnalyzer:
    """Class for analyzing stakeholders, factors, and pain points from text"""

    def __init__(
            self,
            model: str = "openai/gpt-4o",
            output_dir: str = "output"
    ):
        # Load environment variables
        load_dotenv()

        self.model = model
        self.output_dir = output_dir
        self.extraction_schema = self._get_extraction_schema()

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def _get_extraction_schema(self) -> Dict[str, Any]:
        """Define the combined extraction schema for stakeholders, factors, and pain points"""
        return {
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
                            },
                            "hierarchy_level": {
                                "type": "string",
                                "description": "Level in the ecosystem hierarchy (Macro, Meso, Micro)."
                            }
                        },
                        "required": ["name", "category", "role", "confidence", "hierarchy_level"]
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
                            },
                            "hierarchy_level": {
                                "type": "string",
                                "description": "Level in the ecosystem hierarchy (Macro, Meso, Micro)."
                            }
                        },
                        "required": ["id", "category", "description", "confidence"]
                    },
                    "description": "List of pain points identified in the text."
                }
            },
            "required": ["stakeholders", "factors", "pain_points"]
        }

    def _get_prompt(self) -> str:
        """Get the analysis prompt template"""
        return (
            "You are an expert analyst tasked with extracting three types of information from the provided text:\n"
            "1. Stakeholders\n"
            "2. Factors\n"
            "3. Pain Points\n\n"

            "FOR STAKEHOLDERS:\n"
            "Identify key organizations and individuals mentioned in the text. For each stakeholder, provide:\n"
            "- name: The name of the stakeholder organization or individual\n"
            "- category: Classify into one of these categories: Regulator, Supplier, Consumer, Competitor, Partner, Influencer, or Internal\n"
            "- role: Brief description of their role or function\n"
            "- confidence: A score from 0-100% indicating your confidence in the classification\n\n"
            "- hierarchy_level: Classify as Macro (national/policy level), Meso (regional/organizational level), or Micro (local/individual level)\n\n"

            "HIERARCHY LEVELS:\n"
            "- Macro: High-level entities with broad influence (government agencies, national regulators, policy makers)\n"
            "- Meso: Mid-level organizations (hospitals, regional NGOs, service providers, industry groups)\n"
            "- Micro: Local actors and individuals (local care centers, community groups, residents)\n\n"

            "STAKEHOLDER CATEGORIES:\n"
            "- Regulator: Government or oversight bodies that create and enforce rules\n"
            "- Supplier: Provides products or services to the organization or industry\n"
            "- Consumer: Receives, uses, or benefits from products or services\n"
            "- Competitor: Other organizations providing similar services or competing for resources\n"
            "- Partner: Organizations working together with shared goals\n"
            "- Influencer: Shapes opinions or decisions without direct authority\n"
            "- Internal: Employees, management, or departments within the organization\n\n"

            "FOR FACTORS:\n"
            "Identify key factors, drivers, and outcomes mentioned in the text. For each factor, provide:\n"
            "- Label: A clear, concise name for the factor (e.g., 'Rising Healthcare Costs')\n"
            "- Type: Classify into one of these types: Indirect Driver, Exogenous Driver, Direct Driver, or Outcome\n"
            "- Tags: Any relevant keywords or tags that help categorize this factor (optional)\n"
            "- Description: A detailed explanation of the factor and how it relates to stakeholders\n\n"

            "FACTOR TYPES:\n"
            "- Indirect Driver: A factor that indirectly influences the system through other factors\n"
            "- Exogenous Driver: An external factor beyond control of system actors (e.g., climate change, demographic shifts)\n"
            "- Direct Driver: A factor that directly causes changes in the system\n"
            "- Outcome: A result or consequence produced by the drivers\n\n"

            "Schema:\n{info}\n\n"
            "Text:\n{topic}\n\n"

            "FOR PAIN POINTS:\n"
            "Identify key pain points or challenges mentioned in the text. For each pain point, provide:\n"
            "- id: A unique identifier in the format PP-XXX (e.g., PP-001)\n"
            "- category: A short category name (e.g., Healthcare Access, Financial Barriers)\n"
            "- description: A brief description of the pain point\n"
            "- confidence: A score from 0-100% indicating your confidence in the identification\n\n"

            "Provide your analysis as properly formatted JSON exactly matching the schema. Include both stakeholders and factors arrays."
        )

    def _create_config(self, max_loops: int = 100) -> Dict[str, Any]:
        """Create the configuration for the model invocation"""
        config_instance = Configuration(
            model=self.model,
            prompt=self._get_prompt(),
            max_loops=max_loops
        )

        # Convert the configuration to a dictionary
        base_config = config_instance.__dict__

        return base_config

    async def analyze_text(
            self,
            text: str,
            doc_id: Optional[str] = None,
            doc_name: Optional[str] = None,
            verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze stakeholders, factors, and pain points in a text

        Args:
            text: Text content to analyze
            doc_id: Optional identifier for the document (generated if not provided)
            doc_name: Optional name for the document (used in output files)
            verbose: Whether to print progress information

        Returns:
            Dictionary containing the analysis results
        """
        start_time = time.time()

        # Generate document ID if not provided
        if not doc_id:
            doc_id = f"doc_{int(time.time())}"

        # Use provided doc_name or default to doc_id
        doc_name = doc_name or f"Document {doc_id}"

        if verbose:
            print(f"\nAnalyzing document: {doc_name}")
            print(f"Text length: {len(text)} characters")

        # Get the configuration
        config = self._create_config()

        try:
            # Extract stakeholders, factors, and pain points
            extraction_result = await self._extract_information(text, doc_id, doc_name, config)

            if verbose:
                self._print_extraction_summary(extraction_result, doc_name)

            # Save the result to a file
            output_file = os.path.join(self.output_dir, f"{doc_id}_analysis.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result, f, indent=2)

            if verbose:
                print(f"Analysis saved to {output_file}")
                elapsed_time = time.time() - start_time
                print(f"Processing time: {elapsed_time:.2f} seconds")

            return extraction_result

        except Exception as e:
            error_msg = f"Error analyzing document: {str(e)}"
            if verbose:
                print(error_msg)

            return {"error": error_msg}

    async def analyze_multiple_texts(
            self,
            texts: List[Dict[str, str]],
            verbose: bool = True,
            aggregate: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze multiple texts and optionally aggregate the results

        Args:
            texts: List of dictionaries, each containing 'text', 'id' (optional), and 'name' (optional)
            verbose: Whether to print progress information
            aggregate: Whether to aggregate results across all texts

        Returns:
            Dictionary containing individual and aggregated results
        """
        if verbose:
            print(f"Analyzing {len(texts)} documents")

        start_time = time.time()
        results = {}

        # Process each text
        for i, text_item in enumerate(texts):
            text = text_item.get('text', '')
            doc_id = text_item.get('id', f"doc_{i + 1}_{int(time.time())}")
            doc_name = text_item.get('name', f"Document {i + 1}")

            if verbose:
                print(f"\nProcessing document {i + 1}/{len(texts)}: {doc_name}")

            result = await self.analyze_text(
                text=text,
                doc_id=doc_id,
                doc_name=doc_name,
                verbose=verbose
            )

            results[doc_id] = result

        # Aggregate results if requested
        if aggregate and results:
            if verbose:
                print("\nAggregating results across all documents...")

            aggregated_results = self._aggregate_results(results)

            # Save aggregated results
            output_file = os.path.join(self.output_dir, f"aggregated_analysis_{int(time.time())}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(aggregated_results, f, indent=2)

            if verbose:
                elapsed_time = time.time() - start_time
                print(f"\nTotal processing time: {elapsed_time:.2f} seconds")
                print(f"Aggregated results saved to {output_file}")
                self._print_aggregation_summary(aggregated_results)

            return {
                "individual_results": results,
                "aggregated_results": aggregated_results
            }

        if verbose:
            elapsed_time = time.time() - start_time
            print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

        return {"individual_results": results}

    async def _extract_information(
            self,
            text: str,
            doc_id: str,
            doc_name: str,
            config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract stakeholders, factors, and pain points from text using the LLM"""
        try:
            # Format the prompt
            prompt_text = config['prompt'].format(
                info=json.dumps(self.extraction_schema, indent=2),
                topic=text
            )

            # Initialize and call the model
            model = init_model({"configurable": config})
            messages = [HumanMessage(content=prompt_text)]
            response = cast(AIMessage, await model.ainvoke(messages))

            # Try to parse the response as JSON
            try:
                # Extract JSON from response
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                result_json = json.loads(content)

                # Add document info to result
                result = {
                    'document_info': {
                        'id': doc_id,
                        'name': doc_name,
                        'content_length': len(text)
                    },
                    'stakeholders': result_json.get('stakeholders', []),
                    'factors': result_json.get('factors', []),
                    'pain_points': result_json.get('pain_points', [])
                }

                return result

            except json.JSONDecodeError as e:
                return {
                    'document_info': {
                        'id': doc_id,
                        'name': doc_name,
                        'content_length': len(text)
                    },
                    'error': f"Error parsing JSON response: {str(e)}",
                    'raw_response': response.content
                }

        except Exception as e:
            return {
                'document_info': {
                    'id': doc_id,
                    'name': doc_name,
                    'content_length': len(text)
                },
                'error': f"Error extracting information: {str(e)}"
            }

    def _aggregate_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate analysis results across multiple documents"""
        # Create structures to store aggregated data
        stakeholder_data = {}
        all_factors = []
        all_pain_points = []
        pain_point_ids = set()
        id_counter = 1

        # Process each document result
        for doc_id, result in results.items():
            if result.get('error'):
                continue

            doc_info = result.get('document_info', {})
            doc_name = doc_info.get('name', 'Unknown')

            # Process stakeholders
            for stakeholder in result.get('stakeholders', []):
                name = stakeholder.get('name')
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
                    'name': doc_name
                })

                # Increment total mentions
                stakeholder_data[name]['total_mentions'] += 1

                # Add category with confidence
                category = stakeholder.get('category')
                confidence = stakeholder.get('confidence', 0)
                if category:
                    if category not in stakeholder_data[name]['categories']:
                        stakeholder_data[name]['categories'][category] = {
                            'count': 0,
                            'confidence_sum': 0
                        }
                    stakeholder_data[name]['categories'][category]['count'] += 1
                    stakeholder_data[name]['categories'][category]['confidence_sum'] += confidence

                # Add role
                role = stakeholder.get('role')
                if role:
                    if role not in stakeholder_data[name]['roles']:
                        stakeholder_data[name]['roles'][role] = 0
                    stakeholder_data[name]['roles'][role] += 1

                # Add hierarchy level
                hierarchy_level = stakeholder.get('hierarchy_level', 'Meso')
                if hierarchy_level:
                    if hierarchy_level not in stakeholder_data[name]['hierarchy_levels']:
                        stakeholder_data[name]['hierarchy_levels'][hierarchy_level] = 0
                    stakeholder_data[name]['hierarchy_levels'][hierarchy_level] += 1

            # Process factors
            for factor in result.get('factors', []):
                # Check if this is a new unique factor
                is_new = True
                label = factor.get('Label')

                for existing in all_factors:
                    if existing.get('Label') == label:
                        is_new = False
                        break

                if is_new and label:
                    # Add document reference
                    factor['source_document'] = {
                        'id': doc_id,
                        'name': doc_name
                    }
                    all_factors.append(factor)

            # Process pain points
            for pain_point in result.get('pain_points', []):
                # Ensure pain point has a valid ID
                if not pain_point.get('id') or pain_point.get('id') in pain_point_ids:
                    new_id = f"PP-{id_counter:03d}"
                    while new_id in pain_point_ids:
                        id_counter += 1
                        new_id = f"PP-{id_counter:03d}"
                    pain_point['id'] = new_id
                    id_counter += 1

                pain_point_ids.add(pain_point['id'])

                # Add document reference
                pain_point['source_document'] = {
                    'id': doc_id,
                    'name': doc_name
                }

                # Assign hierarchy level if not present
                if not pain_point.get('hierarchy_level'):
                    description = pain_point.get('description', '').lower()
                    if any(term in description for term in ['system', 'policy', 'government', 'budget', 'national']):
                        pain_point['hierarchy_level'] = 'Macro'
                    elif any(term in description for term in ['local', 'individual', 'resident', 'community']):
                        pain_point['hierarchy_level'] = 'Micro'
                    else:
                        pain_point['hierarchy_level'] = 'Meso'

                all_pain_points.append(pain_point)

        # Create aggregated stakeholders list
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

        # Sort stakeholders by mentions (descending)
        aggregated_stakeholders.sort(key=lambda x: x.get('mentions', 0), reverse=True)

        # Create multi-level ecosystem representation
        hierarchy_relationships = self._generate_relationships(aggregated_stakeholders)

        multi_level_ecosystem = {
            "ecosystemMap": {
                "name": "Multi-Level Ecosystem Analysis",
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

        # Create final aggregated result
        aggregated_result = {
            'total_documents': len(results),
            'stakeholders': aggregated_stakeholders,
            'factors': all_factors,
            'pain_points': all_pain_points,
            'ecosystem_map': multi_level_ecosystem,
            'relationships': hierarchy_relationships
        }

        return aggregated_result

    def _generate_relationships(self, stakeholders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate relationships between stakeholders based on hierarchy levels"""
        hierarchy_relationships = []
        relationship_id = 1

        # Create relationships between macro and meso levels
        macro_stakeholders = [s for s in stakeholders if s.get('hierarchy_level') == 'Macro']
        meso_stakeholders = [s for s in stakeholders if s.get('hierarchy_level') == 'Meso']
        micro_stakeholders = [s for s in stakeholders if s.get('hierarchy_level') == 'Micro']

        # Limit the number of relationships to prevent explosion
        max_relationships_per_level = 20
        relationship_count = 0

        for macro in macro_stakeholders:
            for meso in meso_stakeholders:
                if relationship_count >= max_relationships_per_level:
                    break

                # Determine relationship type based on categories
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
                relationship_count += 1

        # Reset counter for next level
        relationship_count = 0

        # Create relationships between meso and micro levels
        for meso in meso_stakeholders:
            for micro in micro_stakeholders:
                if relationship_count >= max_relationships_per_level:
                    break

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
                relationship_count += 1

        return hierarchy_relationships

    def _print_extraction_summary(self, result: Dict[str, Any], doc_name: str) -> None:
        """Print a summary of the extraction results"""
        if result.get('error'):
            print(f"Error processing {doc_name}: {result.get('error')}")
            return

        stakeholders = result.get('stakeholders', [])
        factors = result.get('factors', [])
        pain_points = result.get('pain_points', [])

        print(f"\nExtracted from {doc_name}:")
        print(f"- {len(stakeholders)} stakeholders")
        print(f"- {len(factors)} factors")
        print(f"- {len(pain_points)} pain points")

        # Print top stakeholders
        if stakeholders:
            print("\nTop stakeholders:")
            for i, s in enumerate(stakeholders[:3]):
                print(f"{i + 1}. {s.get('name')} ({s.get('category')}) - {s.get('confidence')}% confidence")

        # Print top factors
        if factors:
            print("\nTop factors:")
            for i, f in enumerate(factors[:3]):
                print(f"{i + 1}. {f.get('Label')} ({f.get('Type')})")
                if f.get('Description'):
                    desc = f.get('Description')
                    print(f"   {desc[:100]}{'...' if len(desc) > 100 else ''}")

        # Print top pain points
        if pain_points:
            print("\nTop pain points:")
            for i, p in enumerate(pain_points[:3]):
                print(f"{i + 1}. {p.get('id')} - {p.get('category')} ({p.get('confidence')}%)")
                if p.get('description'):
                    desc = p.get('description')
                    print(f"   {desc[:100]}{'...' if len(desc) > 100 else ''}")

    def _print_aggregation_summary(self, result: Dict[str, Any]) -> None:
        """Print a summary of the aggregation results"""
        stakeholders = result.get('stakeholders', [])
        factors = result.get('factors', [])
        pain_points = result.get('pain_points', [])

        print(f"\nAggregated Results:")
        print(f"- {len(stakeholders)} unique stakeholders")
        print(f"- {len(factors)} unique factors")
        print(f"- {len(pain_points)} unique pain points")

        # Print hierarchy breakdown
        macro_stakeholders = [s for s in stakeholders if s.get('hierarchy_level') == 'Macro']
        meso_stakeholders = [s for s in stakeholders if s.get('hierarchy_level') == 'Meso']
        micro_stakeholders = [s for s in stakeholders if s.get('hierarchy_level') == 'Micro']

        print("\nStakeholder Hierarchy Breakdown:")
        print(f"- Macro level: {len(macro_stakeholders)} stakeholders")
        print(f"- Meso level: {len(meso_stakeholders)} stakeholders")
        print(f"- Micro level: {len(micro_stakeholders)} stakeholders")

        # Print top stakeholders by mentions
        if stakeholders:
            print("\nTop stakeholders by mentions:")
            for i, s in enumerate(stakeholders[:5]):
                print(f"{i + 1}. {s.get('name')} ({s.get('category')}) - {s.get('mentions')} mentions")


# Create a simple function to easily call from other scripts
async def analyze_text(
        text: str,
        model: str = "openai/gpt-4o",
        output_dir: str = "output",
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze stakeholders, factors, and pain points in a text

    Args:
        text: The text content to analyze
        model: The LLM model to use
        output_dir: Directory for output files
        verbose: Whether to print progress information

    Returns:
        Dictionary containing the analysis results
    """
    analyzer = StakeholderAnalyzer(model=model, output_dir=output_dir)
    return await analyzer.analyze_text(text=text, verbose=verbose)


# Create a function to analyze multiple texts
async def analyze_multiple_texts(
        texts: List[Dict[str, str]],
        model: str = "openai/gpt-4o",
        output_dir: str = "output",
        verbose: bool = True,
        aggregate: bool = True
) -> Dict[str, Any]:
    """
    Analyze multiple texts and optionally aggregate the results

    Args:
        texts: List of dictionaries, each containing 'text', 'id' (optional), and 'name' (optional)
        model: The LLM model to use
        output_dir: Directory for output files
        verbose: Whether to print progress information
        aggregate: Whether to aggregate results across all texts

    Returns:
        Dictionary containing individual and aggregated results
    """
    analyzer = StakeholderAnalyzer(model=model, output_dir=output_dir)
    return await analyzer.analyze_multiple_texts(
        texts=texts,
        verbose=verbose,
        aggregate=aggregate
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        # Single text analysis
        text = """
        The healthcare ecosystem in the region includes several key stakeholders. The Department of Health (DOH) 
        serves as the primary regulator, setting guidelines for healthcare delivery. Local hospitals like City 
        General Hospital provide direct patient care, while pharmaceutical companies such as MediCorp supply 
        essential medications. Patient advocacy groups, including the Patient Rights Association, represent 
        consumer interests. Insurance providers like HealthFirst offer coverage plans, and competing hospital 
        networks such as Regional Health System are expanding their presence. Medical schools partner with 
        hospitals for training programs. Rising costs and access challenges remain significant pain points,
        especially for rural communities.
        """

        analyzer = StakeholderAnalyzer()
        result = await analyzer.analyze_text(text)
        print("\nAnalysis complete!")


    asyncio.run(main())