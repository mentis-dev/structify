#!/usr/bin/env python3
import os
import asyncio
import argparse
import time
import json
from dotenv import load_dotenv

from enrichment.state import InputState
from enrichment.configuration import Configuration
from enrichment.graph import graph

def print_banner(text):
    """Print a banner with text centered"""
    width = 80
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width + "\n")

async def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Classify stakeholders and factors from documents in a workspace")
    parser.add_argument("workspace_id", help="ID of the workspace to process")
    parser.add_argument("--output-dir", default="output", help="Directory for output files")
    parser.add_argument("--max-documents", type=int, default=None, help="Maximum number of documents to process per brain (None for all)")
    parser.add_argument("--max-brains", type=int, default=None, help="Maximum number of brains to process (None for all)")
    parser.add_argument("--model", default="openai/gpt-4o", help="The LLM model to use")
    
    args = parser.parse_args()
    
    # Define the combined extraction schema for stakeholders and factors
    extraction_schema = {
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
    
    # Create the initial InputState
    initial_state = InputState(
        workspace_id=args.workspace_id,
        max_documents=args.max_documents,
        max_brains=args.max_brains,
        output_dir=args.output_dir,
        extraction_schema=extraction_schema
    )
    
    # Set up the configuration with the prompt in the main script
    config_instance = Configuration(
        model=args.model,
        prompt=(
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
        ),
        max_loops=100
    )
    
    # Convert the configuration to a dictionary
    config = config_instance.__dict__
    
    print_banner(f"STARTING STAKEHOLDER AND FACTOR ANALYSIS FOR WORKSPACE: {args.workspace_id}")
    print(f"Processing all brains: {'Yes' if args.max_brains is None else 'No, max ' + str(args.max_brains)}")
    print(f"Processing all documents: {'Yes' if args.max_documents is None else 'No, max ' + str(args.max_documents)}")
    
    start_time = time.time()
    
    # Run the workflow graph
    try:
        # Include recursion_limit in config to handle larger workspaces
        graph_config = {
            "configurable": config,
            "recursion_limit": 1000  # Setting a high recursion limit
        }
        
        final_state = await graph.ainvoke(initial_state, graph_config)
        
        print_banner("PROCESSING COMPLETED SUCCESSFULLY")
        
        # Print the final results
        if final_state.get("error"):
            print(f"Processing completed with error: {final_state.get('error')}")
        else:
            print(f"Processed {len(final_state.get('processed_documents', {}))} documents")
            
            # Print stakeholder results
            if final_state.get("aggregated_stakeholders"):
                stakeholders = final_state.get("aggregated_stakeholders", {}).get("stakeholders", [])
                print(f"Identified {len(stakeholders)} unique stakeholders")
                
                # Print top stakeholders
                if stakeholders:
                    print("\nTop stakeholders:")
                    for i, s in enumerate(stakeholders[:5]):
                        print(f"{i+1}. {s.get('name')} ({s.get('category')}) - {s.get('confidence')}% confidence")
            
            # Print factor results
            if final_state.get("factors"):
                factors = final_state.get("factors", [])
                print(f"\nIdentified {len(factors)} factors")
                
                # Print top factors
                if factors and len(factors) > 0:
                    print("\nTop factors:")
                    for i, f in enumerate(factors[:5]):
                        print(f"{i+1}. {f.get('Label')} ({f.get('Type')})")
                        if f.get('Description'):
                            desc = f.get('Description')
                            print(f"   {desc[:100]}{'...' if len(desc) > 100 else ''}")
                else:
                    print("\nNo factors were identified in the documents.")
            elif final_state.get("info") and "factors" in final_state.get("info", {}):
                factors = final_state.get("info", {}).get("factors", [])
                print(f"\nIdentified {len(factors)} factors")
                
                # Print top factors
                if factors and len(factors) > 0:
                    print("\nTop factors:")
                    for i, f in enumerate(factors[:5]):
                        print(f"{i+1}. {f.get('Label')} ({f.get('Type')})")
                        if f.get('Description'):
                            desc = f.get('Description')
                            print(f"   {desc[:100]}{'...' if len(desc) > 100 else ''}")
                else:
                    print("\nNo factors were identified in the documents.")
            else:
                print("\nNo factors were identified in the documents.")


            # Print pain point results
            if final_state.get("pain_points"):
                pain_points = final_state.get("pain_points", [])
                print(f"\nIdentified {len(pain_points)} pain points")
    
                # Print top pain points
                if pain_points and len(pain_points) > 0:
                    print("\nTop pain points:")
                    for i, p in enumerate(pain_points[:5]):
                        print(f"{i+1}. {p.get('id')} - {p.get('category')} ({p.get('confidence')}%)")
                        if p.get('description'):
                            print(f"   {p.get('description')}")
                        else:
                            print("\nNo pain points were identified in the documents.")
            elif final_state.get("info") and "pain_points" in final_state.get("info", {}):
                pain_points = final_state.get("info", {}).get("pain_points", [])
                print(f"\nIdentified {len(pain_points)} pain points")
                    
                # Print top pain points
                if pain_points and len(pain_points) > 0:
                    print("\nTop pain points:")
                    for i, p in enumerate(pain_points[:5]):
                        print(f"{i+1}. {p.get('id')} - {p.get('category')} ({p.get('confidence')}%)")
                        if p.get('description'):
                            print(f"   {p.get('description')}")
                else:
                    print("\nNo pain points were identified in the documents.")
            else:
                print("\nNo pain points were identified in the documents.")
                
            print(f"\nFull results saved to {args.output_dir}/workspace_{args.workspace_id}_analysis.json")
        
    except Exception as e:
        print_banner("PROCESSING FAILED")
        print(f"Error: {str(e)}")
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal processing time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
