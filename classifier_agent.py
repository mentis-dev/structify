#!/usr/bin/env python3
import asyncio
import json
from pathlib import Path

# Import the analysis function from the improved module
from core.stakeholder_analyzer import analyze_workspace


async def process_multiple_workspaces():
    """Example of processing multiple workspaces in sequence"""
    workspaces = [
        "10f39e32-1633-469d-acbd-5a08c9cf122b"
    ]

    results = {}

    for workspace_id in workspaces:
        print(f"Processing workspace: {workspace_id}")

        # Call the function with custom parameters
        result = await analyze_workspace(
            workspace_id=workspace_id,
            model="openai/gpt-4o",
            output_dir="analysis_results",
            max_documents=5,  # Limit to 50 documents per brain
            verbose=True  # Show detailed output
        )

        results[workspace_id] = result

    # Save the combined results
    output_path = Path("analysis_results/combined_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Combined results saved to {output_path}")


async def batch_processing_example():
    """Example of using the analyzer in a batch processing script"""
    # Define your workspaces and configuration
    workspace_configs = [
        {"id": "5a08c9cf122b", "max_documents": 5, "max_brains": 2},
    ]

    # Process each workspace with specific settings
    for config in workspace_configs:
        result = await analyze_workspace(
            workspace_id=config["id"],
            max_documents=config["max_documents"],
            max_brains=config["max_brains"],
            output_dir=f"results_{config['id']}",  # Custom output directory per workspace
            verbose=True
        )

        # You can do additional processing with the results here
        if not result.get("error"):
            print(f"Successfully processed {config['id']}")

            # Example: Extract and process only stakeholders
            if result.get("aggregated_stakeholders"):
                stakeholders = result.get("aggregated_stakeholders", {}).get("stakeholders", [])
                # Do something with the stakeholders
                print(f"  Found {len(stakeholders)} stakeholders")
        else:
            print(f"Error processing {config['id']}: {result.get('error')}")


if __name__ == "__main__":
    # Choose which example to run
    asyncio.run(process_multiple_workspaces())
    # Or uncomment to run the batch processing example
    # asyncio.run(batch_processing_example())